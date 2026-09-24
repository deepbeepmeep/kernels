# DFlash2 GPU acceptance — 2026-09-20

Follow-up 2026-09-22: [draft RoPE was corrected and both block predictors were remeasured](DFLASH2_ROPE_AND_DRAFT_COUNTS.md).
The timing comparison below predates that correction.

The common CUDA DFlash2 path now performs block acceptance on the GPU. Draft
length stays on the device through target verification, and the host receives
one completed block for scheduling, streaming and recurrent-state commit.
Intermediate draft-ID readbacks, per-position acceptance readbacks and dynamic
`nonzero()` candidate extraction are removed from this path.

## Scope and compatibility

- Enabled automatically for DFlash2 on CUDA in the existing vLLM integration.
  Qwen Q2/Q3/Q4 and Bonsai use the same acceptance implementation. No config
  keys or migration. The initial SM120 guard was removed after clarification:
  the request to defer other GPUs concerns kernel compilation/tuning, not
  availability of this portable PyTorch CUDA-graph path. Measurements below
  remain specific to the RTX 5090; no other GPU kernels were compiled or tuned.
- Greedy and sampled top-k 1–128 use fixed-shape graph operations. Target
  repetition penalties apply once per unique token, including hypothetical
  accepted prefixes. Suppression and thinking rules preserve their precedence;
  forced thinking closure and a close token inside a proposed block are handled
  on the GPU. Python processor state updates only for emitted tokens.
- Arbitrary/stateful processors, presence penalties, unsupported top-k settings
  and history-dependent draft processors retain the established sampler. A
  bounded candidate buffer includes top-k boundary ties; overflowing support or
  ambiguous ties at the nucleus boundary invoke the exact reference path.
  The decision to fall back examines the potentially reachable block and is
  independent of acceptance random numbers, preventing selection bias.
- Rejection still uses `min(1, p/q)` and samples the normalized positive part of
  `p-q` after rejection. Invalid draft suffixes are ignored, and correction,
  bonus and stop-token handling preserve the accepted-prefix cache contract.
- Random-number batching changes the sampled continuation for a given seed
  compared with the previous implementation. The target sampling distribution
  is preserved; exact greedy parity is validated. Repeated calls with the same
  new path and seed remain reproducible.
- Native MTP and DSpark retain their old acceptance logic. The shared change is
  extraction of that logic into an overridable helper. Draft anchor scalar
  writes also use GPU fills instead of synchronous CPU-to-GPU scalar copies.
- Acceptance graphs retain only bounded sampling buffers and are cleared with
  runtime/graph caches. Model weights, qtype kernels and vision residency policy
  are unchanged.

## Measurements

RTX 5090, uncensored Qwen Q4_K_M, original BF16 DFlash2 drafter, seven draft
tokens, two 2048-token fiction prompts with 256 sampled completion tokens each.
Warm graphs, Q8 KV, 32K capacity, temperature 0.6, top-p 0.95, top-k 20,
min-p 0.05, repetition penalty 1.05. Stage profiling disabled for throughput.

| Path | Decode tok/s | Output tokens / verification | Peak allocated GiB |
| --- | ---: | ---: | ---: |
| Previous acceptance | 84.35 | 2.317 | 21.810 |
| GPU block acceptance | 96.41 | 2.427 | 21.827 |

This short workload measured **14.3% higher throughput**, with approximately
18 MiB more peak allocated VRAM and 38 MiB more live allocated VRAM. Sampled
continuations and acceptance lengths differ because RNG draws are batched;
this is not an identical-token kernel comparison or a universal speedup.
Allocator-reserved memory can differ by more than live/peak tensor allocations.

In stage profiles, sampling/acceptance averaged about **0.72 ms** over 25
samples, versus **3.23 ms** in the preceding 27-sample audit. Target verification
remains the largest interval at about 17 ms. GPU-event intervals include idle
gaps and profiler overhead; CPU waits must not be added again to GPU time.

The final uninstrumented run, including warmup, used GPU acceptance in
**216/220 rounds**; four rounds took the exact sampler fallback. A 64-token
trace comparison recorded GPU-to-host transfers falling from **195 to 24**, and
`nonzero()` calls from **66 to 1**. The remaining exceptional calls come from
fallback/reference work; the ordinary block path has one host readback. That
trace preceded the final GPU-fill cleanup of anchor writes.

## Validation and reproduction

Real checkpoint checks passed for Q2, Q3, Q4 and Bonsai: CPU/CUDA default device,
cancellation/recovery, repeated greedy and sampled calls, snapshot/rewind and
exact greedy parity with reference acceptance. Greedy comparisons include a
thinking-budget transition. Q4 also passed real image inspection, vision
unloading before decoding, and restoration of the original text continuation.
Native MTP and DSpark passed real Q4 generation smoke tests after the refactor.

All 173 focused regression checks pass. They cover target distributions, top-k ties/overflow, rejection
sampling statistics, fallback independence from acceptance RNG, stop tokens,
invalid/empty draft prefixes, repetition, thinking masks, graph reuse and the
absence of scalar reads or dynamic candidate extraction in warm GPU acceptance.

Use `tools/audit_dflash2_runtime.py` with the usual benchmark arguments and
`--no-stage-profile`. Add `--legacy-acceptance` for the prior path; use
`--stage-every 8 --profile decode` separately for attribution. These overrides
are process-local and do not change application settings.

Durable raw results, checkpoint identities, runtime outputs and trace counts:
`benchmarks/dflash2_gpu_acceptance_2026-09-20.json`.
