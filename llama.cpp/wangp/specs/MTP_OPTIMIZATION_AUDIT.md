# MTP optimization audit — 2026-09-20

Historical prototype results; the subsequent production integration is recorded
in [MTP_GPU_SAMPLING.md](MTP_GPU_SAMPLING.md). The audit command now selects the
production/reference paths directly instead of installing the original adapter.

The next useful optimization is shared MTP sampling, before additional PTQ1
kernel work. A process-local prototype on the RTX 5090 improved this short
workload by 12.1% for Bonsai PTQ1 and 7.3% for Qwen Q4. Production MTP dispatch
and application settings were not changed. No CUDA kernels were compiled.

## Experiments

Two matched 2048-token fiction prompts, 256 sampled completion tokens each,
two MTP drafts, warm graphs, 32K capacity and Q8 KV. Temperature 0.6, top-p 0.95,
top-k 20, min-p 0.05, repetition penalty 1.05. Stage profiling is disabled for
these measurements. Target/drafter checkpoints retain their existing precision.

| Target | Current MTP | GPU acceptance only | GPU acceptance and draft filtering |
| --- | ---: | ---: | ---: |
| Bonsai PTQ1 | 104.31 tok/s | 109.49 tok/s | **116.94 tok/s** |
| Uncensored Qwen Q4 | 117.93 tok/s | 116.56 tok/s | **126.55 tok/s** |

Acceptance alone is not a reliable throughput improvement on both targets.
The combined change is more promising. Average cycle time fell from 20.54 to
18.02 ms on PTQ1 and from 19.21 to 17.52 ms on Q4. Output tokens per target pass
changed from 2.142 to 2.107 on PTQ1 and from 2.265 to 2.216 on Q4.

Peak allocated VRAM was essentially unchanged: PTQ1 7.787 to 7.789 GiB; Q4
17.484 to 17.492 GiB. No model weights were expanded or converted.

Random-number scheduling and proposal tie handling change the sampled
continuation for a fixed seed. These are short workload measurements, not
identical-token kernel comparisons or guaranteed gains on every prompt.

## Shared improvements

1. **Reuse GPU block acceptance for native MTP.** Native MTP still executes the
   reference Python acceptance loop. The prototype reuses the tested DFlash
   acceptor through an audit-only adapter. It pads the restricted MTP proposal
   vocabulary with zero probability for target verification, preserving the
   residual sampling support. Production work should extract a common helper,
   rather than retaining the adapter or introducing MTP/DFlash inheritance.
2. **Remove dynamic extraction from MTP draft sampling.** The current draft
   sampler uses `nonzero()` after top-k. Even when selected draft IDs stay on
   the GPU, dynamically sized candidate extraction synchronizes. The prototype
   uses fixed-shape candidate filtering inside a graph. The proposal's exact
   computed q is supplied to rejection sampling; the target filter still uses
   its exact fallback for ambiguous ties/overflow. Proposal tie ordering can
   change without changing the intended target distribution.
3. **Combine draft forward and sampling graphs after correctness validation.**
   They currently replay separately. Capturing the common fixed-depth chain
   can reduce launches and copies further. Preserve predictor cache growth,
   masks, penalties, seed behavior and cancellation; retain the reference path
   for unsupported processors. This remains unimplemented and unmeasured.
4. **Avoid blocking accepted-token uploads during MTP refresh.** Multi-token
   `_advance_mtp` copies a newly created CPU tensor into its GPU input buffer
   with a blocking copy. Existing pinned input buffers or device-resident
   accepted IDs can remove that handoff. The prototype does not change it.

These improvements are not intrinsically tied to a target qtype or SM120.
Only the RTX 5090 was measured; no other GPU compilation/tuning was performed.
Q2/Q3 benefit is plausible from the shared sampler, but was not benchmarked
in this audit. A production change still needs repeated generation, exact
greedy parity, statistical sampling checks and state/vision restoration across
the supported target variants and processor fallbacks.

## PTQ1-specific findings

The three-row PTQ1 matrix-vector kernel accounts for approximately **65% of GPU
kernel time** in the 64-token trace. With two drafts, the target verifies the
anchor plus two proposals. This is the primary quantization-specific bottleneck.

The current source already shares PTQ1 unpacking across two or three activation
rows (`vec_dot_ptq1_0_q8_1_multi` in the bundled GGUF kernels). Recommending that
as a new optimization would duplicate existing work. Further useful work needs
measurements of register pressure, occupancy and matrix shapes, then comparison
of specialized small-batch implementations on the current GPU.

The fused Prism decode path is enabled only for a single activation row.
Multi-token verification uses separate Hadamard transformation and packed
matrix multiplication. Extending fusion to the verification batch is possible,
but Hadamard kernels account for only about **2.3%** of this trace, so eliminating
those launches alone cannot produce a large speedup. The packed dot products
must become cheaper for a substantial additional gain.

Bonsai's MTP sidecar itself is Q8_0. Q8_0 matrix-vector kernels account for only
about 3.6% of GPU time in this trace. More aggressively quantizing that small
predictor is therefore a lower priority and could reduce acceptance quality.

MTP refresh already projects logits only for its last token. Refreshing the
accepted prefix is necessary because the predictor must use verified target
hidden states; simply reusing speculative predictor states would change context.

## Reproduction and limits

`tools/audit_mtp_runtime.py` wraps the real benchmark with process-local
overrides. Supply ordinary `benchmark_qwen38_engine.py` arguments with
`--method mtp --draft 2 --no-stage-profile`. Add `--gpu-acceptance`, then
`--gpu-draft`, for the two experiments. The adapter is an investigation tool,
not the production implementation.

Raw results, checkpoint/binary identities and experiment settings are retained
in `benchmarks/mtp_optimization_audit_2026-09-20.json`. Prompts, generated text and
trace files are under `tmp/mtp_audit/`. All six uninstrumented runs generated
real output with zero predictor/target cache alignment delta.

The initial per-stage telemetry passes a profile object into distribution
filtering, which disables its existing sampling graph. Consequently those
stage means overstate normal sampling overhead and were not used to estimate
the gain. The optional `--timing-only` wrapper avoids this telemetry behavior;
reported speedups use uninstrumented end-to-end decode timings instead.
