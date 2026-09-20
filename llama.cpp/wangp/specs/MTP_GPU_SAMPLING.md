# Native MTP GPU sampling

Native MTP now shares GPU block acceptance with DFlash2 and uses fixed-shape
GPU proposal filtering in vLLM mode. The implementation is independent of the
target quantization and has no SM120 gate. Legacy, CG, DSpark, arbitrary target
processors and unsupported top-k settings retain their established paths.

## Implementation

`engine/speculative_sampling.py` owns the common acceptance graph and native
MTP proposal filter. `ModelRunner` and `BlockDraftRunner` call the same helper;
the reference acceptance loop remains in `ModelRunner` for exact fallback and
A/B validation. DFlash2 still supplies its device-resident valid draft length.

The acceptance graph applies repetition, bias, suppression and thinking rules
to every possible accepted prefix. Its only host readback contains the completed
block. Ambiguous target nucleus ties or overflowing target top-k support use
the existing reference sampler; the fallback decision remains independent of
acceptance RNG. Processor state advances only for emitted tokens.

Native MTP's restricted draft vocabulary is copied into reusable full-vocabulary
probability buffers with a zero suffix. Mixed proposal widths, including a
thinking rule expanding a proposal, reuse the same acceptance graph safely.
There is no dequantized weight materialization or change to checkpoint precision.

The draft filter avoids dynamic `nonzero()` extraction. Its fixed-support tie
ordering may differ from the reference proposal, but rejection sampling uses
the exact returned normalized proposal probabilities. Consequently sampled
continuations for a fixed seed can change; the intended target distribution
is preserved. Greedy behavior retains exact target verification.
The measured improvement concerns sampled decoding. Greedy confidence scheduling
and draft chains that require host token IDs still use their established path.

Graph caches are bounded (four acceptance shapes, eight distribution shapes)
and released by the existing graph-cache/reset lifecycle. The new paths require
vLLM sampling and graph mode. No application config or settings migration is
needed. No new compiled kernel or architecture tuning was added.

## Reproduction

`tools/audit_mtp_runtime.py` now benchmarks the production implementation directly.
It no longer installs the prototype's state-sharing adapter. Without switches,
it selects the retained reference paths; `--gpu-acceptance --gpu-draft` enables
both production improvements. Use the benchmark arguments described in
`MTP_OPTIMIZATION_AUDIT.md`.

`tools/test_qwen_dflash2.py --method mtp` exercises the real shared loader,
CPU/CUDA defaults, cancellation, repeated generation, snapshot/rewind, greedy
parity and optional real image inspection/restoration. Its original DFlash2
mode remains available for regression checks.

The earlier prototype measurements remain unchanged in
`benchmarks/mtp_optimization_audit_2026-09-20.json`. Production validation and
measurements are recorded separately below.

## Validation and measurements — 2026-09-20

The focused suite passed **180 tests**, including target-distribution checks,
statistical rejection checks, thinking/repetition rules, RNG-independent fallback,
restricted/mixed draft vocabulary reuse, backend guards, and warm-graph profiling
without scalar reads or dynamic candidate extraction. Syntax checks also passed.

Real checkpoint checks passed for native MTP on **Bonsai PTQ1, Q2, Q3 and Q4**,
and for the refactored DFlash2 path on **Bonsai PTQ1 and Q4**. All covered CPU/CUDA
defaults, cancellation followed by successful generation, sampled output, greedy
parity including the thinking budget, and snapshot/rewind continuation. Actual
image inspection and restored text continuation passed for Bonsai MTP and Q4
DFlash2. Vision weights stayed on CPU outside inspection.

Fresh A/B measurements used an RTX 5090, PyTorch 2.10/CUDA 13, GGUF kernels 1.0.22,
two matched 2048-token prompts, 256 sampled tokens each, two MTP drafts, 32K cache
capacity and Q8 KV. Temperature 0.6, top-p 0.95, top-k 20, min-p 0.05, repetition
penalty 1.05, seed 123. Both paths were warmed; stage profiling was disabled.
Throughput below is total generated tokens divided by total decode time.

| Target | Reference MTP | Production MTP | Gain | Extra peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: |
| Bonsai PTQ1 | 122.85 tok/s | 134.70 tok/s | 9.6% | 2.27 MiB |
| Qwen Q4 | 134.51 tok/s | 144.64 tok/s | 7.5% | 7.60 MiB |

Mean cycle time fell from 17.44 to 15.64 ms for Bonsai and 16.84 to 15.32 ms for
Q4. Tokens per target pass changed from 2.142 to 2.107 and 2.265 to 2.216,
respectively. All eight timed completions retained a zero cache alignment delta.
The exact target fallback was exercised once on Bonsai and twice on Q4, including
warmup. Q2/Q3 compatibility was validated, but their speedups were not benchmarked.

These are short sampled workloads with different RNG scheduling/proposals between
paths, not guaranteed gains for every prompt or an identical-token comparison.
The fresh reference timings supersede comparisons against the earlier prototype
session's absolute speeds. Other GPUs were neither compiled for nor tuned.

Full runtime records, benchmark settings, binary identity and per-run results:
[`benchmarks/mtp_gpu_sampling_2026-09-20.json`](benchmarks/mtp_gpu_sampling_2026-09-20.json).
Raw prompts, completions and memory records remain under `tmp/mtp_integration/`.
