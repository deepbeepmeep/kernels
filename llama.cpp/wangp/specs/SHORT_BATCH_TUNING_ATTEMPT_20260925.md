# Short-batch speed attempt after 1.0.24 (25 September 2026, RTX 5090)

Goal: more decode speed for Qwen3.8 Q4_K_M and Bonsai PTQ1_0 speculative decoding without precision loss or extra VRAM.

## Kept (WanGP only, uncommitted at time of writing)

| Change | Qwen Q4 MTP-4 cycle | Bonsai MTP-4 cycle |
| --- | ---: | ---: |
| Baseline | 17.25 ms | 13.16 ms |
| Shared top-p tie rule: keep every token tied with the lowest kept nucleus value in all three samplers (`layers/sampler.py`, `ModelRunner._filter_speculative_distribution`, `dflash_sampling.target_probabilities`). GPU acceptance no longer falls back (4-12% of cycles before). | 17.08 ms | 12.98 ms |
| Typed BF16 output and fused SiLU-multiply for 2-8 row GGUF linears (`QLinearGGUF.forward`), wherever `supports_linear_fusions` allows (not 8 rows). Removes a PyTorch conversion kernel and the separate SiLU kernel per call; bit-identical. | 16.81 ms | 12.79 ms |

Total -2.6% (Qwen) / -2.8% (Bonsai). The tie rule is a deliberate, tiny change of sampling semantics; outputs remain exactly distributed under it. Typed output leaves the sampled tokens unchanged (identical tokens per cycle in A/B).

## Measured, no gain

- MTP draft depth 3/4/5/6: 4 is optimal for both models (expectation model from measured per-position acceptance 0.83/0.67/0.63/0.64 and 1.1-1.3 ms per extra draft).
- Tree speculation: sibling acceptance after a rejected draft, computed exactly with recursive rejection sampling on 187 real cycles, is 11-30% (third candidates 0-6%). Best 8-row tree: +5.5% tokens per cycle before costs, about +1-2% net.
- Frequency-ranked draft vocabulary: a 32K head saves 0.6 ms per cycle, but token IDs are frequency-ordered only up to ~16K (first 32K IDs cover 90-94% of text), and a static ranking would slow non-English users. About +3% with a context-aware set; not pursued.
- Kernel retuning (prototype `tools/experiments/short_batch_v2`, SM120): Q4_K warps x 16-row tiles x pipeline stages and PTQ1_0 warps x stages x units per warp, on the real shapes with cache-cold rotation. Production configurations are best on every shape except Q4_K 5120->1024 (8 warps, 1.16x, ~25 us per pass). An exact two-instruction int->float conversion instead of I2F changed nothing. PTQ1_0 knock-outs: removing trit decoding saves 12-26%, activation loads 8-24%, MMAs 0%; deeper pipelines are slower. PTQ1_0 runs 0.56-1.0 TB/s and is instruction- or shared-memory-bound, not latency-bound (63 registers, occupancy fine). Q4_K uses 126 registers and 43 KB shared memory (2 blocks, 8 warps per SM).
- Separate activation quantization costs 0.8-1.1 us per call inside a graph (~1-1.5% of a cycle); folding it into the matrix kernels was not attempted.

## Measurement pitfalls found

- An idle RTX 5090 runs GDDR7 at half clock (7001 vs 13801 MHz); short graph replays after host-side work time at the low clock. Warm up with ~150 ms of sustained replays.
- The first timing on freshly allocated weights can be up to 2x slow even after warm-up (first touch under Windows). Discard it, alternate configurations and take the fastest round.
- Nsight Compute needs GPU performance counters enabled for the user (NVIDIA Control Panel, Developer settings), which is a system setting; without it, knock-out experiments were the only diagnostic.

## PTQ1_0 kernel: instruction-count reductions (kept, kernel repository, uncommitted)

The SASS mix of the PTQ1_0 kernel is ~65% integer bit manipulation (1184 instructions; LOP3 330, IMAD 191, IADD 136, PRMT 90), and at the SM issue rate it reaches ~45% of its instruction-bound ceiling, so fewer instructions per weight is the lever. Three exact changes, measured in the prototype over the real Bonsai shapes (5 rows, cache-cold) and then in the production source:

| Change | PTQ1_0 time per Bonsai verify pass |
| --- | ---: |
| 1.0.24 | 8.50 ms |
| Raw base-3 digits {0, 1, 2} into the tensor cores; subtract the 32-value block's exact integer activation sum afterwards (removes `__vsub4` per four trits; quantization stores the integer sum for PTQ1_0) | 7.56 ms |
| Extract only the shared-byte digits each quad thread uses, directly: ((v * 3^n) mod 256) * 3 >> 8 | 7.27 ms |
| Quantize PTQ1_0 activations in fragment order (`kPtq1Dest`): two 16-byte loads per thread and block instead of eight scattered 4-byte loads; 16-byte scale loads | 6.10 ms (-28%) |

All 259 outputs of an exactness sweep (PTQ1_0 and Q4_K, 1-9 rows, BF16/FP16/FP32, plain, typed and fused SiLU) are bit-identical to 1.0.24. End to end with an SM120-only build (identical tokens per cycle, so identical text):

| Bonsai PTQ1_0, RTX 5090 | 1.0.24 | Candidate | Change |
| --- | ---: | ---: | ---: |
| MTP 4 cycle / speed | 12.94 ms / 175 tok/s | 11.35 ms / 199 tok/s | -12.3% / +14% |
| DFlash2 5 cycle / speed | 14.36 ms / 171 tok/s | 12.71 ms / 193 tok/s | -11.5% / +13% |

Qwen3.8 Q4 is unchanged (16.9 ms per cycle): the same permuted-activation change on the Q4_K kernel gave 0.97-1.01x, consistent with Q4_K being limited by memory latency and occupancy (126 registers, 43 KB shared memory) rather than instruction issue.

## Next step if pursued

Q4_K: occupancy (register and shared-memory footprint) for the medium matrices; confirm with Nsight Compute, which is now enabled (the first attempt still reported ERR_NVGPUCTRPERM with `RmProfilingAdminOnly=1`).
