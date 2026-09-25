# RTX 50 short-batch tensor-core linear for Q4_K and PTQ1_0 (25 September 2026)

Speculative verification multiplies every weight matrix by 2-8 activation rows. On RTX 50 GPUs the native GGUF package now runs these Q4_K and PTQ1_0 linears on INT8 tensor cores with coalesced shared-memory staging. Precision is unchanged or better, weights stay packed and peak VRAM is unchanged.

| RTX 5090, Deepy thought path, 512 tokens | Cycle time, release → new | Median decode tok/s, release → new |
| --- | ---: | ---: |
| Qwen3.8 Q4_K_M, 4 drafts, 2K | 19.2 → 17.5 ms (-9%) | 144 → 153-156 |
| Qwen3.8 Q4_K_M, 4 drafts, 20K | 18.4 → 16.8 ms (-9%) | 146 → 147 |
| Bonsai PTQ1_0, 2 drafts, 2K / 20K | 16.1 → 12.3 ms (-24%) | 122 / 122 → 164 / 153 |
| Bonsai PTQ1_0, 4 drafts, 2K / 20K | 19.2 → 13.4 ms (-30%) | 125 / 116 → 195 / 171 |
| Bonsai PTQ1_0, no MTP (reference) | - | 119 / 110 |

Cycle time (decode seconds / target passes) is the robust metric: numerically different kernels change the sampled continuation and therefore acceptance. Qwen tok/s medians move with acceptance (176-219 passes per 512 tokens). Two release runs gave identical tokens and timings within 0.5%. Bonsai now gains 55-62% over its best previous configuration (no MTP), and four drafts beat two; before, MTP barely helped because verification was expensive.

## Diagnosis

Decode is GPU-bound: ~22 ms GPU time per ~23.5 ms cycle before this work, with 5-row Q4_K/Q6_K MMVQ taking ~15.5 ms. Cache-cold (24 layers cycled in a CUDA graph) measurements on the RTX 5090:

| Read of the same weight bytes | Bandwidth |
| --- | ---: |
| MMVQ-style access (each warp load = 8 rows x 32 B) | 1.12 TB/s |
| Coalesced 16 B loads along each row | 1.49 TB/s |
| Large contiguous stream (card ceiling) | 1.69 TB/s |

Native 5-row Q4_K MMVQ ran at 0.9-1.2 TB/s: the access pattern, not arithmetic, capped it. A first INT8-MMA prototype with the same scattered loads also stopped near 1.1 TB/s. With coalesced staging, a knock-out experiment showed the remaining limiter was re-reading activations per 16-row block (activation traffic ~56% of weight traffic); reusing each activation fragment for two MMA tiles removed it. L2-resident microbenchmarks (the 5090 has 96 MB L2) are misleading here and were not used for decisions.

PTQ1_0 has a second cost: MMVQ re-decodes its base-3 trits once per activation row.

## Implementation

`E:/ML/kernels/llama.cpp/csrc/short_batch_mma.cu` (first written as `blackwell_short_batch.cu`), dispatched from `run_linear_cuda` before the PTQ1 batch splitting and MMVQ selection. Conditions: Q4_K or PTQ1_0, 2-8 rows, `K % 256 == 0`, CUDA builds only. On by default for compute capability 12.0 (RTX 50 / RTX PRO Blackwell); other 8.0+ GPUs use it when WanGP measures it faster on the model's weights, see [GGUF_SHORT_BATCH_OTHER_GPUS_PLAN.md](GGUF_SHORT_BATCH_OTHER_GPUS_PLAN.md).

- Staging: each block copies its weight tile into shared memory with 16-byte `cp.async.cg`, double buffered. Row strides are padded so fragment reads are bank-conflict free. Q4_K: 32 rows x 4 split-K warps x 4 super-blocks per chunk (43 KB). PTQ1_0: 16 rows, 8 split-K warps when rows <= 6144 or K >= 16384, else 4 (measured best on every Bonsai shape).
- Arithmetic: `mma.sync.m16n8k32.s8` per 32-value activation block. Quad thread t owns contiguous values 8t..8t+7 of a block for both operands; the dot product is invariant to this shared k-permutation. Q4_K: a byte's low/high nibbles feed sub-blocks 2p/2p+1. PTQ1_0: each packed word is decoded once (the vendored multiply-by-three trit extraction) and mapped to fragments following the layout (bytes 0-15: values i+16n; 16-23: 80+i'+8n; qh: 120+b+2n). Integer dot products are exact; scales are applied in FP32 as in MMVQ. Warps reduce in a fixed order.
- Activations: quantized exactly like `quantize_mmvq_q8_1_typed` (d = amax/127, roundf), including fused SiLU-multiply rounding. The Q4_K minimum term uses d * sum(q), the exact equivalent of MMVQ's dp4a sum.
- Outputs: FP32/FP16/BF16 stored directly. Allocations: Q8 activations and scales only (a few hundred KB); no weight copies or persistent buffers.

Not retained: a Q6_K version (native Q6_K already reads 1.30-1.67 TB/s; the 210-byte blocks and 16-value scales made the MMA variant 4-35% slower) and single-row use (native MMVQ reads rows contiguously and was 5-8% faster). One row, 9+ rows and all other formats keep the release dispatch.

## Validation

- Real checkpoint matrices, two processes (candidate package first on `PYTHONPATH` vs release): 150 cases over Q4_K (`ffn_gate`, `ffn_down` including fused SiLU, `attn_k`) and PTQ1_0 (`ffn_gate`, `ffn_down`, `attn_k`, output head), rows 1-9, BF16/FP16/FP32 outputs. Error vs dense FP32 reference at most 0.45% above release (reduction order); at 8 rows Q4_K improves from 1.27e-2 (release MMQ) to 5.6e-3. Rows 1 and 9 byte-identical to release. CUDA graph replay equal to eager.
- `tests/test_gguf_short_batch_linear.py` (19 tests, synthetic Q4_K/PTQ1_0 weights): for 2-8 rows the batched error must not exceed per-row single-row MMVQ error; fused SiLU; graph replay.
- GGUF fallback, backend isolation, Prism decode/GDN and fused SiLU suites pass with the new binary. Three `test_prism_gguf.py` failures are pre-existing (the fixture lacks `model.config`) and occur with the release binary too.
- End-to-end: two alternating release/candidate process pairs for Qwen Q4, and Bonsai at 0/2/4 drafts; metadata records the loaded `_C` path for every run.

Measurement pitfall found: `tools/benchmark_qwen38_engine.py --kernel-library` never replaced the native module. CPython returns the already-imported `llamacpp_gguf_cuda._C` extension object, so candidate and release ran the same code. The option now fails loudly; comparisons must use a package copy placed first on `PYTHONPATH` in a separate process. Earlier results that relied on `--kernel-library` should be re-checked.

## Host overhead (investigated, not changed)

A torch-profiler trace suggested ~2 ms of GPU idle per cycle, but the profiler inflates host time. Direct instrumentation of the Deepy thought path (new kernels, 4 drafts) measured a median 0.55-0.63 ms from the acceptance readback to the next draft launch, which includes ~0.3 ms of MTP-refresh GPU work: roughly 0.25-0.5 ms idle, 2-3% of a 17.5 ms cycle. Means were ~1 ms because of occasional pauses; Python's garbage collector ran zero collections during decode and `gc.freeze()` changed nothing. Hiding the remaining gap would require launching the next cycle's drafts before scheduler bookkeeping (repetition history, KV block allocation), which is not justified by 2-3%.

## Status and limits

- Local installation: the SM120-only `_C` built with `TORCH_CUDA_ARCH_LIST=12.0` (CUDA 13.1) replaces the py311 release `_C`; the release binary is kept beside it as `_C.cp311-win_amd64.pyd.release1023` and in `C:/temp/gguf_rtx50/`. Release candidate 1.0.24 wheels compile `short_batch_mma.cu` for every toolkit architecture (PyTorch 2.10/CUDA 13 and PyTorch 2.7.1/CUDA 12.8); nothing is published.
- Measured on one RTX 5090. Other RTX 50 cards run the same SM120 code; their speed-up is unmeasured. Rows 2-8 only; draft depths above 7 fall back to MMQ for the 9-row verification.
- Prefill, single-token decode and Q6_K are unchanged.
