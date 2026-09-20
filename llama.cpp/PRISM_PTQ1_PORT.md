# Prism PTQ1_0 port

Pinned upstream: https://github.com/PrismML-Eng/llama.cpp/tree/1a07bfa5f4144274c8f1c9963821dd9d9a51854b

| Upstream source | Merged functionality |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml-common.h` | ID 143, 128-value / 28-byte block layout |
| `ggml-cuda/common.cuh`, `vecdotq.cuh` | Type traits, SIMD base-3 decoding, DP4A, multi-column decode reuse |
| `ggml-cuda/mmvq.cu` | PTQ dispatch and two/three-column specialization |
| `ggml-cuda/mmq-load-tiles.cuh`, `mmq-config-ampere.cuh`, `mmq.cuh` | Shared tiles, tensor-core configurations, D4 scales and StreamK integration |
| `ggml-cuda/dequantize.cuh`, `convert.cu` | Pair decode and shared-transpose conversion |
| `src/llama-model.cpp` | Hadamard metadata and GDN permutation contract |

The existing row-first MMVQ loop is retained for old formats. The existing typed activation loaders, MMQ scratch pool, bias/output conversion and attention extension are reused. The Python API keeps all prior signatures and adds `prism_hadamard`. No full dequantized PTQ weights are allocated in the default packed linear/embedding paths. The explicitly selected `cublas` mode continues to materialize weights, as before.

Validation compares all 18 existing formats, both linear modes, FP16/BF16/FP32, batches 1/2/3/7/8/17/64/129 and existing embeddings: 870 outputs. PTQ-specific tests also cover 130-row tails, 128/384/512/5120 input widths, and changed-input CUDA graph replay. Hardware results are for RTX 5090 only; this is not a claim of universal GPU or full-model benchmark parity with Prism's separate llama.cpp runtime.

The accompanying `wangp/` snapshot also fuses compatible Bonsai QKV/gate and alpha/beta projections, prepares GDN output-row order during loading, and supports the separate adapted Q8 MTP head. These changes reuse the 1.0.22 native ABI; other quantization dispatch is unchanged. See `wangp/docs/BONSAI_PTQ1.md` for real-checkpoint validation, sidecar provenance, the official executable comparison and measured performance limitations. Two draft tokens were faster than four in the initial test, but the final sampled workload was slightly faster with MTP disabled. Do not equate these results with the model card's 129.9 tok/s PQ2_0 entry.

The SM120 vLLM path additionally fuses single-token GDN preparation and uses a validated 64-channel launch for the existing FLA four-tap convolution. This improves the paired target-only benchmark from 100.2/101.8 to 110.8/109.5 tok/s at 256/2048 prompt tokens without increasing peak VRAM. This is a Python/Triton integration change; the native wheel remains 1.0.22. The activation optimizations are now shared with Qwen 27B Q4_K_M, IQ3_S and IQ2_M without changing their checkpoint layouts. See `wangp/docs/QWEN_GDN_DECODE.md`. Prefill, speculative multi-token verification, other GPUs, other model sizes and cg/legacy retain their existing kernel paths.
