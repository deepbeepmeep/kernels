# GGUF 1.0.25 release validation

Built and tested on 25 September 2026 on an RTX 5090, Windows and Ubuntu 22.04 under WSL. Other NVIDIA architectures were compiled and inspected, but not tested on hardware.

This release speeds up the PTQ1_0 (Bonsai ternary) short-batch tensor-core path used by speculative verification. The tensor cores take the raw base-3 digits {0, 1, 2} and the exact integer activation sum of each 32-value block is subtracted afterwards; each thread extracts only the shared-byte digits its fragments use; and PTQ1_0 activations are quantized in fragment order, read with 16-byte loads. Outputs are bit-identical to 1.0.24. Q4_K and every other path are unchanged.

| Target | PyTorch | CUDA / ROCm | SASS targets | Wheel SHA-256 |
|---|---|---|---:|---|
| win_py310 | 2.7.1+cu128 | 12.8 | 17 | `f726609f3a8ec43d9034c208d594ee783aab8a0d017d0b80bbf357f551797d33` |
| win_py311 | 2.10.0+cu130 | 13.0 | 12 | `74e280baac174c128c9b29627fa756c14c865cdda8bff118a90e42cca3803ada` |
| linux_py310 | 2.7.1+cu128 | 12.8 | 17 | `be79df12b6151371300032861af28d7bfe1e0024b04f34e5240d03b4451c5829` |
| linux_py311 | 2.10.0+cu130 | 13.0 | 12 | `726938c8d09300c9846b80b532772b99f875528e6150b37a12c5929bf062a1b5` |
| win_hip_py311 | 2.10.0+rocm7.14.0 | ROCm 7.14 | gfx1201 | `b58bf0ac3bdcb7e3de375052d93a637072cdef23d63cc9294310cc4cd5537de4` |

The CUDA wheels were built incrementally in the 1.0.24 workspaces: only `csrc/short_batch_mma.cu` was recompiled for every architecture; the new `kPtq1Dest` table is present in each `_C`. All three native extensions in every CUDA wheel passed the toolkit-wide SASS/PTX inventory. With the default policy, each CUDA wheel passed bit-exact comparison against the published 1.0.24 wheel of its stack (existing GGUF formats, standard GGUF/attention suite, PTQ1, Prism Hadamard/decode, fused paths), 259 PTQ1_0/Q4_K short-batch outputs (1-9 rows, BF16/FP16/FP32, typed output, fused SiLU) were bit-identical to 1.0.24, and WanGP's 27 short-batch tests passed.

The HIP build does not compile the changed source; its 1.0.24 wheel was relabelled 1.0.25 (metadata and `version.py` only; native binaries unchanged from the validated 1.0.24 HIP build, no AMD hardware available).

RTX 5090, Bonsai PTQ1_0 with 4 MTP drafts, the same 12 generated texts (identical 2.38 accepted tokens per cycle): 13.23 ms per cycle and 179.8 tok/s with 1.0.24 and the previous WanGP code, 11.63 ms and 204.6 tok/s with the installed 1.0.25 wheel (+13.8%; the SM120 test build measured 11.46 ms and 207.6 tok/s). PTQ1_0 verification linears take 28% less time. Qwen3.8 Q4_K_M is unchanged.
