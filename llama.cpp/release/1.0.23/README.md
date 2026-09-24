# GGUF 1.0.23 release validation

Built and tested on 24 September 2026 on an RTX 5090, Windows and Ubuntu 22.04 under WSL. Other NVIDIA architectures were compiled and inspected, but not tested on hardware.

This release adds optional typed-output and SiLU-and-multiply fusion to packed short-batch GGUF MMVQ. The established path remains available. The source tree includes CUDA and HIP compatibility sources, vendored GGML changes, the Gluon source and build script for SM120 binaries, and a hashed WanGP kernel integration overlay. Existing HIP wheel 1.0.22 remains the published AMD build.

| Target | PyTorch | CUDA | SASS targets | Wheel SHA-256 |
|---|---|---|---:|---|
| win_py310 | 2.7.1+cu128 | 12.8 | 17 | `ba12a2c11fc6f2684a51e339681aef650e133c50df2aff5b91254adbc46e439f` |
| win_py311 | 2.10.0+cu130 | 13.0 | 12 | `d6b1f6ed0f551d6c2b565e6d66ec6bfc7f50103882de135f643c051442116507` |
| linux_py310 | 2.7.1+cu128 | 12.8 | 17 | `1c46c16d62957a41ff10d3800b15095a29afe68d4a01a7c5ad0562b9fe0aa66d` |
| linux_py311 | 2.10.0+cu130 | 13.0 | 12 | `380cca99e2639e0361ee8727b1f64cffd7a778de59784f385b94ec17228fdf1d` |

All three native extensions in every wheel passed the toolkit-wide SASS/PTX inventory check. Each wheel passed bit-exact comparison of the 18 existing GGUF formats against the 1.0.21 reference, the standard GGUF/attention suite, PTQ1, Prism Hadamard/decode, and the new fused-path numerical, bias, and CUDA-graph checks. Windows PyTorch 2.10 also passed 34 WanGP fusion mode tests and the saved native fusion baseline.

The adjacent JSON files preserve compiler configuration, source hashes, architecture lists, binary hashes, and per-check durations. No throughput gain is claimed by this release validation.
