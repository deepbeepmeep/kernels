# GGUF 1.0.24 release validation

Built and tested on 25 September 2026 on an RTX 5090, Windows and Ubuntu 22.04 under WSL. Other NVIDIA architectures were compiled and inspected, but not tested on hardware.

This release adds a short-batch INT8 tensor-core linear for Q4_K and PTQ1_0 with 2-8 activation rows (speculative verification), compiled for compute capability 8.0 and newer, with a runtime policy: `auto` (default) applies per-shape decisions recorded by the caller and otherwise enables the path only on compute capability 12.0; `native` keeps the 1.0.23 dispatch; `mma` forces the new path. `LLAMACPP_GGUF_SHORT_BATCH` overrides the mode at import. WanGP measures both kernels per GPU on the model's own weights before CUDA graph capture and records its choices. The HIP build does not include the new path.

| Target | PyTorch | CUDA / ROCm | SASS targets | Wheel SHA-256 |
|---|---|---|---:|---|
| win_py310 | 2.7.1+cu128 | 12.8 | 17 | `6eaee6fafdca33b21681aa4b431680a6fc158c6f446d8250a10f78629e97cb67` |
| win_py311 | 2.10.0+cu130 | 13.0 | 12 | `37664f8fb654b9be4c37db9222380bdf241b92637ff4842cff6308c9c1f3525e` |
| linux_py310 | 2.7.1+cu128 | 12.8 | 17 | `6e5c7a2b6e270802649462668a0e5e878215a60b9c244833b69730142b4265f7` |
| linux_py311 | 2.10.0+cu130 | 13.0 | 12 | `09c44b22026a2427c71c043877daa58bca7e7892eb330a749acff061a21be28b` |
| win_hip_py311 | 2.10.0+rocm7.14.0 | ROCm 7.14 | gfx1201 | `4ae0f41bf320d6c8a217716e09eccb6644ca04ca257fc9adf09418de503f4645` |

All three native extensions in every CUDA wheel passed the toolkit-wide SASS/PTX inventory check. With `LLAMACPP_GGUF_SHORT_BATCH=native`, each CUDA wheel passed bit-exact comparison of the 18 existing GGUF formats against the 1.0.23 wheel of the same stack, the standard GGUF/attention suite, PTQ1, Prism Hadamard/decode, and fused-path checks. With the default policy on the RTX 5090, 858 of 870 outputs of the compatibility snapshot stay bit-identical; the others are Q4_K at 2, 3, 7 and 8 rows (tensor-core path, maximum relative difference 1.7e-3). Each CUDA wheel also passed WanGP's 27 short-batch tests: accuracy against single-row MMVQ for 2-8 rows and three output dtypes, fused SiLU, CUDA graph replay, bit-exact application of recorded decisions, and the selection and cache logic. The HIP wheel was compiled and its three extensions imported under ROCm PyTorch 2.10.0; no AMD GPU was available.

Build notes:

- The Windows CUDA wheels and the HIP wheel were built while `setup.py` / `hip_build.py` still declared 1.0.23; their metadata was relabelled to 1.0.24 with `wheel unpack/pack` (the binaries contain no version string). Both files now declare 1.0.24; the Linux wheels were built with it.
- After building, `llamacpp_gguf_cuda/__init__.py` in all five wheels was replaced by the committed version, which validates short-batch policy arguments in Python. On the Linux builds a C++ check failure inside `_C` terminates the process instead of raising (also true of 1.0.23), so invalid values must not reach the native checks. Native binaries were not modified; all validation above ran on the final wheels.

The adjacent JSON files preserve compiler configuration, source hashes, architecture lists, binary hashes and per-check durations. RTX 5090 measurements of the new path are recorded in the WanGP overlay (`wangp/specs/`); they are not a claim for other GPUs.
