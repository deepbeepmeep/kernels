# GGUF 1.0.28

PTQ1_0 models (Bonsai and other Prism-folded checkpoints) decode faster on Turing
GPUs (SM75: RTX 20XX, GTX 16XX, T4) when a linear receives 1 to 7 activation
rows, as in token-by-token decoding and speculative (MTP) verifications.

On these GPUs llama.cpp's PTQ1 MMVQ kernel decodes the ternary weights of each
block again for every row, and the tensor-core short-batch path needs SM80.
`linear` now sends PTQ1_0 calls with 1 to 7 rows on SM75 to a kernel compiled at
run time with NVRTC (`_ptq1_turing.cu`, like the existing packed-row Turing
kernel):

- the same Q8_1 activation quantization, base-3 decoding, DP4A products and
  per-block float operations as MMVQ;
- each weight block decoded once for all the rows;
- every row computed in the same order whatever the row count, so the rows of a
  verification equal single-row results bit for bit;
- a launch configuration tuned once per shape and cached.

Other GPUs, the other formats and the HIP builds keep their kernels. The native
extensions are byte-identical to those of 1.0.27: only the Python package and the
new kernel source differ.

## Measurements (SM75, BF16, CUDA graph replay)

Time per call in ms, MMVQ / new kernel:

| Linear (K->N) | 1 row | 2 rows | 5 rows | 7 rows |
|---|---:|---:|---:|---:|
| 17408->5120 | 0.192 / 0.059 | 0.211 / 0.062 | 0.378 / 0.085 | 0.785 / 0.189 |
| 5120->34816 | 0.115 / 0.087 | 0.295 / 0.093 | 0.678 / 0.182 | 0.945 / 0.242 |
| 5120->16384 | 0.056 / 0.046 | 0.139 / 0.049 | 0.329 / 0.094 | 0.459 / 0.121 |
| 5120->248320 | 0.791 / 0.539 | 2.014 / 0.588 | 4.864 / 1.189 | 6.796 / 1.634 |

Decoding a 27B PTQ1_0 model at a 2K context goes from 26.4 to 34.9 tokens/s
without drafts and from 24.9 to 50.5 tokens/s with two MTP drafts.

## Wheels

| Platform | Python | PyTorch | Runtime | Native targets |
|---|---|---|---|---|
| Windows x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Linux x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Windows x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Linux x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Windows x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |
| Linux x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |

Each wheel is the 1.0.27 wheel of its stack with the 1.0.28 Python package: the
native extensions, cubins and PTX are unchanged, so the toolchain, glibc and
driver requirements of 1.0.27 apply. The HIP wheels include the new routing
module but not the CUDA kernel source; SM75 eligibility is false there.

## Validation results

- On SM75 with real 27B PTQ1_0 weights of every linear shape and 1 to 7 rows:
  bit-identical to MMVQ for five of six shapes; the sixth (136 blocks per row)
  within 1.9e-6 from FP32 summation order, with the same error against an FP64
  product as MMVQ to seven digits. Bit-identical to the fused-output path with
  BF16 output; rows identical whatever the row count; every launch configuration
  identical; CUDA graph replay and repeated calls identical. The package tests
  pass with the route active.
- The kernel shipped in the wheels was also compiled to SM75 PTX and run through
  the driver's JIT on a newer GPU: 13 agreement, precision, invariance,
  configuration and graph checks pass.
- Each CUDA wheel passed `scripts/validate_wheel.py` in its own Python/PyTorch
  stack against the 1.0.27 wheel: previous formats bit for bit, standard, PTQ1,
  Hadamard, Prism decode, linear fusion and row-loop suites
  (`*_validation.json`).
- The HIP wheels import under PyTorch 2.13.0+rocm10.0.0 with the Turing route
  reported ineligible. AMD execution remains unverified.

## Reproduce

The wheels are the 1.0.27 release wheels unpacked, given the 1.0.28 `__init__.py`,
`version.py` and `_ptq1_turing.py` (plus `_ptq1_turing.cu` for CUDA), their
dist-info renamed with the new version and README, and packed again with
`python -m wheel pack`; every other file is byte-identical. A full build follows
`release/1.0.27/README.md` with `BASE_VERSION = "1.0.28"`.
