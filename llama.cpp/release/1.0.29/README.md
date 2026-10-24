# GGUF 1.0.29

Faster prefill for every quantization and faster speculative verification for
IQ2_S models, on NVIDIA GPUs of compute capability 8.0 and later. The decode
kernels of the other formats, the HIP builds and the Blackwell (SM120) defaults
are unchanged.

## BF16 output of the MMQ kernels

`linear(..., output_dtype=bfloat16)` with more rows than the MMVQ batch sizes
used to run llama.cpp's stream-k MMQ into an FP32 product and convert it
afterwards. The kernels now write BF16 themselves: complete tiles are rounded
once in the kernel; the FP32 partial sums of tiles split between blocks keep two
fixup slots per block (the tail a block finished, the head it left) and the fixup
kernel rounds their sum once, in the order of the FP32 path. The results are bit
for bit those of the FP32 product rounded to BF16 (`tests/test_mmq_typed_output.py`,
18 quantization types plus PTQ1_0; complete, split and edge tiles). FP16 outputs
and HIP keep the FP32 product. `has_typed_mmq()` tells callers the path exists.

Per linear at 1024 rows the conversion pass was 2-12% of the time (12% on a
5120->34816 Q4_K gate|up projection). A 27B Q4_K_M prefill gains 5-6.5% end to
end, an IQ2_M prefill 5-6%.

## PTQ1_0 tiles up to 128 columns

The Ampere MMQ configuration of PTQ1_0 stopped at 64 activation columns per
tile, so every weight tile was decoded for half as many rows as the other
Q8_0-layout formats. It now has the 80, 96, 112 and 128 column tiles of Q8_0
(same shared memory).

## Measurements

Prefill of a 27B model on a compute capability 8.9 GPU (tokens/s, 2K / 20K context, same prompts and settings):

| Checkpoint | 1.0.27 | 1.0.29 |
|---|---|---|
| Q4_K_M | 3383 / 2723 | 3602 / 2863 |
| IQ2_M | 3113 / 2533 | 3330 / 2671 |
| PTQ1_0 (Bonsai) | 1801 / 1604 | 2856 / 2382 |

Decode times per pass unchanged (the decode kernels are the same). Per PTQ1_0 linear at 1024 rows the MMQ went from 95-112 to
152-171 TOPS. Peak VRAM unchanged.

An IQ2_S short-batch tensor-core kernel was also built and measured for this release: slower than MMVQ at every shape and row count
on compute capability 8.9 (0.63-0.86x of MMVQ), so it is not part of the package.

## Wheels

| Platform | Python | PyTorch | Runtime | Native targets |
|---|---|---|---|---|
| Windows x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Linux x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Windows x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Linux x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Windows x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |
| Linux x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |

The CUDA native extensions were rebuilt from this source with `scripts/build_release.py`
in the 1.0.27 build workspaces: Ninja recompiled the objects whose sources or headers
changed (the MMQ, typed-output, short-batch and binding translation units) and reused the
others; the SM120 attention binaries are those of 1.0.28. `_C` grows by about 60% (the BF16
MMQ instances of every quantization type): 295 -> 462 MiB / 159 -> 257 MiB on Windows,
309 -> 477 MiB / 174 -> 275 MiB on Linux (CUDA 12.8 / 13). Linux wheels are built against
a glibc 2.28 sysroot with GCC 13, link libstdc++ statically (highest symbol version GLIBC
2.17) and carry only `$ORIGIN`-relative RPATHs. Every CUDA wheel passed
`scripts/inspect_wheel.py` (SASS and PTX inventory).

The HIP wheels keep the native extensions of 1.0.28, byte for byte: the 1.0.29 changes are
CUDA-only, so only their Python package (`has_typed_mmq()`, which reports False there),
version and metadata changed.

## Validation results

Each CUDA wheel passed `scripts/validate_wheel.py` in its own Python/PyTorch stack against
the published 1.0.28 wheel (previous formats bit for bit, GGUF, PTQ1, Hadamard, Prism
decode, linear fusions, row loop) and `tests/test_mmq_typed_output.py` (353 cases: the
typed BF16 MMQ output equals the FP32 product rounded to BF16 bit for bit). Results are in
`*_validation.json`.

The HIP wheels install and import under PyTorch 2.13.0+rocm10.0.0 on Windows and Linux,
with `has_typed_mmq()` False; no AMD GPU was available, so their kernels did not run.
Hardware: compute capability 8.9 (validation and measurements); a compute capability 7.5
GPU ran the package test suite (574 passed) and the prefill probes on a 1.0.29 build.
