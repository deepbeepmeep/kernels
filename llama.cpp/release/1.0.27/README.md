# GGUF 1.0.27

Large GGUF linears no longer need one MMQ launch, one FP32 product and one copy
per row slice. `linear_chunked` runs the whole linear in a single cooperative
launch: row chunks are quantized to Q8 into a ring of three slots while other
SMs multiply the previous chunks, and the bias is added in the output type during
the tile write-back. In a CUDA graph a linear becomes one kernel node and one
memset node instead of several nodes per slice.

Every output tile accumulates its full K range in one block, so results do not
depend on the chunk size or the GPU and match ordinary MMQ tiles bit for bit.
The ring never uses more memory than the row slices it replaces.

- Formats: Q4_K, Q5_K, Q6_K and Q8_0, FP16/BF16 input and output, optional
  16-bit or FP32 bias.
- Eligibility (`linear_chunked_supported`) is structural only: tensor-core MMQ
  data layout (SM75 and newer), cooperative launch, shared memory and occupancy.
  There is no GPU or shape list.
- Callers choose where it is faster. WanGP times the row loop against its row
  slices once per layer shape, row range and GPU, and keeps the slices where
  they win. With an older wheel that lacks `linear_chunked`, WanGP keeps doing
  the row slices in Python as before.
- HIP builds report the row loop unsupported and keep the existing kernels.

## AMD ROCm 10

The experimental HIP wheels move to Python 3.12, PyTorch 2.13.0+rocm10.0.0 and
the ROCm SDK 10.0.0 packages, on Windows and Linux, for gfx1200 and gfx1201
(RDNA4). See `HIP.md` for the build. The previous Python 3.11 / PyTorch
2.10.0+rocm7.14.0 wheels remain in the 1.0.25 and 1.0.26 releases.

## Wheels

| Platform | Python | PyTorch | Runtime | Native targets |
|---|---|---|---|---|
| Windows x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Linux x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Windows x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Linux x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Windows x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |
| Linux x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |

All native extensions were recompiled from this source. Linux wheels are built
against a glibc 2.28 sysroot with GCC 13 and carry no build-machine RPATH: the
CUDA wheels link libstdc++ statically (highest symbol version GLIBC 2.17); the
HIP wheel needs GLIBC 2.14 and GLIBCXX 3.4.21. PyTorch itself requires glibc
2.28 on Linux.

## Validation results

Each CUDA wheel passed `scripts/validate_wheel.py` in its own Python/PyTorch
stack. The suite checks the GGUF, PTQ1, Hadamard, Prism decode and linear-fusion
kernels, and checks that previous formats match the published 1.0.26 wheel bit
for bit. It now also runs `tests/test_linear_chunked.py`: 30 cases
(four formats, BF16/FP16, no/16-bit/FP32 bias, 9,999-20,000 rows), CUDA graph
replay and eligibility. Results are in `*_validation.json`.

Wider correctness runs (126 cases at LTX-2 shapes up to 93,840 rows) are bit
identical to row-sliced MMQ, except in slices where llama.cpp itself chose
stream-K. There the error against an FP64 product stays within 0.2% of the
slices' error.

Graph-replay timings of the row loop against row slices (paired runs, Q4_K and
Q6_K, 23,460 and 93,840 rows):

| Shape (K->N) | Q4_K no bias | Q4_K bias | Q6_K no bias | Q6_K bias |
|---|---:|---:|---:|---:|
| 4096->4096 | -7.7 / -9.2% | -16.5 / -18.1% | -7.6 / -9.9% | -15.3 / -17.3% |
| 4096->16384 | -10.0 / -10.1% | -18.2 / -18.8% | -9.2 / -9.7% | -15.9 / -16.5% |
| 16384->4096 | -6.0 / -6.3% | -7.9 / -8.4% | -5.4 / -6.5% | -7.4 / -8.2% |
| 4096->2048 | -8.6 / -8.8% | -17.0 / -16.9% | -9.2 / -9.3% | -16.2 / -15.9% |

The per-call memory peak, output included, is 2-6 MiB below the row slices in
all 60 measured configurations. LLM prefill shapes (Q4_K, 1k-16k rows) are mostly
faster, by up to 15%. The long-K narrow 25600->5120 shape is up to 8% slower
between about 2k and 8k rows, which is why the choice stays with the caller.
Timings were taken on one GPU architecture; other GPUs can differ, and WanGP
measures each GPU before using the row loop.

The HIP wheels compiled for gfx1200 and gfx1201 and import under PyTorch
2.13.0+rocm10.0.0, with the row loop reported unsupported. AMD execution,
numerical validation, performance and peak VRAM remain unverified.

## Reproduce

Build each CUDA wheel with `scripts/build_release.py` in the matching Python
environment (see the README's "Reproduce a wheel" section, including the Linux
toolchain), and each HIP wheel as described in `HIP.md`. Validate a CUDA wheel
against the published 1.0.26 wheel of the same stack:

```text
python scripts/validate_wheel.py --wheel <1.0.27 wheel> --reference-wheel <1.0.26 wheel> --output <directory>
```
