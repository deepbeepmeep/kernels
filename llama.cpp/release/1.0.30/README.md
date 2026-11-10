# GGUF 1.0.30

Faster speculative decoding of IQ3_S / IQ2_S models, a cheaper draft step for
models that draft over a large vocabulary, and a fix for a read past the end of
some weights. Everything is in the Python package and in kernels compiled at
first use with NVRTC, the run-time compiler PyTorch installs (as the Turing
kernels of the previous releases): no compiler is needed on the user's machine,
and the native extensions of the six wheels are those of 1.0.29, byte for byte.

## IQ3_S and IQ2_S with 1 to 8 rows

llama.cpp's MMVQ decodes each block of these formats (grid lookups, signs) once
per activation row, so the rows of a speculative verification cost almost as
much as separate passes. The new kernels (`_iq_rows.cu`) decode each block once
for all the rows of a call:

- `dp4a`, from compute capability 6.1;
- `mma`, `mma2` and `mma2x2`, from 8.0: INT8 tensor cores with `cp.async`
  staging (`mma2` reads the next weight blocks during the products, `mma2x2`
  also shares the activations between two tiles of rows).

A row does not depend on the other rows of its call (bit-identical from 1 to 8
rows). The error against an FP64 reference is within 1% of MMVQ's on ordinary
inputs and far below it at the extremes, because the activation scale stays in
FP32: rows of 1e-30 and of 1e30 keep a finite, accurate result. The fused SwiGLU
input is supported.

Which variant serves a weight shape and a row count is the caller's measured
decision: `iq_rows_variants(device)`, `set_iq_rows_decision(...)` and
`iq_rows_revision()` (decisions hold for the revision they were measured with).
Without a decision the package uses a tensor-core variant on compute capability
12.0, where it was measured, and MMVQ elsewhere. `LLAMACPP_GGUF_IQ_ROWS=0` turns
the kernels off.

One weight read cache-cold inside a CUDA graph on a compute capability 12.0 GPU
(microseconds):

| Weight | Kernel | 1 row | 3 | 5 | 8 |
|---|---|---|---|---|---|
| IQ3_S 34816 x 5120 | MMVQ | 51.6 | 62.1 | 70.5 | 99.4 |
| | `mma2` | 52.5 | 54.2 | 55.4 | 58.8 |
| IQ3_S 5120 x 17408 | MMVQ | 26.7 | 31.0 | 39.4 | 52.4 |
| | `mma2` | 28.8 | 29.6 | 31.0 | 32.4 |
| IQ2_S 34816 x 5120 | MMVQ | 43.7 | 54.0 | 67.4 | 98.0 |
| | `mma2x2` | 47.6 | 51.2 | 52.3 | 55.7 |
| IQ2_S 5120 x 17408 | MMVQ | 22.4 | 30.5 | 39.2 | 52.9 |
| | `mma2` | 25.7 | 29.2 | 31.3 | 37.0 |

Speculative decoding of a 27B model with 4 drafts, the same passes replayed:
215 -> 236 tokens/s on an IQ3_S checkpoint and 211 -> 225 tokens/s on an IQ2
checkpoint. Other architectures were not measured: a caller that times the
variants on its own weights keeps MMVQ wherever it is faster.

## Draft-head candidates

A drafter that proposes tokens scores its whole output head at each draft,
although a few tokens hold nearly all the probability. Three functions let a
caller score a short list instead:

- `head_sign_bits(weight)`: a one-bit copy of the head's dense rows (a sign per
  weight) and a scale per row, 61 MiB for a head of 98,304 x 5120;
- `head_group_bits(bits, groups)`: the layout the kernel reads, by group of
  rows, so that each load is one contiguous run;
- `head_candidates(input, bits, scale, ids, groups, keep)`: for one hidden
  state, the `keep` best rows of each group by the one-bit copy
  (`_head_candidates.cu`, compute capability 5.0+; `has_head_candidates(device)`).

The caller then scores those rows exactly with the regular kernels, so the
values it uses are exact; only the choice of rows comes from the copy. On a head
of 98,304 rows with 1,024 candidates (256 groups of 4), a draft step goes from
247 to 69 microseconds on a compute capability 12.0 GPU. On 6,000 inputs of a
27B model's draft head, the head's best token is among the candidates in 99.9%
of the cases and the tokens left out hold 0.4% of its probability on average.
Decoding the same passes with 4 to 7 drafts gains 4.3% (Q4_K_M), 4.9% (IQ3_S)
and 4.5% (IQ2): the passes are as many, each one is shorter.

## Bounded batched reads

The batched kernel (MMQ, from 9 input rows on NVIDIA GPUs) takes a weight row by
steps of 256 values. A row of 32-value blocks (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
IQ4_NL) whose width is not a multiple of 256 ends inside a step, and the step
read on: into the next row, which is harmless, and for the last rows past the
end of the weight, where a value that is not finite gave NaN in the last output
features and unmapped memory an illegal access.

`linear` now serves such a weight in two calls of the same kernel: the last rows
from a copy followed by zero blocks, the other rows as a weight of their own.
`linear_chunked` leaves these weights to `linear`. Results are those of the
kernel; no output depends on the bytes after the weight any more
(`tests/test_batched_bounds.py`, the six formats, widths of 32 to 5120, with and
without bias). `has_bounded_batched_reads()` tells callers the package does this
itself. Formats of 256-value blocks, PTQ1_0 and widths that are multiples of 256
were never affected.

## Run-time compiled kernels

The two new kernels are CUDA sources in the package. The first use on a GPU
compiles them for that GPU's architecture with NVRTC and stores the result in
the user's cache folder; later sessions load it from there. They are eligible
only on NVIDIA GPUs where NVRTC loads from the PyTorch installation; elsewhere,
and in the HIP wheels, the package behaves as 1.0.29 (apart from the bounded
reads, which apply everywhere).

## Wheels

| Platform | Python | PyTorch | Runtime | Native targets |
|---|---|---|---|---|
| Windows x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Linux x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | SM50-SM120 (17 SASS) + PTX |
| Windows x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Linux x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | SM75-SM121 (12 SASS) + PTX |
| Windows x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |
| Linux x86-64 | 3.12 | 2.13.0+rocm10.0.0 | ROCm 10 | gfx1200, gfx1201 |

The six wheels are the published 1.0.29 wheels with the Python package of this
source: `__init__.py`, `_cuda_jit.py`, `version.py` and the new `_iq_rows.py`,
`_head_candidates.py` and, in the CUDA wheels, their `.cu` sources (the HIP
wheels carry no CUDA source, as before), plus the version and README in the
metadata. Every other file, the native extensions and GPU binaries
included, is byte-identical to 1.0.29.

## Validation results

Each CUDA wheel was validated in its own Python/PyTorch stack, Windows and Linux,
on a compute capability 12.0 GPU:

- `scripts/validate_wheel.py` against the published 1.0.29 wheel: the kernels both
  versions have are bit for bit the same (the multi-row IQ kernels are left out of
  that comparison: they are not MMVQ bit for bit), then the GGUF, PTQ1, Hadamard,
  Prism decode, linear fusion and row loop suites. Results are in
  `*_validation.json`.
- `tests/test_iq_rows.py`: 208 checks over the four variants and both formats
  (error against FP64 no larger than MMVQ's, rows independent of their call,
  weights at any alignment and at the end of their buffer, fused SwiGLU, zero,
  outlier, tiny and huge activations), no failure.
- `tests/test_head_candidates.py` and `tests/test_batched_bounds.py`: 50 passed.

The two kernel sources also compile, with the NVRTC of each stack, for every
architecture they are eligible on: SM50 to SM120 with CUDA 12.8 (13
architectures) and SM75 to SM120 with CUDA 13 (8 architectures), about 0.2 s
per kernel. Only compute capability 12.0 ran them; the speed figures above are
from that GPU.

The HIP wheels were checked without ROCm PyTorch or an AMD GPU: their Python
package compiles under Python 3.12 and, with the HIP backend declared and the
native extensions replaced by stand-ins, reports no multi-row IQ variant and no
draft-head candidates, sends IQ3_S linears to the native kernel and never asks
for NVRTC. Their native extensions, unchanged since 1.0.28, were not loaded for
this release.
