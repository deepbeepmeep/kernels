# GGUF 1.0.26

MiniMax H3 reference-image projections can restore matrix rows that cross GGUF
quantization blocks. The CUDA package now owns the packed row implementations
previously kept in WanGP. `linear` dispatches these layouts before the native MMQ
alignment assertion, and `supports_linear_layout` exposes the capability check.
`packed_row_kernel` returns the package-owned custom op for callers such as
WanGP that already classify restored rows. Input validation runs inside the
custom op, avoiding repeated tensor-subclass interception of metadata checks.
Weights remain packed; only tiles decode into registers/shared memory, with no
dense weight matrix or matrix padding.

Supported crossing-row formats: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1,
Q5_0, Q5_1 and Q8_0. RTX 20/SM75 uses NVRTC and FP16 WMMA with FP32
accumulation; BF16 range is preserved inside tiles. RTX 30/40/50
(SM86/89/120) share the Triton implementation. The older PyTorch 2.7 stack
was checked with Triton 3.3; the current PyTorch 2.10 stack with Triton 3.6.
First use compiles/tunes kernels and subsequent calls reuse caches. Existing
aligned GGUF, PTQ1, Prism, attention and explicit cuBLAS paths retain their
native implementations. IQ/PTQ1 crossing-row layouts are not newly supported.

## Wheels

| Platform | Python | PyTorch | Runtime | Update |
|---|---|---|---|---|
| Windows x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | Package-owned packed rows |
| Linux x86-64 | 3.10 | 2.7.1 | CUDA 12.8 | Package-owned packed rows |
| Windows x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | Package-owned packed rows |
| Linux x86-64 | 3.11 | 2.10.0 | CUDA 13.0 | Package-owned packed rows |
| Windows x86-64, gfx1201 | 3.11 | 2.10.0+rocm7.14.0 | ROCm 7.14 | Version number only |

No native extension was recompiled. Each CUDA wheel preserves all 19 extension,
cubin and PTX payloads byte-for-byte from the checksum-verified 1.0.25 wheel.
The architecture inventories therefore remain unchanged: 17 native SASS targets
for CUDA 12.8, 12 for CUDA 13, plus the original PTX coverage.

The HIP wheel changes only `__version__`, the METADATA Version field,
dist-info directory and RECORD. Every other payload is byte-identical, including
Python code, native binaries and the exact PyTorch/ROCm import guards. It does
not acquire the CUDA packed-row kernels or support a different ROCm stack.

## Reproduce without native compilation

Download the corresponding published 1.0.25 wheels, then run:

```text
python scripts/repackage_release.py --old-wheels <download-directory> --dist <output-directory>
python scripts/relabel_hip_release.py --old-wheel <published-1.0.25-HIP-wheel> --dist <output-directory>
```

The scripts verify baseline wheel checksums, preserve and compare binary payloads,
regenerate RECORD, and write manifests. CUDA repackaging rejects native source or
compilation-setting changes. Full source builds remain available through
`scripts/build_release.py`; the version-only HIP release is reproduced with
the relabeling script.

Validate each CUDA wheel using its matching Python/PyTorch environment with
`scripts/validate_wheel.py`, followed by the three `tests/test_packed_*.py`
suites. These cover independent GGUF decoding references, numerical precision,
all ten formats, FP16/BF16/FP32, odd widths, batched/noncontiguous inputs,
zero/extreme scales and values, graph replay, malformed metadata, architecture
dispatch and output-only GPU allocation. The SM75 backend is also exercised on
the available RTX 5090 through driver JIT; this is not RTX 20 hardware timing.

Checksums, unchanged payload hashes and results are recorded in `summary.json`,
`repackage_manifest.json` and `hip_relabel_manifest.json`.

## Validation results

All four CUDA stacks passed 425 packed-kernel tests each (1,700 total), all
native GGUF/PTQ1/Prism/attention/fusion suites, and 870 bit-exact comparisons
against 1.0.25 per stack. The final wheel runtime payloads were checked against
the extracted packages used in the tests. WanGP passed 534 integration and
documentation checks after removing its local kernel copies. The relabeled HIP
wheel imported all three extensions under PyTorch 2.10.0+rocm7.14.0.

The complete 27-block H3 vision encoder produced bit-identical outputs before
and after the move. The following are median milliseconds from 40 measured,
alternating warm runs, using the model's real Q4_K_M vision weights. The
benchmark matches H3's `device_explicit` execution (no default device); Python
garbage collection is disabled during this component comparison. Both paths
have exactly the same peak allocated and reserved VRAM.

| Backend / dtype | Image size | Previous local kernels | Package kernels | Peak allocated / reserved MiB, both |
|---|---:|---:|---:|---:|
| Triton / BF16 | 256 | 59.41 | 54.77 | 383.97 / 418 |
| Triton / BF16 | 512 | 42.54 | 43.04 | 418.97 / 478 |
| Triton / BF16 | 1024 | 133.71 | 132.58 | 557.56 / 628 |
| Forced SM75 / FP16 | 256 | 53.90 | 55.41 | 383.97 / 418 |
| Forced SM75 / FP16 | 512 | 66.73 | 65.65 | 418.97 / 478 |
| Forced SM75 / FP16 | 1024 | 183.98 | 183.77 | 557.56 / 628 |

All paired-bootstrap 95% timing-ratio intervals include parity: no consistent
slowdown was detected. Short runs have substantial host timing noise; the raw
samples and intervals are retained in the migration JSON files. This does not
establish speed on physical RTX 20/30/40 cards, which were unavailable.

One complete H3 reference-image generation (12 steps, 384x256, 107 frames,
32 kHz stereo) completed in 36.1 seconds. Every decoded video/audio record is
bit-identical to the pre-move packed-kernel generation; see `generation.json`.
Every generation phase also has identical peak allocated and reserved VRAM.
The original packed-kernel versus historical dense-fallback measurements and
quality assessment are retained separately in `historical_baseline.json`.
