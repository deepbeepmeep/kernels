# llamacpp-gguf-cuda 1.0.21

GGUF CUDA linear, embedding and paged-attention kernels for WanGP. The source tree contains the complete vendored llama.cpp/GGML implementation and the sources needed to reproduce the wheels.

## What is included

- Packed MMVQ for decoding and short speculative batches, using llama.cpp's architecture-aware selection; packed MMQ for larger batches. Neither requires dense weight materialization.
- MMVQ row-loop reuse and direct FP16/BF16/FP32 activation quantization, avoiding a temporary FP32 input matrix.
- Maximum-tile MMQ activation padding for safe CUDA-graph replay. Output and scratch buffers avoid unnecessary zero fills.
- Native Q8 paged attention and experimental dense FP16/BF16 paged attention. The Q8 vector path selects up to 64 automatic splits on SM120-class devices and 32 on other devices.
- **Precompiled SM120 asynchronous-copy attention** for Q8 prefill and grouped decode/verification. Four cubins per CUDA major cover FP16/BF16 and prefill/grouped operation. Runtime query counts, heads, pages, context lengths and splits do not compile new variants. These use `cp.async`, not TMA. Native C++ launches the binaries on PyTorch's current CUDA stream, including during graph capture. This path has no runtime Triton dependency.
- The WanGP overlay in `wangp/` preserves the shared engine/kernel integration and tests. Its dispatch enables the new async path only in the vLLM backend on compute capability 12.0, with head dimension 256. Other GPUs retain the shared implementations; legacy/cg retain their existing native/PyTorch paths.

The GPU binaries contain the same high/low tensor-core arithmetic as the validated Gluon source. This release does not change checkpoint quantization or sampling settings. Only an RTX5090 was available for hardware validation; fatbin coverage is not a claim of testing on every GPU.

## Release targets

| Python | PyTorch | Toolkit used | Native SASS architectures |
|---|---|---|---|
| 3.10 | 2.7.1+cu128 | CUDA 12.8 | 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120 |
| 3.11 | 2.10.0+cu130 | CUDA 13.1 | 75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121 |

Both targets have Windows x86-64 and Linux x86-64 wheels, with PTX for the highest toolkit architecture. CUDA 13 no longer compiles pre-SM75 targets; use the CUDA 12.8 stack for those devices, subject to PyTorch's own GPU support. SM120 async binaries are separately bundled for CUDA 12 and 13 and are never selected on other architectures.

## Reproduce a wheel

Install the matching CUDA-enabled PyTorch, setuptools, wheel and ninja in the target Python environment, plus the indicated CUDA toolkit. Linux needs a compatible C++ compiler and Python development headers. Windows needs MSVC and the Windows SDK; setup configures their paths without calling vcvars64.bat.

Run the helper with the target environment's Python. Build workspaces must be outside the source directory. It records compiler versions, architectures, source hashes, timing and wheel checksums in `build_manifest.json` and compiler output in `build.log`.

```powershell
# Windows, Python 3.11 / PyTorch 2.10
python scripts/build_release.py --target py311 --cuda-home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" --work-dir D:\gguf-build\win311 --dist-dir D:\gguf-build\dist
# Windows, Python 3.10 / PyTorch 2.7 (run with that environment's Python)
python scripts/build_release.py --target py310 --cuda-home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" --work-dir D:\gguf-build\win310 --dist-dir D:\gguf-build\dist
```

```bash
# Linux / WSL; use the matching Python for each command
python scripts/build_release.py --target py310 --cuda-home /usr/local/cuda-12.8 --work-dir /tmp/gguf-build/linux310 --dist-dir /tmp/gguf-build/dist
python scripts/build_release.py --target py311 --cuda-home /path/to/cuda-13.1 --work-dir /tmp/gguf-build/linux311 --dist-dir /tmp/gguf-build/dist
```

The release helper intentionally clears `TORCH_CUDA_ARCH_LIST` to include every code reported by `nvcc --list-gpu-code`. Default compilation concurrency is one job with two NVCC threads; increase `--max-jobs` only if sufficient RAM is available. For a private, architecture-limited build, use `TORCH_CUDA_ARCH_LIST` with `python -m pip wheel . --no-build-isolation --no-deps` directly instead.

### Recompile the SM120 GPU programs

Ordinary wheel builds package the checked-in cubins and do not need Triton. To regenerate them after changing `csrc/sm120_async.py`, use a separate Python environment with Triton 3.6.0 (Gluon), and run both commands before building the wheels:

```bash
python scripts/compile_sm120.py --cuda-major 12 --ptxas /path/to/cuda-12.8/bin/ptxas
python scripts/compile_sm120.py --cuda-major 13 --ptxas /path/to/cuda-13.1/bin/ptxas
```

On Windows, use the toolkit's `ptxas.exe`. The script writes four cubins, their PTX, signatures, launch metadata and SHA-256 checksums under `src/llamacpp_gguf_cuda/kernels/sm120_cu12` and `sm120_cu13`. The cubins are GPU programs shared by Windows and Linux; the native launcher is built separately for each Python/PyTorch/platform ABI. The emitted programs require no global/profile scratch allocation.

## Install and validate

Install the wheel matching the existing environment without replacing PyTorch:

```bash
python -m pip install --force-reinstall --no-deps /path/to/matching.whl
python tests/validate_release.py --output validation.json --checkpoint /path/to/Qwen3.8-Q4_K_M.gguf
```

The checkpoint argument is optional. Tests cover 18 qtypes with nonzero weights, FP16/BF16 inputs, decoding through prefill batch sizes, embeddings, attention and repeated CUDA-graph replays with changed data. On SM120 they also exercise the precompiled path without importing Triton. The WanGP integration tests and real-model benchmark are included in the overlay.

## Runtime controls

`WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE` takes precedence over `WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE`:

- `fast`, `mmq`, `packed`, `low_vram`: packed MMVQ/MMQ.
- `materialized`, `dense`, `cublas`: explicitly materialize weights for cuBLAS.
- With no override, the packed path is selected. The older LINEAR_MODE variable accepts `mmq` or `cublas`; `auto` selects packed operation.

The Python wrapper reads these variables at each eager call. Changes affect new graph captures; existing graphs must be rebuilt to change their recorded operations. There is no `refresh_env()` API and no configurable Stream-K environment variable in this implementation.

Packed operations accumulate into an FP32 output and cast to the requested dtype. MMQ keeps a reusable Stream-K workspace, sized from the device's SM count and rounded to 16 MiB; `prepare_runtime_buffers()` allocates it before graph capture, and `release_runtime_buffers()` releases it only when no graph using it remains live. It does not allocate a dense weight cache.

`q8_paged_attention` consumes INT8 K/V pages with one FP16 scale per 32 values, supports grouped-query attention and causal speculative verification. Its original vector path uses Q8_1 query quantization and dp4a products. The separate `sm120` interface supplies async tensor-core prefill and grouped partials; WanGP applies the shared reduction afterward.
