# llamacpp-gguf-cuda

Reusable GGUF CUDA kernels packaged as a wheel.

This package exposes the unified GGUF CUDA path used in WanGP:
- `linear` with `auto/mmq/cublas` backend selection
- `embedding` for supported GGUF qtypes
- decode-only paged Q8 KV-cache attention derived from llama.cpp `fattn-vec`, with native FP16/BF16 I/O and FP32 accumulation

`q8_paged_attention` consumes INT8 K/V pages with one FP16 Q8_0 scale per 32 values. Its paged traversal and reduction adapt llama.cpp's vector attention structure to Nano-vLLM block tables, while Q8_1 query quantization and Q8_0 x Q8_1 `dp4a` products reuse llama.cpp CUDA primitives directly. It supports grouped-query attention, batched single-token decode, and causal multi-token speculative verification without materializing the full cache. It selects power-of-two split-K partitions with llama.cpp's occupancy and GPU-wave-efficiency heuristic, constrained by cache capacity and capped at 32 after SM89 graph-replay tuning. Temporary storage uses PyTorch and is safe to record and replay in CUDA graphs. Prompt prefill is intentionally outside this API.

## Build

### WSL / Linux

The repo includes a target-aware WSL build helper:

```bash
cd /mnt/e/ML/kernels/llama.cpp
chmod +x scripts/build_wsl_wheels.sh
./scripts/build_wsl_wheels.sh py310
./scripts/build_wsl_wheels.sh py311
```

Supported Linux targets:
- `py310`: conda `base`, Python `3.10`, PyTorch `2.7.1+cu128`, Linux CUDA toolkit `12.8`
- `py311`: conda `py311`, Python `3.11`, PyTorch `2.10.0+cu130`, env-local CUDA toolkit `13.x`

Set `MAX_JOBS` to control parallel compilation and `TORCH_CUDA_ARCH_LIST` if you want to narrow the generated fatbin.

### Windows

```powershell
cd E:\ML\kernels\llama.cpp
C:\Users\Marc\anaconda3\envs\py311\python.exe -m pip wheel . --no-build-isolation -w dist
```

By default the wheel builds a fatbin for every GPU code reported by the local CUDA toolkit `nvcc --list-gpu-code`.
Set `TORCH_CUDA_ARCH_LIST` explicitly if you want to override that and build a narrower wheel.
Set `LLAMACPP_GGUF_CUDA_BUILD_COMPONENTS=attention` to rebuild only the attention extension while repackaging an existing compatible MMQ/cuBLAS `_C` binary.

## Install

```powershell
C:\Users\Marc\anaconda3\envs\py311\python.exe -m pip install --force-reinstall --no-deps dist\llamacpp_gguf_cuda-*.whl
```

The wheel expects an existing CUDA-enabled PyTorch installation in the target environment.

## Runtime

Backend selection is controlled by `WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE`:
- `auto`
- `mmq`
- `cublas`

For user-facing selection, `WGP_GGUF_LLAMACPP_CUDA_MATMUL_MODE` accepts:
- `fast` (alias `materialized`): use MMQ for small workloads and materialize larger workloads to the requested FP16/BF16 dtype for cuBLAS
- `low_vram` (alias `mmq`): always multiply directly from packed GGUF weights without a dense weight temporary

The fast policy follows llama.cpp's existing NVIDIA threshold: MMQ below 64 input rows,
materialized cuBLAS at 64 rows and above. `mmq` describes the strict no-materialization
mode; its total runtime VRAM peak can still exceed fast mode when MMQ workspaces dominate.

The new variable takes precedence over `WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE`.
Materialized BF16 uses BF16 inputs and weights with FP32 accumulation.
Environment selection is cached at import. After changing one of these variables in
an existing Python process, call `llamacpp_gguf_cuda.refresh_env()` between generations.

Stream-K is enabled by default. `WGP_GGUF_LLAMACPP_CUDA_STREAM_K=0` disables only
Stream-K while retaining the packed-weight MMQ path; `1`, `on`, and `auto` enable it.
`WGP_GGUF_LLAMACPP_CUDA_STREAM_K_BUFFER_MB` sets the persistent per-device fixup
workspace in MiB and defaults to `16`. A value of `0` also disables Stream-K. The
workspace is allocated lazily by the first eager MMQ call, retained until module/process
shutdown, and reused during CUDA graph recording and replay. If a requested MMQ launch
needs more than the configured workspace, that launch transparently uses conventional
MMQ tiling. Both variables are cached by the same `refresh_env()` call and are never
read from the environment in the kernel call path.
Refreshing changes eager calls and subsequent graph captures; already-recorded CUDA
graphs must be recreated before they can reflect a new Stream-K setting.

In `auto` mode, BF16 output uses MMQ on supported GPUs. Set
`WGP_GGUF_LLAMACPP_CUDA_BF16_FP16=1` to restore the legacy behavior that
computes BF16 requests through the FP16 cuBLAS path. An explicit
`WGP_GGUF_LLAMACPP_CUDA_LINEAR_MODE=mmq` or `cublas` overrides this compatibility setting.

MMQ automatically applies per-row power-of-two scaling when a qtype stores Q8_1
partial sums in FP16. This prevents finite, high-range activations from overflowing
those sums without changing the quantized ratios.

FP16 and BF16 MMQ projections write their final values directly from the FP32 accumulator.
Stream-K keeps only its split-tile fixups in FP32, avoiding a complete temporary
FP32 output without changing the launch geometry.
