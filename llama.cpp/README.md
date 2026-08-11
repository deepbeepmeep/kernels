# llamacpp-gguf-cuda

Reusable GGUF CUDA kernels packaged as a wheel.

This package exposes the unified GGUF CUDA path used in WanGP:
- `linear` with `auto/mmq/cublas` backend selection
- `embedding` for supported GGUF qtypes

## Build

```powershell
cd E:\ML\kernels\llama.cpp
C:\Users\Marc\anaconda3\envs\py311\python.exe -m pip wheel . --no-build-isolation -w dist
```

By default the wheel builds a fatbin for every GPU code reported by the local CUDA toolkit `nvcc --list-gpu-code`.
Set `TORCH_CUDA_ARCH_LIST` explicitly if you want to override that and build a narrower wheel.

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
