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
