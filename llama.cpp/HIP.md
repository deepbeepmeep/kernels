# HIP build of the modified WanGP GGUF kernels

This builds the maintained sources, including local changes to GGML, rather than
substituting an upstream llama.cpp wheel. The Python package keeps its historical
`llamacpp_gguf_cuda` name because ROCm PyTorch also uses the `torch.cuda` API.

Implemented coverage:

- Existing 18 GGUF formats, typed activation quantization, packed MMVQ/MMQ,
  short batches, graph padding, bounded scratch storage and embeddings.
- PTQ1_0 base-3 SIMD decoding, multi-column MMVQ reuse, D4 MMQ tile loading,
  AMD MMQ configuration tables and odd-width handling.
- Signed Prism Hadamard transforms, GDN permutation and the fused Prism decoder.
- Native Q8 and dense paged attention, speculative batches, split reductions,
  HIP integer dot products and FP16/BF16 conversions.
- AMD architecture identification and logical 32-lane shuffle groups.

NVIDIA SM120 attention cubins are excluded. The shared attention implementations
provide the corresponding operations on AMD. WanGP's automatically selected GDN
and Prism fusions remain restricted to validated NVIDIA hardware; compiling their
AMD implementations does not establish numerical correctness or performance.

## Reproduce on Windows

The tested compiler environment is Python 3.11, ROCm 7.14 and ROCm PyTorch 2.10.0.
It matches WanGP's PyTorch 2.10.0 requirement in `docs/INSTALLATION.md`,
`setup_config.json` (recommended cu130 option), and the `py311` environment.
The build rejects other PyTorch versions; the wheel records and checks the exact
ROCm PyTorch build at import. It is separate from the NVIDIA environment. Example installation on D:

```powershell
python -m venv D:\AMD\qwen-hip210
D:\AMD\qwen-hip210\Scripts\python.exe -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "torch[device-gfx1201]==2.10.0+rocm7.14.0" "rocm[libraries,devel,device-gfx1201]==7.14.0"
D:\AMD\qwen-hip210\Scripts\python.exe -m pip install "setuptools<82" wheel ninja packaging numpy gguf pytest
D:\AMD\qwen-hip210\Scripts\rocm-sdk.exe init
.\build_hip.ps1
```

MSVC 2022 Community and the Windows SDK must already be installed. The script sets
their environment directly; it does not invoke vcvars64.bat. Its `-Python`,
`-Architectures` and `-OutputDirectory` parameters override the defaults. Other
architectures require matching SDK device assets and their own validation.
Only `gfx1201` was compiled for the supplied wheel.

Generated HIP sources live under `build/hip-src-torch210rocm714py311`; maintained CUDA sources are
not rewritten. HIP outputs use `build/hip-torch210rocm714py311` and the `+torch210rocm714py311` version suffix. Install
the wheel into an environment with matching ROCm PyTorch:

```powershell
D:\AMD\qwen-hip210\Scripts\python.exe -m pip install --no-deps D:\AMD\dist\llamacpp_gguf_cuda-1.0.24+torch210rocm714py311-cp311-cp311-win_amd64.whl
D:\AMD\qwen-hip210\Scripts\python.exe tests\validate_hip.py --output D:\AMD\validation
```

The validator requires an accessible AMD GPU. It checks all original formats,
PTQ1_0, embeddings, bias, shape tails, signed Hadamard/permutation, fused Prism,
paged attention and graph replay against independent numerical references.
The complete WanGP environment and real Qwen checkpoint tests are still required.
This wheel is not a binary match for the original report's ROCm 7.13 nightly.

## Validation status

Native compilation and loading of all three HIP extensions succeeded for gfx1201.
No AMD GPU is attached to the build machine, so AMD execution, full Qwen generation,
performance and peak VRAM remain unverified. See the accompanying WanGP
`specs/QWEN38_HIP_PORT.md` for the shared Triton changes and regression evidence.
