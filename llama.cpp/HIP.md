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
The CUDA packed-row kernels (1.0.26) and the single-launch row loop (1.0.27) are
not part of the HIP build: `linear_chunked_supported` returns False, so callers
keep their existing row slices.

## Version 1.0.27 target

The HIP wheels now target the ROCm 10 stack on Windows and Linux x86-64:
Python 3.12, PyTorch 2.13.0+rocm10.0.0 and the ROCm SDK 10.0.0 Python packages,
for gfx1200 and gfx1201 (RDNA4). The build rejects other PyTorch and Python
versions; the wheel records and checks the exact ROCm PyTorch build at import.
Wheels carry the `+torch213rocm100py312` version suffix. The earlier Python
3.11 / PyTorch 2.10.0+rocm7.14.0 wheels remain available in the 1.0.25 and
1.0.26 releases.

## Build environment

Use a dedicated Python 3.12 environment, separate from any NVIDIA environment:

```text
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.13.0+rocm10.0.0" "rocm-sdk-core==10.0.0" "rocm-sdk-devel==10.0.0" "rocm-sdk-libraries==10.0.0"
python -m pip install setuptools wheel ninja filelock numpy
rocm-sdk init
```

`rocm-sdk init` unpacks the compiler and device libraries into the environment
(`site-packages/_rocm_sdk_devel`); the build uses its clang and HIP headers.

## Reproduce on Windows

MSVC 2022 Community and the Windows SDK must already be installed. The script sets
their environment directly; it does not invoke vcvars64.bat:

```powershell
.\build_hip.ps1 -Python <environment>\python.exe -OutputDirectory <dist>
```

`-Architectures` overrides the default `gfx1200;gfx1201`. Other architectures
require matching SDK device assets and their own validation.

## Reproduce on Linux

The Linux wheel is compiled with the SDK's clang against GCC 13 libstdc++ headers
and a glibc 2.28 sysroot, so it loads on distributions older than the build
machine (highest symbol versions: GLIBC 2.14, GLIBCXX 3.4.21). Install the
toolchain with conda-forge and pass its sysroot to the HIP compiler:

```bash
mamba create -p /opt/gcc13-sysroot228 -c conda-forge gcc_linux-64=13 gxx_linux-64=13 sysroot_linux-64=2.28
T=/opt/gcc13-sysroot228
G=$(echo $T/lib/gcc/x86_64-conda-linux-gnu/13.*)
SDK=<environment>/lib/python3.12/site-packages/_rocm_sdk_devel
export ROCM_HOME=$SDK HIP_PATH=$SDK PYTORCH_ROCM_ARCH='gfx1200;gfx1201'
export CC=$T/bin/x86_64-conda-linux-gnu-g++ CXX=$T/bin/x86_64-conda-linux-gnu-g++ LDSHARED="$T/bin/x86_64-conda-linux-gnu-g++ -shared"
export LLAMACPP_GGUF_HIP_EXTRA_FLAGS="--sysroot=$T/x86_64-conda-linux-gnu/sysroot -nostdinc++ -isystem $G/include/c++ -isystem $G/include/c++/x86_64-conda-linux-gnu -isystem $G/include/c++/backward"
export PATH=$SDK/bin:$SDK/lib/llvm/bin:<environment>/bin:$PATH
python setup.py bdist_wheel --dist-dir <dist>
```

`LLAMACPP_GGUF_HIP_EXTRA_FLAGS` is appended to both the host and HIP compiler
flags. As for the CUDA wheels, delete the `%{!static:-rpath <environment>/lib}`
entry from the conda-forge compiler's `specs` file; the build itself strips
build-machine RPATHs from the link.

## Outputs and hardware validation

Generated HIP sources live under `build/hip-src-torch213rocm100py312`; maintained
CUDA sources are not rewritten. HIP outputs use `build/hip-torch213rocm100py312`.
Install the wheel into an environment with the matching ROCm PyTorch and run the
validator:

```text
python -m pip install --no-deps <dist>/llamacpp_gguf_cuda-1.0.27+torch213rocm100py312-cp312-cp312-<platform>.whl
python tests/validate_hip.py --output <validation-directory>
```

The validator requires an accessible AMD GPU. It checks all original formats,
PTQ1_0, embeddings, bias, shape tails, signed Hadamard/permutation, fused Prism,
paged attention and graph replay against independent numerical references.
Complete model tests in WanGP are still required.

## Validation status

Native compilation of all three HIP extensions succeeded for gfx1200 and gfx1201
on Windows and Linux. The Windows wheel imports under PyTorch 2.13.0+rocm10.0.0
and reports the CUDA-only row loop as unsupported; the Linux wheel imports in the
same stack. No RDNA4 GPU was available, so AMD execution, numerical validation,
performance and peak VRAM remain unverified.
