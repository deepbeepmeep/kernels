# Precompiled GPU kernels

Precompiled kernels and their source are available in the repository releases.

## GGUF CUDA kernels

[Version 1.0.28](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.28) provides Windows/Linux wheels for Python 3.11 / PyTorch 2.10 / CUDA 13 and Python 3.10 / PyTorch 2.7.1 / CUDA 12.8, plus experimental HIP wheels for Python 3.12 / PyTorch 2.13 / ROCm 10 (RDNA4, gfx1200/gfx1201). The GGUF matrix, Prism fused-decode and standard attention extensions cover the architectures supported by each toolkit. Large GGUF linears (Q4_K, Q5_K, Q6_K, Q8_0) can run as a single-launch row loop that is faster and lighter than row slices; packed rows crossing GGUF blocks have native kernels. PTQ1_0 is merged with the existing quantization formats; on Turing (SM75) its linears with 1 to 7 rows decode each weight block once for all rows. Precompiled SM120 async attention is included for RTX50xx; other architectures retain the shared implementations.

See [llama.cpp/README.md](llama.cpp/README.md) for the complete build instructions, architecture lists, runtime controls and tests. The source includes the vendored dependencies, SM120 compiler inputs and GPU binaries.
