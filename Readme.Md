# Precompiled GPU kernels

Precompiled kernels and their source are available in the repository releases.

## GGUF CUDA kernels

[Version 1.0.21](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.21) provides Windows/Linux wheels for Python 3.11 / PyTorch 2.10 / CUDA 13 and Python 3.10 / PyTorch 2.7.1 / CUDA 12.8. The matrix and standard attention extensions cover the architectures supported by each toolkit. Precompiled SM120 async attention is included for RTX50xx; other architectures retain the shared implementations.

See [llama.cpp/README.md](llama.cpp/README.md) for the complete build instructions, architecture lists, runtime controls and tests. The source includes the vendored dependencies, SM120 compiler inputs and GPU binaries, and the WanGP engine source overlay.
