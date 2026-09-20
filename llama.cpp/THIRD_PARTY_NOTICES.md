# Third-party source notices

The vendored GGML/CUDA sources derive from ggml-org/llama.cpp and ggml-org/ggml. Their MIT license texts are preserved at `_vendor/llama.cpp/LICENSE` and `_vendor/llama.cpp/ggml/LICENSE`. Individual source files retain their original notices. The exact vendored sources, rather than a mutable upstream checkout, are included for reproducible compilation.

PTQ1_0 format definitions, SIMD ternary decoding, MMVQ, MMQ tile loading/configuration and conversion routines are selectively ported from https://github.com/PrismML-Eng/llama.cpp at revision `1a07bfa5f4144274c8f1c9963821dd9d9a51854b`. This port adds only PTQ1_0 to the existing WanGP kernels; unrelated upstream qtype and architecture changes are not imported. The adapted embedding lookup uses Prism's pair dequantizer. The signed FWHT wrapper is a local implementation of the metadata-defined normalized Sylvester transform. See `PRISM_PTQ1_PORT.md` for the source mapping.

SM120 async source uses the Triton/Gluon compiler API at build time. Triton is not bundled as a Python dependency in the wheel. GPU cubins and their build inputs are included under `src/llamacpp_gguf_cuda/kernels` and `csrc/sm120_async.py`.
