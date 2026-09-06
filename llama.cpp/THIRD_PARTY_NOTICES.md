# Third-party source notices

The vendored GGML/CUDA sources derive from ggml-org/llama.cpp and ggml-org/ggml. Their MIT license texts are preserved at `_vendor/llama.cpp/LICENSE` and `_vendor/llama.cpp/ggml/LICENSE`. Individual source files retain their original notices. The exact vendored sources, rather than a mutable upstream checkout, are included for reproducible compilation.

SM120 async source uses the Triton/Gluon compiler API at build time. Triton is not bundled as a Python dependency in the wheel. GPU cubins and their build inputs are included under `src/llamacpp_gguf_cuda/kernels` and `csrc/sm120_async.py`.
