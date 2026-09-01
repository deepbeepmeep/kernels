#include <torch/extension.h>

#include "gguf_llamacpp_ops.h"

#include <pybind11/stl.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supports_linear_qtype_name", &gguf_cuda_supports_linear_qtype_name, "Return whether the CUDA GGUF linear fast path supports the qtype.");
    m.def("supports_embedding_qtype_name", &gguf_cuda_supports_embedding_qtype_name, "Return whether the CUDA GGUF embedding fast path supports the qtype.");
    m.def("supports_qtype_name", &gguf_cuda_supports_qtype_name, "Return whether the CUDA GGUF fast path supports the qtype.");
    m.def("release_runtime_buffers", &gguf_cuda_release_runtime_buffers, "Release reusable CUDA runtime scratch buffers after graph teardown.");
    m.def("prepare_runtime_buffers", &gguf_cuda_prepare_runtime_buffers, "Set the reusable CUDA scratch-buffer size before graph capture.");
    m.def(
        "linear",
        &gguf_cuda_linear,
        "GGUF linear using llama.cpp CUDA kernels.",
        pybind11::arg("raw_weight"),
        pybind11::arg("qtype_name"),
        pybind11::arg("tensor_shape"),
        pybind11::arg("input"),
        pybind11::arg("bias"),
        pybind11::arg("output_dtype_name"),
        pybind11::arg("linear_mode_name") = "auto"
    );
    m.def("embedding", &gguf_cuda_embedding, "GGUF embedding lookup using CUDA row dequantization.");
}
