#include <torch/extension.h>

#include "gguf_llamacpp_ops.h"
#if !defined(GGML_USE_HIP)
#include "short_batch_mma.h"
#endif

#include <pybind11/stl.h>

at::Tensor prism_hadamard(at::Tensor, at::Tensor, bool, int64_t, int64_t, int64_t);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("supports_linear_fusions", &gguf_cuda_supports_linear_fusions);
    m.def("prism_hadamard", &prism_hadamard, "Prism signed block Hadamard transform.");
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
        pybind11::arg("linear_mode_name") = "auto",
        pybind11::arg("fused_output") = false,
        pybind11::arg("silu_mul") = false
    );
    m.def("embedding", &gguf_cuda_embedding, "GGUF embedding lookup using CUDA row dequantization.");
#if !defined(GGML_USE_HIP)
    m.def("set_short_batch_mode", &short_batch_set_mode, "Short-batch Q4_K/PTQ1_0 tensor-core policy: auto, native or mma.");
    m.def("short_batch_mode", &short_batch_mode, "Current short-batch tensor-core policy.");
    m.def("set_short_batch_decision", &short_batch_set_decision, "Enable or disable the tensor-core path for one (qtype, rows, out, in) shape under the auto policy.");
    m.def("clear_short_batch_decisions", &short_batch_clear_decisions, "Forget per-shape short-batch decisions.");
#endif
}
