#include <torch/extension.h>
at::Tensor prism_decode(at::Tensor, at::Tensor, at::Tensor, const c10::optional<at::Tensor> &, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode", &prism_decode, "Fused packed PTQ1 single-token decode.");
}
