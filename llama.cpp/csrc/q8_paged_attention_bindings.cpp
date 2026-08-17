#include <torch/extension.h>

#include "q8_paged_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "q8_paged_attention",
        &q8_paged_attention,
        "Decode-only paged Q8_0-style KV-cache attention (FP16/BF16 I/O, FP32 accumulation).",
        pybind11::arg("query"),
        pybind11::arg("key_cache"),
        pybind11::arg("value_cache"),
        pybind11::arg("key_scales"),
        pybind11::arg("value_scales"),
        pybind11::arg("block_table"),
        pybind11::arg("context_lens"),
        pybind11::arg("softmax_scale"),
        pybind11::arg("forced_num_splits") = 0);
    m.def(
        "q8_paged_attention_num_splits",
        &q8_paged_attention_num_splits,
        "Return the automatically selected Q8 paged-attention split count.",
        pybind11::arg("query"),
        pybind11::arg("cache_capacity"));
}
