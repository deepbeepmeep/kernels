#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "q8_paged_attention_cuda.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr int64_t kQuantBlock = 32;
constexpr int64_t kTargetTokensPerSplit = 128;
constexpr int64_t kMaxAutoSplits = 32;
constexpr int64_t kMaxSplits = 128;

int64_t choose_num_splits(const at::Tensor & query, int64_t cache_capacity) {
    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, query_heads, head_dim]");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0, "query dimensions must be nonzero");
    TORCH_CHECK(cache_capacity >= 0, "cache_capacity must be non-negative");
    if (cache_capacity == 0) {
        return 1;
    }
    const int64_t requested = std::clamp<int64_t>((cache_capacity + kTargetTokensPerSplit - 1) / kTargetTokensPerSplit, 1, kMaxAutoSplits);
    int64_t splits = 1;
    while (splits < requested) {
        splits <<= 1;
    }
    return splits;
}

void validate_inputs(const at::Tensor & query, const at::Tensor & key_cache, const at::Tensor & value_cache, const at::Tensor & key_scales, const at::Tensor & value_scales, const at::Tensor & block_tables, const at::Tensor & context_lens) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, "query must be fp16 or bf16");
    TORCH_CHECK(query.dim() == 3 && query.is_contiguous(), "query must be contiguous [batch, query_heads, head_dim]");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0, "query dimensions must be nonzero");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda() && key_scales.is_cuda() && value_scales.is_cuda(), "KV cache and scales must be CUDA tensors");
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar, "K/V cache must be int8");
    TORCH_CHECK(key_scales.scalar_type() == at::kHalf && value_scales.scalar_type() == at::kHalf, "K/V scales must be fp16");
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(), "K/V cache must have matching [blocks, page, kv_heads, head_dim] shapes");
    TORCH_CHECK(key_cache.size(0) > 0 && key_cache.size(1) > 0 && key_cache.size(2) > 0 && key_cache.size(3) > 0, "KV cache dimensions must be nonzero");
    TORCH_CHECK(key_scales.dim() == 4 && value_scales.sizes() == key_scales.sizes(), "K/V scales must have matching [blocks, page, kv_heads, head_dim/32] shapes");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous() && key_scales.is_contiguous() && value_scales.is_contiguous(), "KV cache and scales must be contiguous");
    TORCH_CHECK(block_tables.is_cuda() && context_lens.is_cuda() && block_tables.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt, "block_tables and context_lens must be CUDA int32 tensors");
    TORCH_CHECK(block_tables.dim() == 2 && context_lens.dim() == 1 && block_tables.is_contiguous() && context_lens.is_contiguous(), "block_tables/context_lens have invalid layout");
    TORCH_CHECK(block_tables.size(0) > 0 && block_tables.size(1) > 0 && context_lens.size(0) > 0, "block_tables/context_lens dimensions must be nonzero");
    TORCH_CHECK(block_tables.size(0) == context_lens.size(0), "block_tables and context_lens must have the same sequence count");
    TORCH_CHECK(block_tables.size(0) == 1 || block_tables.size(0) == query.size(0), "one sequence may own all query rows, otherwise one sequence is required per query row");
    TORCH_CHECK(key_cache.size(2) > 0 && query.size(1) % key_cache.size(2) == 0, "query_heads must be divisible by kv_heads");
    TORCH_CHECK(query.size(2) == key_cache.size(3), "query and cache head dimensions must match");
    TORCH_CHECK(query.size(2) <= 256 && query.size(2) % kQuantBlock == 0, "head_dim must be a multiple of 32 no larger than 256");
    TORCH_CHECK(key_scales.size(0) == key_cache.size(0) && key_scales.size(1) == key_cache.size(1) && key_scales.size(2) == key_cache.size(2) && key_scales.size(3) == key_cache.size(3) / kQuantBlock, "scale layout must match Q8_0 cache blocks");
    TORCH_CHECK(query.get_device() == key_cache.get_device() && query.get_device() == value_cache.get_device() && query.get_device() == key_scales.get_device() && query.get_device() == value_scales.get_device() && query.get_device() == block_tables.get_device() && query.get_device() == context_lens.get_device(), "all tensors must be on the same CUDA device");
}

} // namespace

at::Tensor q8_paged_attention(at::Tensor query, at::Tensor key_cache, at::Tensor value_cache, at::Tensor key_scales, at::Tensor value_scales, at::Tensor block_tables, at::Tensor context_lens, double softmax_scale, int64_t forced_num_splits) {
    validate_inputs(query, key_cache, value_cache, key_scales, value_scales, block_tables, context_lens);
    TORCH_CHECK(forced_num_splits >= 0, "forced_num_splits must be zero or a power of two in [1, 128]");
    c10::cuda::CUDAGuard device_guard(query.device());
    const int64_t cache_capacity = block_tables.size(1) * key_cache.size(1);
    const int64_t num_splits = forced_num_splits > 0 ? forced_num_splits : choose_num_splits(query, cache_capacity);
    TORCH_CHECK(num_splits >= 1 && num_splits <= kMaxSplits && (num_splits & (num_splits - 1)) == 0, "forced_num_splits must be a power of two in [1, 128]");

    auto float_options = query.options().dtype(at::kFloat);
    auto quantized_query = at::empty(query.sizes(), query.options().dtype(at::kChar));
    auto query_scales = at::empty({query.size(0), query.size(1), query.size(2) / kQuantBlock}, query.options().dtype(at::kHalf));
    auto partial_values = at::empty({query.size(0), query.size(1), num_splits, query.size(2)}, float_options);
    auto partial_maxima = at::empty({query.size(0), query.size(1), num_splits}, float_options);
    auto partial_sums = at::empty({query.size(0), query.size(1), num_splits}, float_options);
    auto output = at::empty({query.size(0), 1, query.size(1), query.size(2)}, query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.get_device());
    const bool bfloat16 = query.scalar_type() == at::kBFloat16;

    q8_quantize_query_cuda(query.const_data_ptr(), bfloat16, quantized_query.data_ptr<int8_t>(), query_scales.mutable_data_ptr(), query.size(0), query.size(1), query.size(2), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    q8_attention_partials_cuda(quantized_query.data_ptr<int8_t>(), query_scales.const_data_ptr(), key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(), key_scales.const_data_ptr(), value_scales.const_data_ptr(), block_tables.data_ptr<int32_t>(), context_lens.data_ptr<int32_t>(), partial_values.data_ptr<float>(), partial_maxima.data_ptr<float>(), partial_sums.data_ptr<float>(), query.size(0), block_tables.size(0), query.size(1), key_cache.size(2), query.size(2), key_cache.size(1), key_cache.size(0), block_tables.size(1), num_splits, static_cast<float>(softmax_scale), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    q8_attention_reduce_cuda(partial_values.data_ptr<float>(), partial_maxima.data_ptr<float>(), partial_sums.data_ptr<float>(), output.mutable_data_ptr(), bfloat16, query.size(0), query.size(1), query.size(2), num_splits, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

int64_t q8_paged_attention_num_splits(const at::Tensor & query, int64_t cache_capacity) {
    return choose_num_splits(query, cache_capacity);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("q8_paged_attention", &q8_paged_attention, "Direct Q8_0 paged KV attention with split-K online softmax", pybind11::arg("query"), pybind11::arg("key_cache"), pybind11::arg("value_cache"), pybind11::arg("key_scales"), pybind11::arg("value_scales"), pybind11::arg("block_tables"), pybind11::arg("context_lens"), pybind11::arg("softmax_scale"), pybind11::arg("forced_num_splits") = 0);
    module.def("q8_paged_attention_num_splits", &q8_paged_attention_num_splits, "Return the split count selected for a query/cache capacity", pybind11::arg("query"), pybind11::arg("cache_capacity"));
}
