#pragma once

#include <ATen/ATen.h>

at::Tensor q8_paged_attention(
    const at::Tensor & query,
    const at::Tensor & key_cache,
    const at::Tensor & value_cache,
    const at::Tensor & key_scales,
    const at::Tensor & value_scales,
    const at::Tensor & block_table,
    const at::Tensor & context_lens,
    double softmax_scale,
    int64_t forced_num_splits);

int64_t q8_paged_attention_num_splits(const at::Tensor & query, int64_t cache_capacity);
