#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

void q8_quantize_query_cuda(const void * query, bool bfloat16, int8_t * quantized_query, void * query_scales, int batch, int query_heads, int head_dim, cudaStream_t stream);
void q8_attention_partials_cuda(const int8_t * query, const void * query_scales, const int8_t * key_cache, const int8_t * value_cache, const void * key_scales, const void * value_scales, const int32_t * block_tables, const int32_t * context_lens, float * partial_values, float * partial_maxima, float * partial_sums, int num_queries, int num_sequences, int query_heads, int kv_heads, int head_dim, int page_size, int num_cache_blocks, int max_blocks_per_sequence, int num_splits, float softmax_scale, cudaStream_t stream);
void q8_attention_reduce_cuda(const float * partial_values, const float * partial_maxima, const float * partial_sums, void * output, bool bfloat16, int batch, int query_heads, int head_dim, int num_splits, cudaStream_t stream);
