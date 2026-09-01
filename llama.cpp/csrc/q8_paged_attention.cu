#include "q8_paged_attention_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kQuantBlock = 32;
constexpr int kThreads = 32;

template <typename T> __device__ __forceinline__ float to_float(T value);
template <> __device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 value) { return __bfloat162float(value); }
template <typename T> __device__ __forceinline__ T from_float(float value);
template <> __device__ __forceinline__ __half from_float(float value) { return __float2half_rn(value); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float value) { return __float2bfloat16_rn(value); }

__device__ __forceinline__ int packed_dot_i8(int a, int b) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, 0);
#else
    int result = 0;
    #pragma unroll
    for (int byte = 0; byte < 4; ++byte) result += static_cast<int>(static_cast<int8_t>(a >> (byte * 8))) * static_cast<int>(static_cast<int8_t>(b >> (byte * 8)));
    return result;
#endif
}

template <typename T>
__global__ void quantize_query_kernel(const T * query, int8_t * quantized_query, float * scales, float * sums, int query_heads, int head_dim) {
    const int q_index = blockIdx.x, q_head = blockIdx.y, q_block = blockIdx.z;
    const int dim = q_block * kQuantBlock + threadIdx.x;
    const int64_t q_base = (static_cast<int64_t>(q_index) * query_heads + q_head) * head_dim;
    const float value = to_float(query[q_base + dim]);
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = fmaxf(maximum / 127.0f, 1.0e-8f);
    const int quantized = max(-127, min(127, __float2int_rn(value / scale)));
    quantized_query[q_base + dim] = static_cast<int8_t>(quantized);
    int sum = quantized;
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) {
        const int64_t index = (static_cast<int64_t>(q_index) * query_heads + q_head) * (head_dim / kQuantBlock) + q_block;
        scales[index] = scale;
        sums[index] = scale * static_cast<float>(sum);
    }
}

__global__ void attention_partials_kernel(const int8_t * query, const float * query_scales, const int8_t * key_cache, const int8_t * value_cache, const __half * key_scales, const __half * value_scales, const int32_t * block_tables, const int32_t * context_lens, float * partial_values, float * partial_maxima, float * partial_sums, int query_heads, int kv_heads, int head_dim, int page_size, int num_cache_blocks, int table_width, int num_splits, float softmax_scale) {
    const int q_index = blockIdx.x, q_head = blockIdx.y, split = blockIdx.z;
    const int kv_head = q_head / (query_heads / kv_heads);
    const int context_len = min(max(0, context_lens[q_index]), table_width * page_size);
    const int split_size = (context_len + num_splits - 1) / num_splits;
    const int token_begin = split * split_size;
    const int token_end = min(context_len, token_begin + split_size);
    const int quant_blocks = head_dim / kQuantBlock;
    float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY, denominator = 0.0f;
    const int64_t q_base = (static_cast<int64_t>(q_index) * query_heads + q_head) * head_dim;
    const int packed_count = head_dim / 4;
    int query_packs[2] = {0, 0};
    float packed_scales[2] = {0.0f, 0.0f};
    int local_pack = 0;
    for (int pack = threadIdx.x; pack < packed_count; pack += kThreads, ++local_pack) {
        query_packs[local_pack] = reinterpret_cast<const int *>(query + q_base)[pack];
        packed_scales[local_pack] = query_scales[(static_cast<int64_t>(q_index) * query_heads + q_head) * quant_blocks + (pack * 4) / kQuantBlock];
    }

    for (int token = token_begin; token < token_end; ++token) {
        const int logical_block = token / page_size;
        const int token_in_block = token % page_size;
        const int physical_block = block_tables[q_index * table_width + logical_block];
        const bool valid = physical_block >= 0 && physical_block < num_cache_blocks;
        int64_t cache_base = 0, scale_base = 0;
        if (valid) {
            cache_base = ((static_cast<int64_t>(physical_block) * page_size + token_in_block) * kv_heads + kv_head) * head_dim;
            scale_base = ((static_cast<int64_t>(physical_block) * page_size + token_in_block) * kv_heads + kv_head) * quant_blocks;
        }
        float dot = 0.0f;
        if (valid) {
            local_pack = 0;
            for (int pack = threadIdx.x; pack < packed_count; pack += kThreads, ++local_pack) {
                const int key_pack = reinterpret_cast<const int *>(key_cache + cache_base)[pack];
                dot += static_cast<float>(packed_dot_i8(query_packs[local_pack], key_pack)) * packed_scales[local_pack] * __half2float(key_scales[scale_base + (pack * 4) / kQuantBlock]);
            }
        }
        for (int offset = 16; offset; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);
        float alpha = 1.0f, probability = 0.0f;
        if (threadIdx.x == 0 && valid) {
            const float score = dot * softmax_scale;
            const float next_maximum = fmaxf(maximum, score);
            alpha = isfinite(maximum) ? expf(maximum - next_maximum) : 0.0f;
            probability = expf(score - next_maximum);
            denominator = denominator * alpha + probability;
            maximum = next_maximum;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        probability = __shfl_sync(0xffffffff, probability, 0);
        if (valid) {
            local_pack = 0;
            for (int pack = threadIdx.x; pack < packed_count; pack += kThreads, ++local_pack) {
                const int packed_value = reinterpret_cast<const int *>(value_cache + cache_base)[pack];
                const float value_scale = __half2float(value_scales[scale_base + pack / 8]);
                const int accumulator_base = local_pack * 4;
                accumulator[accumulator_base + 0] = accumulator[accumulator_base + 0] * alpha + probability * static_cast<float>(static_cast<int8_t>( packed_value        & 0xff)) * value_scale;
                accumulator[accumulator_base + 1] = accumulator[accumulator_base + 1] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed_value >>  8) & 0xff)) * value_scale;
                accumulator[accumulator_base + 2] = accumulator[accumulator_base + 2] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed_value >> 16) & 0xff)) * value_scale;
                accumulator[accumulator_base + 3] = accumulator[accumulator_base + 3] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed_value >> 24) & 0xff)) * value_scale;
            }
        }
    }
    const int64_t partial = (static_cast<int64_t>(q_index) * query_heads + q_head) * num_splits + split;
    if (threadIdx.x == 0) { partial_maxima[partial] = maximum; partial_sums[partial] = denominator; }
    local_pack = 0;
    for (int pack = threadIdx.x; pack < packed_count; pack += kThreads, ++local_pack) {
        reinterpret_cast<float4 *>(partial_values + partial * head_dim)[pack] = make_float4(
            accumulator[local_pack * 4 + 0], accumulator[local_pack * 4 + 1],
            accumulator[local_pack * 4 + 2], accumulator[local_pack * 4 + 3]);
    }
}

template <typename T>
__global__ void attention_reduce_kernel(const float * values, const float * maxima, const float * sums, T * output, int query_heads, int head_dim, int num_splits) {
    const int q_index = blockIdx.x, q_head = blockIdx.y;
    const int64_t base = (static_cast<int64_t>(q_index) * query_heads + q_head) * num_splits;
    float global_maximum = -INFINITY, global_denominator = 0.0f;
    if (threadIdx.x == 0) {
        for (int split = 0; split < num_splits; ++split) if (sums[base + split] > 0.0f) global_maximum = fmaxf(global_maximum, maxima[base + split]);
        for (int split = 0; split < num_splits; ++split) if (sums[base + split] > 0.0f) global_denominator += sums[base + split] * expf(maxima[base + split] - global_maximum);
    }
    global_maximum = __shfl_sync(0xffffffff, global_maximum, 0);
    global_denominator = __shfl_sync(0xffffffff, global_denominator, 0);
    const int64_t output_base = (static_cast<int64_t>(q_index) * query_heads + q_head) * head_dim;
    for (int dim = threadIdx.x; dim < head_dim; dim += kThreads) {
        float value = 0.0f;
        if (global_denominator > 0.0f) {
            for (int split = 0; split < num_splits; ++split) if (sums[base + split] > 0.0f) value += values[(base + split) * head_dim + dim] * expf(maxima[base + split] - global_maximum);
            value /= global_denominator;
        }
        output[output_base + dim] = from_float<T>(value);
    }
}
} // namespace

void q8_quantize_query_cuda(const void * query, bool bf16, int8_t * quantized, float * scales, float * sums, int batch, int heads, int dim, cudaStream_t stream) {
    const dim3 grid(batch, heads, dim / kQuantBlock);
    if (bf16) quantize_query_kernel<<<grid, kQuantBlock, 0, stream>>>(static_cast<const __nv_bfloat16 *>(query), quantized, scales, sums, heads, dim);
    else quantize_query_kernel<<<grid, kQuantBlock, 0, stream>>>(static_cast<const __half *>(query), quantized, scales, sums, heads, dim);
}

void q8_attention_partials_cuda(const int8_t * query, const float * q_scales, const int8_t * keys, const int8_t * values, const void * k_scales, const void * v_scales, const int32_t * tables, const int32_t * lengths, float * partial_values, float * maxima, float * sums, int batch, int q_heads, int kv_heads, int dim, int page, int blocks, int table_width, int splits, float scale, cudaStream_t stream) {
    attention_partials_kernel<<<dim3(batch, q_heads, splits), kThreads, 0, stream>>>(query, q_scales, keys, values, static_cast<const __half *>(k_scales), static_cast<const __half *>(v_scales), tables, lengths, partial_values, maxima, sums, q_heads, kv_heads, dim, page, blocks, table_width, splits, scale);
}

void q8_attention_reduce_cuda(const float * values, const float * maxima, const float * sums, void * output, bool bf16, int batch, int heads, int dim, int splits, cudaStream_t stream) {
    const dim3 grid(batch, heads);
    if (bf16) attention_reduce_kernel<<<grid, kThreads, 0, stream>>>(values, maxima, sums, static_cast<__nv_bfloat16 *>(output), heads, dim, splits);
    else attention_reduce_kernel<<<grid, kThreads, 0, stream>>>(values, maxima, sums, static_cast<__half *>(output), heads, dim, splits);
}
