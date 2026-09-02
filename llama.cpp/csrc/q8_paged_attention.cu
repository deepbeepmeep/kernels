#include "q8_paged_attention_cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kQuantBlock = 32;
constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr int kThreads = kWarp * kWarps;

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
__global__ void quantize_query_kernel(const T * query, int8_t * quantized, __half * scales, int heads, int dim) {
    const int q = blockIdx.x, head = blockIdx.y, qblock = blockIdx.z;
    const int d = qblock * kQuantBlock + threadIdx.x;
    const int64_t base = (static_cast<int64_t>(q) * heads + head) * dim;
    const float value = to_float(query[base + d]);
    float maximum = fabsf(value);
    for (int offset = 16; offset; offset >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    const float scale = fmaxf(maximum / 127.0f, 1.0e-8f);
    quantized[base + d] = static_cast<int8_t>(max(-127, min(127, __float2int_rn(value / scale))));
    if (threadIdx.x == 0) scales[(static_cast<int64_t>(q) * heads + head) * (dim / kQuantBlock) + qblock] = __float2half_rn(scale);
}

template <int SPLITS, int DIM>
__global__ void attention_partials_kernel(const int8_t * query, const __half * query_scales, const int8_t * keys, const int8_t * values, const __half * key_scales, const __half * value_scales, const int32_t * tables, const int32_t * lengths, float * partial_values, float * partial_maxima, float * partial_sums, int num_queries, int num_sequences, int q_heads, int kv_heads, int page, int blocks, int table_width, float softmax_scale) {
    const int q = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const int kv_head = head / (q_heads / kv_heads);
    const int sequence = num_sequences == num_queries ? q : 0;
    const int causal_offset = num_sequences == 1 ? num_queries - 1 - q : 0;
    const int context = min(max(0, lengths[sequence] - causal_offset), table_width * page);
    const int split_size = (context + SPLITS - 1) / SPLITS;
    const int begin = split * split_size, end = min(context, begin + split_size);
    constexpr int qblocks = DIM / kQuantBlock, packs = DIM / 4;
    const int64_t qbase = (static_cast<int64_t>(q) * q_heads + head) * DIM;
    int qp[2] = {0, 0};
    float qs[2] = {0.0f, 0.0f};
    int lp = 0;
    for (int pack = lane; pack < packs; pack += kWarp, ++lp) {
        qp[lp] = reinterpret_cast<const int *>(query + qbase)[pack];
        qs[lp] = __half2float(query_scales[(static_cast<int64_t>(q) * q_heads + head) * qblocks + pack / 8]);
    }
    float accumulator[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY, denominator = 0.0f;

    for (int token = begin + warp; token < end; token += kWarps) {
        const int physical = tables[sequence * table_width + token / page];
        const bool valid = physical >= 0 && physical < blocks;
        int64_t cache_base = 0, scale_base = 0;
        if (valid) {
            cache_base = ((static_cast<int64_t>(physical) * page + token % page) * kv_heads + kv_head) * DIM;
            scale_base = ((static_cast<int64_t>(physical) * page + token % page) * kv_heads + kv_head) * qblocks;
        }
        float dot = 0.0f;
        if (valid) {
            lp = 0;
            for (int pack = lane; pack < packs; pack += kWarp, ++lp) {
                dot += static_cast<float>(packed_dot_i8(qp[lp], reinterpret_cast<const int *>(keys + cache_base)[pack])) * qs[lp] * __half2float(key_scales[scale_base + pack / 8]);
            }
        }
        for (int offset = 16; offset; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);
        float alpha = 1.0f, probability = 0.0f;
        if (lane == 0 && valid) {
            const float score = dot * softmax_scale;
            const float next = fmaxf(maximum, score);
            alpha = isfinite(maximum) ? expf(maximum - next) : 0.0f;
            probability = expf(score - next);
            denominator = denominator * alpha + probability;
            maximum = next;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        probability = __shfl_sync(0xffffffff, probability, 0);
        if (valid) {
            lp = 0;
            for (int pack = lane; pack < packs; pack += kWarp, ++lp) {
                const int packed = reinterpret_cast<const int *>(values + cache_base)[pack];
                const float scale = __half2float(value_scales[scale_base + pack / 8]);
                const int a = lp * 4;
                accumulator[a + 0] = accumulator[a + 0] * alpha + probability * static_cast<float>(static_cast<int8_t>( packed        & 0xff)) * scale;
                accumulator[a + 1] = accumulator[a + 1] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed >>  8) & 0xff)) * scale;
                accumulator[a + 2] = accumulator[a + 2] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xff)) * scale;
                accumulator[a + 3] = accumulator[a + 3] * alpha + probability * static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xff)) * scale;
            }
        }
    }

    extern __shared__ float shared[];
    float * warp_values = shared;
    float * warp_maxima = shared + kWarps * DIM;
    float * warp_sums = warp_maxima + kWarps;
    lp = 0;
    for (int pack = lane; pack < packs; pack += kWarp, ++lp) reinterpret_cast<float4 *>(warp_values + warp * DIM)[pack] = make_float4(accumulator[lp * 4], accumulator[lp * 4 + 1], accumulator[lp * 4 + 2], accumulator[lp * 4 + 3]);
    if (lane == 0) { warp_maxima[warp] = maximum; warp_sums[warp] = denominator; }
    __syncthreads();

    if (warp == 0) {
        float merged_maximum = -INFINITY, merged_sum = 0.0f;
        if (lane == 0) {
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged_maximum = fmaxf(merged_maximum, warp_maxima[w]);
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged_sum += warp_sums[w] * expf(warp_maxima[w] - merged_maximum);
        }
        merged_maximum = __shfl_sync(0xffffffff, merged_maximum, 0);
        merged_sum = __shfl_sync(0xffffffff, merged_sum, 0);
        const int64_t partial = (static_cast<int64_t>(q) * q_heads + head) * SPLITS + split;
        for (int d = lane; d < DIM; d += kWarp) {
            float merged = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged += warp_values[w * DIM + d] * expf(warp_maxima[w] - merged_maximum);
            partial_values[partial * DIM + d] = merged;
        }
        if (lane == 0) { partial_maxima[partial] = merged_maximum; partial_sums[partial] = merged_sum; }
    }
}

template <typename T, int SPLITS, int DIM>
__global__ void attention_reduce_kernel(const float * values, const float * maxima, const float * sums, T * output, int heads) {
    const int q = blockIdx.x, head = blockIdx.y;
    const int64_t base = (static_cast<int64_t>(q) * heads + head) * SPLITS;
    float maximum = -INFINITY, denominator = 0.0f;
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) if (sums[base + s] > 0.0f) maximum = fmaxf(maximum, maxima[base + s]);
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) if (sums[base + s] > 0.0f) denominator += sums[base + s] * expf(maxima[base + s] - maximum);
    }
    maximum = __shfl_sync(0xffffffff, maximum, 0);
    denominator = __shfl_sync(0xffffffff, denominator, 0);
    const int64_t out = (static_cast<int64_t>(q) * heads + head) * DIM;
    for (int d = threadIdx.x; d < DIM; d += kWarp) {
        float value = 0.0f;
        if (denominator > 0.0f) {
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s) if (sums[base + s] > 0.0f) value += values[(base + s) * DIM + d] * expf(maxima[base + s] - maximum);
            value /= denominator;
        }
        output[out + d] = from_float<T>(value);
    }
}

template <int SPLITS, int DIM>
void launch_partials_dim(const int8_t * q, const __half * qs, const int8_t * k, const int8_t * v, const __half * ks, const __half * vs, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int num_queries, int num_sequences, int qh, int kvh, int page, int blocks, int width, float scale, cudaStream_t stream) {
    constexpr size_t shared = static_cast<size_t>(kWarps) * (DIM + 2) * sizeof(float);
    attention_partials_kernel<SPLITS, DIM><<<dim3(num_queries, qh, SPLITS), kThreads, shared, stream>>>(q, qs, k, v, ks, vs, tables, lengths, pv, pm, ps, num_queries, num_sequences, qh, kvh, page, blocks, width, scale);
}

template <int SPLITS>
void launch_partials(const int8_t * q, const __half * qs, const int8_t * k, const int8_t * v, const __half * ks, const __half * vs, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int num_queries, int num_sequences, int qh, int kvh, int dim, int page, int blocks, int width, float scale, cudaStream_t stream) {
    #define CASE_DIM(D) case D: launch_partials_dim<SPLITS, D>(q, qs, k, v, ks, vs, tables, lengths, pv, pm, ps, num_queries, num_sequences, qh, kvh, page, blocks, width, scale, stream); break
    switch (dim) { CASE_DIM(32); CASE_DIM(64); CASE_DIM(96); CASE_DIM(128); CASE_DIM(160); CASE_DIM(192); CASE_DIM(224); CASE_DIM(256); }
    #undef CASE_DIM
}

template <typename T, int SPLITS, int DIM>
void launch_reduce_dim(const float * v, const float * m, const float * s, T * out, int batch, int heads, cudaStream_t stream) {
    attention_reduce_kernel<T, SPLITS, DIM><<<dim3(batch, heads), kWarp, 0, stream>>>(v, m, s, out, heads);
}

template <typename T, int SPLITS, int DIM>
__global__ void dense_attention_partials_kernel(const T * query, const T * keys, const T * values, const int32_t * tables, const int32_t * lengths, float * partial_values, float * partial_maxima, float * partial_sums, int num_queries, int num_sequences, int q_heads, int kv_heads, int page, int blocks, int table_width, float softmax_scale) {
    const int q = blockIdx.x, head = blockIdx.y, split = blockIdx.z;
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const int kv_head = head / (q_heads / kv_heads);
    const int sequence = num_sequences == num_queries ? q : 0;
    const int causal_offset = num_sequences == 1 ? num_queries - 1 - q : 0;
    const int context = min(max(0, lengths[sequence] - causal_offset), table_width * page);
    const int split_size = (context + SPLITS - 1) / SPLITS;
    const int begin = split * split_size, end = min(context, begin + split_size);
    constexpr int per_lane = DIM / kWarp;
    const int64_t qbase = (static_cast<int64_t>(q) * q_heads + head) * DIM;
    float qv[per_lane], acc[per_lane];
    #pragma unroll
    for (int i = 0; i < per_lane; ++i) { qv[i] = to_float(query[qbase + lane + i * kWarp]); acc[i] = 0.0f; }
    float maximum = -INFINITY, denominator = 0.0f;
    for (int token = begin + warp; token < end; token += kWarps) {
        const int physical = tables[sequence * table_width + token / page];
        const bool valid = physical >= 0 && physical < blocks;
        const int64_t base = valid ? ((static_cast<int64_t>(physical) * page + token % page) * kv_heads + kv_head) * DIM : 0;
        float dot = 0.0f;
        if (valid) {
            #pragma unroll
            for (int i = 0; i < per_lane; ++i) dot += qv[i] * to_float(keys[base + lane + i * kWarp]);
        }
        for (int offset = 16; offset; offset >>= 1) dot += __shfl_down_sync(0xffffffff, dot, offset);
        float alpha = 1.0f, probability = 0.0f;
        if (lane == 0 && valid) {
            const float score = dot * softmax_scale, next = fmaxf(maximum, score);
            alpha = isfinite(maximum) ? expf(maximum - next) : 0.0f;
            probability = expf(score - next);
            denominator = denominator * alpha + probability;
            maximum = next;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0); probability = __shfl_sync(0xffffffff, probability, 0);
        if (valid) {
            #pragma unroll
            for (int i = 0; i < per_lane; ++i) acc[i] = acc[i] * alpha + probability * to_float(values[base + lane + i * kWarp]);
        }
    }
    extern __shared__ float shared[];
    float * warp_values = shared, * warp_maxima = shared + kWarps * DIM, * warp_sums = warp_maxima + kWarps;
    #pragma unroll
    for (int i = 0; i < per_lane; ++i) warp_values[warp * DIM + lane + i * kWarp] = acc[i];
    if (lane == 0) { warp_maxima[warp] = maximum; warp_sums[warp] = denominator; }
    __syncthreads();
    if (warp == 0) {
        float merged_maximum = -INFINITY, merged_sum = 0.0f;
        if (lane == 0) {
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged_maximum = fmaxf(merged_maximum, warp_maxima[w]);
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged_sum += warp_sums[w] * expf(warp_maxima[w] - merged_maximum);
        }
        merged_maximum = __shfl_sync(0xffffffff, merged_maximum, 0); merged_sum = __shfl_sync(0xffffffff, merged_sum, 0);
        const int64_t partial = (static_cast<int64_t>(q) * q_heads + head) * SPLITS + split;
        for (int d = lane; d < DIM; d += kWarp) {
            float merged = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) if (warp_sums[w] > 0.0f) merged += warp_values[w * DIM + d] * expf(warp_maxima[w] - merged_maximum);
            partial_values[partial * DIM + d] = merged;
        }
        if (lane == 0) { partial_maxima[partial] = merged_maximum; partial_sums[partial] = merged_sum; }
    }
}

template <typename T, int SPLITS, int DIM>
void launch_dense_dim(const T * q, const T * k, const T * v, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int nq, int ns, int qh, int kvh, int page, int blocks, int width, float scale, cudaStream_t stream) {
    dense_attention_partials_kernel<T, SPLITS, DIM><<<dim3(nq, qh, SPLITS), kThreads, static_cast<size_t>(kWarps) * (DIM + 2) * sizeof(float), stream>>>(q, k, v, tables, lengths, pv, pm, ps, nq, ns, qh, kvh, page, blocks, width, scale);
}

template <typename T, int SPLITS>
void launch_dense(const T * q, const T * k, const T * v, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int nq, int ns, int qh, int kvh, int dim, int page, int blocks, int width, float scale, cudaStream_t stream) {
    if (dim == 256) launch_dense_dim<T, SPLITS, 256>(q, k, v, tables, lengths, pv, pm, ps, nq, ns, qh, kvh, page, blocks, width, scale, stream);
}

template <typename T, int SPLITS>
void launch_reduce(const float * v, const float * m, const float * s, T * out, int batch, int heads, int dim, cudaStream_t stream) {
    #define CASE_DIM(D) case D: launch_reduce_dim<T, SPLITS, D>(v, m, s, out, batch, heads, stream); break
    switch (dim) { CASE_DIM(32); CASE_DIM(64); CASE_DIM(96); CASE_DIM(128); CASE_DIM(160); CASE_DIM(192); CASE_DIM(224); CASE_DIM(256); }
    #undef CASE_DIM
}
} // namespace

void q8_quantize_query_cuda(const void * query, bool bf16, int8_t * quantized, void * scales, int batch, int heads, int dim, cudaStream_t stream) {
    const dim3 grid(batch, heads, dim / kQuantBlock);
    if (bf16) quantize_query_kernel<<<grid, kWarp, 0, stream>>>(static_cast<const __nv_bfloat16 *>(query), quantized, static_cast<__half *>(scales), heads, dim);
    else quantize_query_kernel<<<grid, kWarp, 0, stream>>>(static_cast<const __half *>(query), quantized, static_cast<__half *>(scales), heads, dim);
}

void q8_attention_partials_cuda(const int8_t * q, const void * qs, const int8_t * k, const int8_t * v, const void * ks, const void * vs, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int num_queries, int num_sequences, int qh, int kvh, int dim, int page, int blocks, int width, int splits, float scale, cudaStream_t stream) {
    #define CASE_PARTIAL(N) case N: launch_partials<N>(q, static_cast<const __half *>(qs), k, v, static_cast<const __half *>(ks), static_cast<const __half *>(vs), tables, lengths, pv, pm, ps, num_queries, num_sequences, qh, kvh, dim, page, blocks, width, scale, stream); break
    switch (splits) { CASE_PARTIAL(1); CASE_PARTIAL(2); CASE_PARTIAL(4); CASE_PARTIAL(8); CASE_PARTIAL(16); CASE_PARTIAL(32); CASE_PARTIAL(64); CASE_PARTIAL(128); }
    #undef CASE_PARTIAL
}

void q8_attention_reduce_cuda(const float * v, const float * m, const float * s, void * out, bool bf16, int batch, int heads, int dim, int splits, cudaStream_t stream) {
    #define CASE_REDUCE(N, T) case N: launch_reduce<T, N>(v, m, s, static_cast<T *>(out), batch, heads, dim, stream); break
    if (bf16) { switch (splits) { CASE_REDUCE(1, __nv_bfloat16); CASE_REDUCE(2, __nv_bfloat16); CASE_REDUCE(4, __nv_bfloat16); CASE_REDUCE(8, __nv_bfloat16); CASE_REDUCE(16, __nv_bfloat16); CASE_REDUCE(32, __nv_bfloat16); CASE_REDUCE(64, __nv_bfloat16); CASE_REDUCE(128, __nv_bfloat16); } }
    else { switch (splits) { CASE_REDUCE(1, __half); CASE_REDUCE(2, __half); CASE_REDUCE(4, __half); CASE_REDUCE(8, __half); CASE_REDUCE(16, __half); CASE_REDUCE(32, __half); CASE_REDUCE(64, __half); CASE_REDUCE(128, __half); } }
    #undef CASE_REDUCE
}

void dense_attention_partials_cuda(const void * q, const void * k, const void * v, bool bf16, const int32_t * tables, const int32_t * lengths, float * pv, float * pm, float * ps, int nq, int ns, int qh, int kvh, int dim, int page, int blocks, int width, int splits, float scale, cudaStream_t stream) {
    #define CASE_DENSE(N, T) case N: launch_dense<T, N>(static_cast<const T *>(q), static_cast<const T *>(k), static_cast<const T *>(v), tables, lengths, pv, pm, ps, nq, ns, qh, kvh, dim, page, blocks, width, scale, stream); break
    if (bf16) { switch (splits) { CASE_DENSE(1, __nv_bfloat16); CASE_DENSE(2, __nv_bfloat16); CASE_DENSE(4, __nv_bfloat16); CASE_DENSE(8, __nv_bfloat16); CASE_DENSE(16, __nv_bfloat16); CASE_DENSE(32, __nv_bfloat16); } }
    else { switch (splits) { CASE_DENSE(1, __half); CASE_DENSE(2, __half); CASE_DENSE(4, __half); CASE_DENSE(8, __half); CASE_DENSE(16, __half); CASE_DENSE(32, __half); } }
    #undef CASE_DENSE
}
