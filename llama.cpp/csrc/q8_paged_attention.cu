#include "q8_paged_attention.h"

#ifdef small
#undef small
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "vecdotq.cuh"

#include <cfloat>
#include <cmath>
#include <limits>

namespace {

constexpr int Q8_BLOCK_SIZE = 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int MAX_GQA_GROUPS = 16;
constexpr int ATTENTION_THREADS = 128;
constexpr int ATTENTION_WARPS = ATTENTION_THREADS / Q8_BLOCK_SIZE;
constexpr int MIN_SPLITS = 1;
constexpr int MAX_AUTO_SPLITS = 32;
constexpr int MAX_SPLITS = 128;

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t * ptr) {
    return static_cast<float>(*ptr);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

template <typename scalar_t>
__global__ void quantize_query_q8_1(
    const scalar_t * __restrict__ query,
    int8_t * __restrict__ query_q8,
    half * __restrict__ query_scales,
    int num_query_heads,
    int head_dim,
    float softmax_scale) {
    const int lane = threadIdx.x;
    const int q8_block = blockIdx.x;
    const int query_head = blockIdx.y;
    const int query_index = blockIdx.z;
    const int base = (query_index * num_query_heads + query_head) * head_dim + q8_block * Q8_BLOCK_SIZE;
    const float value = load_float(query + base + lane) * softmax_scale;
    float amax = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    const float scale = amax / 127.0f;
    query_q8[base + lane] = scale == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    if (lane == 0) {
        query_scales[(query_index * num_query_heads + query_head) * (head_dim / Q8_BLOCK_SIZE) + q8_block] = __float2half(scale);
    }
}

template <typename scalar_t, int num_splits>
__global__ void q8_attention_partials(
    const int8_t * __restrict__ query_q8,
    const half * __restrict__ query_scales,
    const int8_t * __restrict__ key_cache,
    const int8_t * __restrict__ value_cache,
    const half * __restrict__ key_scales,
    const half * __restrict__ value_scales,
    const int32_t * __restrict__ block_table,
    const int32_t * __restrict__ context_lens,
    float * __restrict__ partials,
    int num_query_heads,
    int num_kv_heads,
    int num_queries,
    int num_sequences,
    int head_dim,
    int cache_block_size,
    int max_blocks_per_sequence) {
    __shared__ float attention_weights[ATTENTION_THREADS];
    __shared__ float warp_max[ATTENTION_WARPS];
    __shared__ float warp_sum_values[ATTENTION_WARPS];
    __shared__ float warp_outputs[ATTENTION_WARPS][MAX_HEAD_DIM];
    const int tid = threadIdx.x;
    const int lane = tid % Q8_BLOCK_SIZE;
    const int warp = tid / Q8_BLOCK_SIZE;
    const int query_head = blockIdx.x;
    const int query_index = blockIdx.y;
    const int split = blockIdx.z;
    const int query_groups = num_query_heads / num_kv_heads;
    const int kv_head = query_head / query_groups;
    const int sequence_index = num_sequences == num_queries ? query_index : 0;
    const int sequence_length = context_lens[sequence_index] - (num_sequences == 1 ? num_queries - 1 - query_index : 0);
    const int token_begin = sequence_length * split / num_splits;
    const int token_end = sequence_length * (split + 1) / num_splits;
    const int q8_blocks = head_dim / Q8_BLOCK_SIZE;
    const int query_base = (query_index * num_query_heads + query_head) * head_dim;

    constexpr int Q8_VALUES_PER_INT = sizeof(int);
    constexpr int Q8_INTS_PER_BLOCK = Q8_BLOCK_SIZE / Q8_VALUES_PER_INT;
    constexpr int MAX_QUERY_INTS_PER_LANE = MAX_HEAD_DIM / (Q8_BLOCK_SIZE * Q8_VALUES_PER_INT);
    constexpr int MAX_OUTPUT_VALUES_PER_LANE = MAX_HEAD_DIM / Q8_BLOCK_SIZE;
    int query_q8_values[MAX_QUERY_INTS_PER_LANE];
    float query_scale_values[MAX_QUERY_INTS_PER_LANE];
    float output_values[MAX_OUTPUT_VALUES_PER_LANE];
#pragma unroll
    for (int index = 0; index < MAX_OUTPUT_VALUES_PER_LANE; ++index) {
        if (index < head_dim / Q8_BLOCK_SIZE) {
            output_values[index] = 0.0f;
        }
    }

    const int query_ints = head_dim / Q8_VALUES_PER_INT;
#pragma unroll
    for (int group_base = 0; group_base < MAX_HEAD_DIM / Q8_VALUES_PER_INT; group_base += Q8_BLOCK_SIZE) {
        const int group = group_base + lane;
        if (group < query_ints) {
            ggml_cuda_memcpy_1<sizeof(int)>(&query_q8_values[group_base / Q8_BLOCK_SIZE], query_q8 + query_base + group * Q8_VALUES_PER_INT);
            query_scale_values[group_base / Q8_BLOCK_SIZE] = __half2float(query_scales[(query_index * num_query_heads + query_head) * q8_blocks + group / Q8_INTS_PER_BLOCK]);
        }
    }

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    for (int tile_begin = token_begin; tile_begin < token_end; tile_begin += ATTENTION_THREADS) {
        float score = -FLT_MAX;
#pragma unroll
        for (int token_in_warp = 0; token_in_warp < Q8_BLOCK_SIZE; ++token_in_warp) {
            const int token = tile_begin + warp * Q8_BLOCK_SIZE + token_in_warp;
            float dot = 0.0f;
            if (token < token_end) {
                const int logical_block = token / cache_block_size;
                const int token_in_block = token - logical_block * cache_block_size;
                const int physical_block = block_table[sequence_index * max_blocks_per_sequence + logical_block];
                const int cache_token = physical_block * cache_block_size + token_in_block;
                const int cache_base = (cache_token * num_kv_heads + kv_head) * head_dim;
                const int scale_base = (cache_token * num_kv_heads + kv_head) * q8_blocks;
#pragma unroll
                for (int group_base = 0; group_base < MAX_HEAD_DIM / Q8_VALUES_PER_INT; group_base += Q8_BLOCK_SIZE) {
                    const int group = group_base + lane;
                    if (group < query_ints) {
                        const int q8_block = group / Q8_INTS_PER_BLOCK;
                        int key_q8;
                        ggml_cuda_memcpy_1<sizeof(int)>(&key_q8, key_cache + cache_base + group * Q8_VALUES_PER_INT);
                        dot += vec_dot_q8_0_q8_1_impl<float, 1>(&key_q8, &query_q8_values[group_base / Q8_BLOCK_SIZE], __half2float(key_scales[scale_base + q8_block]), query_scale_values[group_base / Q8_BLOCK_SIZE]);
                    }
                }
                dot = warp_sum(dot);
            }
            if (lane == token_in_warp) {
                score = token < token_end ? dot : -FLT_MAX;
            }
        }

        float tile_max = score;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));
        }
        const float next_max = fmaxf(running_max, tile_max);
        const float old_weight = running_sum == 0.0f ? 0.0f : __expf(running_max - next_max);
        const float new_weight = score == -FLT_MAX ? 0.0f : __expf(score - next_max);
        const float tile_sum = warp_sum(new_weight);
        running_sum = running_sum * old_weight + tile_sum;
        running_max = next_max;
        attention_weights[tid] = new_weight;
#pragma unroll
        for (int index = 0; index < MAX_OUTPUT_VALUES_PER_LANE; ++index) {
            if (index < head_dim / Q8_BLOCK_SIZE) {
                output_values[index] *= old_weight;
            }
        }
        __syncwarp();

#pragma unroll
        for (int token_in_warp = 0; token_in_warp < Q8_BLOCK_SIZE; ++token_in_warp) {
            const int token = tile_begin + warp * Q8_BLOCK_SIZE + token_in_warp;
            if (token >= token_end) {
                break;
            }
            const int logical_block = token / cache_block_size;
            const int token_in_block = token - logical_block * cache_block_size;
            const int physical_block = block_table[sequence_index * max_blocks_per_sequence + logical_block];
            const int cache_token = physical_block * cache_block_size + token_in_block;
            const int cache_base = (cache_token * num_kv_heads + kv_head) * head_dim;
            const int scale_base = (cache_token * num_kv_heads + kv_head) * q8_blocks;
            const float weight = attention_weights[warp * Q8_BLOCK_SIZE + token_in_warp];
#pragma unroll
            for (int group_base = 0; group_base < MAX_HEAD_DIM / Q8_VALUES_PER_INT; group_base += Q8_BLOCK_SIZE) {
                const int group = group_base + lane;
                if (group < query_ints) {
                    int value_q8;
                    ggml_cuda_memcpy_1<sizeof(int)>(&value_q8, value_cache + cache_base + group * Q8_VALUES_PER_INT);
                    const int8_t * values = reinterpret_cast<const int8_t *>(&value_q8);
                    const float value_scale = __half2float(value_scales[scale_base + group / Q8_INTS_PER_BLOCK]);
                    const int output_base = group_base / Q8_BLOCK_SIZE * Q8_VALUES_PER_INT;
#pragma unroll
                    for (int index = 0; index < Q8_VALUES_PER_INT; ++index) {
                        output_values[output_base + index] += static_cast<float>(values[index]) * value_scale * weight;
                    }
                }
            }
        }
    }

#pragma unroll
    for (int group_base = 0; group_base < MAX_HEAD_DIM / Q8_VALUES_PER_INT; group_base += Q8_BLOCK_SIZE) {
        const int group = group_base + lane;
        if (group < query_ints) {
#pragma unroll
            for (int index = 0; index < Q8_VALUES_PER_INT; ++index) {
                warp_outputs[warp][group * Q8_VALUES_PER_INT + index] = output_values[group_base / Q8_BLOCK_SIZE * Q8_VALUES_PER_INT + index];
            }
        }
    }
    if (lane == 0) {
        warp_max[warp] = running_max;
        warp_sum_values[warp] = running_sum;
    }
    __syncthreads();

    float combined_max = warp_max[0];
#pragma unroll
    for (int index = 1; index < ATTENTION_WARPS; ++index) {
        combined_max = fmaxf(combined_max, warp_max[index]);
    }
    float combined_sum = 0.0f;
#pragma unroll
    for (int index = 0; index < ATTENTION_WARPS; ++index) {
        if (warp_sum_values[index] != 0.0f) {
            combined_sum += warp_sum_values[index] * __expf(warp_max[index] - combined_max);
        }
    }

    const int partial_stride = head_dim + 2;
    const int partial_base = ((query_index * num_query_heads + query_head) * num_splits + split) * partial_stride;
    if (tid == 0) {
        partials[partial_base] = combined_max;
        partials[partial_base + 1] = combined_sum;
    }
    for (int dim = tid; dim < head_dim; dim += ATTENTION_THREADS) {
        float combined_value = 0.0f;
#pragma unroll
        for (int index = 0; index < ATTENTION_WARPS; ++index) {
            if (warp_sum_values[index] != 0.0f) {
                combined_value += warp_outputs[index][dim] * __expf(warp_max[index] - combined_max);
            }
        }
        partials[partial_base + 2 + dim] = combined_value;
    }
}

template <typename scalar_t, int num_splits>
__global__ void q8_attention_reduce(
    const float * __restrict__ partials,
    scalar_t * __restrict__ output,
    int num_query_heads,
    int head_dim) {
    const int lane = threadIdx.x;
    const int query_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int q8_blocks = head_dim / Q8_BLOCK_SIZE;
    const int partial_stride = head_dim + 2;
    const int group_base = (batch * num_query_heads + query_head) * num_splits * partial_stride;

    float global_max = -FLT_MAX;
#pragma unroll
    for (int split = 0; split < num_splits; ++split) {
        global_max = fmaxf(global_max, partials[group_base + split * partial_stride]);
    }

    float denominator = 0.0f;
#pragma unroll
    for (int split = 0; split < num_splits; ++split) {
        const int base = group_base + split * partial_stride;
        const float partial_sum = partials[base + 1];
        if (partial_sum != 0.0f) {
            denominator += partial_sum * __expf(partials[base] - global_max);
        }
    }

    const int output_base = (batch * num_query_heads + query_head) * head_dim;
#pragma unroll
    for (int block = 0; block < MAX_HEAD_DIM / Q8_BLOCK_SIZE; ++block) {
        if (block < q8_blocks) {
            const int dim = block * Q8_BLOCK_SIZE + lane;
            float value = 0.0f;
#pragma unroll
            for (int split = 0; split < num_splits; ++split) {
                const int base = group_base + split * partial_stride;
                const float partial_sum = partials[base + 1];
                if (partial_sum != 0.0f) {
                    value += partials[base + 2 + dim] * __expf(partials[base] - global_max);
                }
            }
            output[output_base + dim] = static_cast<scalar_t>(denominator == 0.0f ? 0.0f : value / denominator);
        }
    }
}

template <typename scalar_t>
int select_num_splits(int device, int num_queries, int num_query_heads, int cache_capacity) {
    int active_blocks_per_sm = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm,
        q8_attention_partials<scalar_t, MIN_SPLITS>,
        ATTENTION_THREADS,
        0));
    const int sm_count = at::cuda::getDeviceProperties(device)->multiProcessorCount;
    const int parallel_groups = num_queries * num_query_heads;
    const int blocks_per_wave = sm_count * active_blocks_per_sm;
    const int capacity_splits = (cache_capacity + ATTENTION_THREADS - 1) / ATTENTION_THREADS;
    const int max_splits = capacity_splits < MAX_AUTO_SPLITS ? capacity_splits : MAX_AUTO_SPLITS;
    const int initial_splits = active_blocks_per_sm < capacity_splits ? active_blocks_per_sm : capacity_splits;

    int first_candidate = MIN_SPLITS;
    while (first_candidate < initial_splits && first_candidate < max_splits) {
        first_candidate *= 2;
    }
    if (first_candidate > max_splits) {
        first_candidate /= 2;
    }

    int best_splits = first_candidate;
    int best_waves = 0;
    int best_efficiency_percent = 0;
    for (int splits = first_candidate; splits <= max_splits; splits *= 2) {
        const int64_t total_blocks = static_cast<int64_t>(parallel_groups) * splits;
        const int waves = static_cast<int>((total_blocks + blocks_per_wave - 1) / blocks_per_wave);
        const int efficiency_percent = static_cast<int>(100 * total_blocks / (static_cast<int64_t>(waves) * blocks_per_wave));

        // Match upstream fattn: once a configuration fills a wave well, avoid
        // paying another wave and a larger partial reduction for no useful gain.
        if (best_efficiency_percent >= 95 && waves > best_waves) {
            break;
        }
        if (efficiency_percent > best_efficiency_percent) {
            best_splits = splits;
            best_waves = waves;
            best_efficiency_percent = efficiency_percent;
        }
    }
    return best_splits;
}

template <typename scalar_t, int num_splits>
at::Tensor launch_q8_paged_attention(
    const at::Tensor & query,
    const at::Tensor & key_cache,
    const at::Tensor & value_cache,
    const at::Tensor & key_scales,
    const at::Tensor & value_scales,
    const at::Tensor & block_table,
    const at::Tensor & context_lens,
    int num_queries,
    int num_sequences,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int cache_block_size,
    int max_blocks_per_sequence,
    float softmax_scale,
    cudaStream_t stream) {
    auto output = at::empty({num_queries, 1, num_query_heads, head_dim}, query.options());
    auto query_q8 = at::empty(query.sizes(), query.options().dtype(at::kChar));
    auto query_scales = at::empty({num_queries, num_query_heads, head_dim / Q8_BLOCK_SIZE}, query.options().dtype(at::kHalf));
    auto partials = at::empty({num_queries, num_query_heads, num_splits, head_dim + 2}, query.options().dtype(at::kFloat));
    const dim3 grid(num_query_heads, num_queries, num_splits);
    const dim3 reduce_grid(num_query_heads, num_queries);
    const dim3 query_grid(head_dim / Q8_BLOCK_SIZE, num_query_heads, num_queries);
    quantize_query_q8_1<scalar_t><<<query_grid, Q8_BLOCK_SIZE, 0, stream>>>(query.data_ptr<scalar_t>(), query_q8.data_ptr<int8_t>(), reinterpret_cast<half *>(query_scales.data_ptr<at::Half>()), num_query_heads, head_dim, softmax_scale);
    q8_attention_partials<scalar_t, num_splits><<<grid, ATTENTION_THREADS, 0, stream>>>(
        query_q8.data_ptr<int8_t>(), reinterpret_cast<const half *>(query_scales.data_ptr<at::Half>()), key_cache.data_ptr<int8_t>(), value_cache.data_ptr<int8_t>(),
        reinterpret_cast<const half *>(key_scales.data_ptr<at::Half>()), reinterpret_cast<const half *>(value_scales.data_ptr<at::Half>()), block_table.data_ptr<int32_t>(),
        context_lens.data_ptr<int32_t>(), partials.data_ptr<float>(), num_query_heads, num_kv_heads, num_queries, num_sequences,
        head_dim, cache_block_size, max_blocks_per_sequence);
    q8_attention_reduce<scalar_t, num_splits><<<reduce_grid, Q8_BLOCK_SIZE, 0, stream>>>(
        partials.data_ptr<float>(), output.data_ptr<scalar_t>(), num_query_heads, head_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

template <typename scalar_t>
at::Tensor dispatch_q8_paged_attention(
    int num_splits,
    const at::Tensor & query,
    const at::Tensor & key_cache,
    const at::Tensor & value_cache,
    const at::Tensor & key_scales,
    const at::Tensor & value_scales,
    const at::Tensor & block_table,
    const at::Tensor & context_lens,
    int num_queries,
    int num_sequences,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int cache_block_size,
    int max_blocks_per_sequence,
    float softmax_scale,
    cudaStream_t stream) {
    if (num_splits == 1) {
        return launch_q8_paged_attention<scalar_t, 1>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 2) {
        return launch_q8_paged_attention<scalar_t, 2>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 4) {
        return launch_q8_paged_attention<scalar_t, 4>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 8) {
        return launch_q8_paged_attention<scalar_t, 8>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 16) {
        return launch_q8_paged_attention<scalar_t, 16>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 32) {
        return launch_q8_paged_attention<scalar_t, 32>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    if (num_splits == 64) {
        return launch_q8_paged_attention<scalar_t, 64>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
    }
    return launch_q8_paged_attention<scalar_t, MAX_SPLITS>(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, softmax_scale, stream);
}

void validate_inputs(
    const at::Tensor & query,
    const at::Tensor & key_cache,
    const at::Tensor & value_cache,
    const at::Tensor & key_scales,
    const at::Tensor & value_scales,
    const at::Tensor & block_table,
    const at::Tensor & context_lens) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, "query must be FP16 or BF16");
    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, query_heads, head_dim]");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "KV cache must be CUDA tensors");
    TORCH_CHECK(key_cache.scalar_type() == at::kChar && value_cache.scalar_type() == at::kChar, "KV cache must be INT8");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes() && key_cache.dim() == 4, "KV cache must have matching [blocks, block_size, kv_heads, head_dim] shapes");
    TORCH_CHECK(key_cache.is_contiguous() && value_cache.is_contiguous(), "KV cache must be contiguous");
    TORCH_CHECK(key_scales.scalar_type() == at::kHalf && value_scales.scalar_type() == at::kHalf, "KV scales must be FP16 Q8_0 scales");
    TORCH_CHECK(key_scales.sizes() == value_scales.sizes() && key_scales.dim() == 4, "KV scales must have matching shapes");
    TORCH_CHECK(key_scales.is_contiguous() && value_scales.is_contiguous(), "KV scales must be contiguous");
    TORCH_CHECK(block_table.is_cuda() && block_table.scalar_type() == at::kInt && block_table.dim() == 2 && block_table.is_contiguous(), "block_table must be contiguous CUDA INT32 [batch, max_blocks]");
    TORCH_CHECK(context_lens.is_cuda() && context_lens.scalar_type() == at::kInt && context_lens.dim() == 1 && context_lens.is_contiguous(), "context_lens must be contiguous CUDA INT32 [batch]");
    TORCH_CHECK(query.device() == key_cache.device() && query.device() == value_cache.device() && query.device() == key_scales.device() && query.device() == value_scales.device() && query.device() == block_table.device() && query.device() == context_lens.device(), "all tensors must be on the same CUDA device");
    TORCH_CHECK(block_table.size(0) == context_lens.size(0), "block_table and context_lens batch dimensions must match");
    TORCH_CHECK(block_table.size(0) == query.size(0) || block_table.size(0) == 1, "cache batch must match query batch, except for single-sequence speculative decode");
    TORCH_CHECK(query.size(2) == key_cache.size(3), "query and cache head dimensions must match");
    TORCH_CHECK(query.size(2) > 0 && query.size(2) <= MAX_HEAD_DIM && query.size(2) % Q8_BLOCK_SIZE == 0, "head_dim must be a multiple of 32 and at most 256");
    TORCH_CHECK(query.size(1) % key_cache.size(2) == 0, "query heads must be divisible by KV heads");
    TORCH_CHECK(query.size(1) / key_cache.size(2) <= MAX_GQA_GROUPS, "at most 16 query heads per KV head are supported");
    TORCH_CHECK(key_scales.size(0) == key_cache.size(0) && key_scales.size(1) == key_cache.size(1) && key_scales.size(2) == key_cache.size(2) && key_scales.size(3) == key_cache.size(3) / Q8_BLOCK_SIZE, "KV scales must have shape [blocks, block_size, kv_heads, head_dim / 32]");
}

} // namespace

int64_t q8_paged_attention_num_splits(const at::Tensor & query, int64_t cache_capacity) {
    TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, "query must be FP16 or BF16");
    TORCH_CHECK(query.dim() == 3, "query must have shape [batch, query_heads, head_dim]");
    TORCH_CHECK(cache_capacity > 0 && cache_capacity <= std::numeric_limits<int>::max(), "cache_capacity must fit in a positive int");
    const at::cuda::CUDAGuard guard(query.device());
    const int num_queries = static_cast<int>(query.size(0));
    const int num_query_heads = static_cast<int>(query.size(1));
    if (query.scalar_type() == at::kHalf) {
        return select_num_splits<c10::Half>(query.device().index(), num_queries, num_query_heads, static_cast<int>(cache_capacity));
    }
    return select_num_splits<c10::BFloat16>(query.device().index(), num_queries, num_query_heads, static_cast<int>(cache_capacity));
}

at::Tensor q8_paged_attention(
    const at::Tensor & query,
    const at::Tensor & key_cache,
    const at::Tensor & value_cache,
    const at::Tensor & key_scales,
    const at::Tensor & value_scales,
    const at::Tensor & block_table,
    const at::Tensor & context_lens,
    double softmax_scale,
    int64_t forced_num_splits) {
    validate_inputs(query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens);
    const at::cuda::CUDAGuard guard(query.device());
    const int num_queries = static_cast<int>(query.size(0));
    const int num_sequences = static_cast<int>(block_table.size(0));
    const int num_query_heads = static_cast<int>(query.size(1));
    const int num_kv_heads = static_cast<int>(key_cache.size(2));
    const int head_dim = static_cast<int>(query.size(2));
    const int cache_block_size = static_cast<int>(key_cache.size(1));
    const int max_blocks_per_sequence = static_cast<int>(block_table.size(1));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.device().index()).stream();
    const int cache_capacity = max_blocks_per_sequence * cache_block_size;
    const float scale = static_cast<float>(softmax_scale);
    TORCH_CHECK(forced_num_splits == 0 || forced_num_splits == 1 || forced_num_splits == 2 || forced_num_splits == 4 || forced_num_splits == 8 || forced_num_splits == 16 || forced_num_splits == 32 || forced_num_splits == 64 || forced_num_splits == 128, "forced_num_splits must be 0 or a power of two from 1 through 128");

    if (query.scalar_type() == at::kHalf) {
        const int num_splits = forced_num_splits ? static_cast<int>(forced_num_splits) : select_num_splits<c10::Half>(query.device().index(), num_queries, num_query_heads, cache_capacity);
        return dispatch_q8_paged_attention<c10::Half>(num_splits, query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, scale, stream);
    }
    const int num_splits = forced_num_splits ? static_cast<int>(forced_num_splits) : select_num_splits<c10::BFloat16>(query.device().index(), num_queries, num_query_heads, cache_capacity);
    return dispatch_q8_paged_attention<c10::BFloat16>(num_splits, query, key_cache, value_cache, key_scales, value_scales, block_table, context_lens, num_queries, num_sequences, num_query_heads, num_kv_heads, head_dim, cache_block_size, max_blocks_per_sequence, scale, stream);
}
