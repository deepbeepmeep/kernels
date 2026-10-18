#pragma once

// Row loop of large GGUF linears in one cooperative launch, without an FP32 product or grid barriers.
//
// A resident grid takes work items from one ticket counter, in list order:
//     group g = [quantize chunk g: one item per row tile and 4*threads columns] + [output tiles of chunk g-1]
// Small quantize items spread a chunk over many SMs: the first tiles do not wait for a few blocks to read whole wide rows.
// Quantized chunks live in a ring of mmq_pipe_slots slots: a quantize item waits until the tiles of chunk c-3 (handed out two groups
// earlier) have released its slot, a tile waits until its chunk is quantized (handed out one group earlier, before a whole chunk of
// tiles). Every wait targets items handed out earlier to resident blocks, so the loop cannot deadlock, and in steady state nobody
// waits: quantization overlaps the matrix products on other SMs and only the last wave of the whole linear is partial.
// Each tile accumulates its full K range in one block, like llama.cpp's tiling mode (no stream-K split): a tile's result does not
// depend on the chunk size, the slot count or the number of SMs. The 16-bit output and the bias (a compile-time variant: a runtime
// bias pointer costs the tile code registers) are written once.
// sync (cleared before each launch): [0] ticket, [1 .. S] quantize items done per slot, [S+1 .. 2S] finished tiles per slot.

static constexpr int mmq_pipe_slots      = 3;
static constexpr int mmq_pipe_sync_words = 1 + 2*mmq_pipe_slots;

static constexpr __host__ __device__ mmq_q8_1_ds_layout mmq_chunked_ds_layout(const ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
}

// Q8 ints of one slot: step rows, plus the padding block row MMQ reads past the last row tile. Plain int64 arithmetic: size_t
// operands here cost the tile code ~60 registers.
static __host__ __device__ __forceinline__ int64_t mmq_pipe_slot_ints(const int step, const int padded) {
    return (int64_t(step) * padded / (4*QK8_1) + 4*QK8_1) * int64_t(sizeof(block_q8_1_mmq) / sizeof(int));
}

static __device__ __forceinline__ void mmq_pipe_wait(const unsigned int * counter, const unsigned int target) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        unsigned int delay = 32;
        while (*(volatile const unsigned int *) counter < target) {
            __nanosleep(delay);
            delay = min(delay * 2, 256u);
        }
        __threadfence();
    }
    __syncthreads();
}

static __device__ __forceinline__ void mmq_pipe_signal(unsigned int * counter) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        __threadfence();
        atomicAdd(counter, 1u);
    }
}

template <ggml_type type, typename scalar_t, bool biased>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, 128, false), 1)
static __global__ void mmq_chunked(
        const char * __restrict__ weight, const scalar_t * __restrict__ input, scalar_t * __restrict__ output,
        const scalar_t * __restrict__ bias, int * __restrict__ q8,
        unsigned int * __restrict__ sync, const int rows, const int k, const int n, const int step) {
    constexpr int J        = 128;
    constexpr int I        = ggml_cuda_mmq_get_I(type, J, false);
    constexpr int threads  = ggml_cuda_mmq_get_nthreads(type, J, false);
    constexpr int qk       = ggml_cuda_type_traits<type>::qk;
    constexpr int S        = mmq_pipe_slots;
    constexpr int sz       = sizeof(block_q8_1_mmq) / sizeof(int);
    // Independent input loads in flight per thread. Q6_K's tile code leaves no registers for 8 (spills).
    constexpr int prefetch = type == GGML_TYPE_Q6_K ? 4 : 8;
    constexpr mmq_q8_1_ds_layout layout = mmq_chunked_ds_layout(type);

    extern __shared__ int ids[];
    __shared__ int item_shared;
    const int tid = threadIdx.y*32 + threadIdx.x;
    for (int j = tid; j < J; j += threads) {
        ids[j] = j;
    }
    const int padded    = GGML_PAD(k, MATRIX_ROW_PADDING);
    const int qgrid_y   = (padded + 4*threads - 1) / (4*threads);
    const int64_t slot  = mmq_pipe_slot_ints(step, padded);
    const int chunks    = (rows + step - 1) / step;
    const int row_tiles = step / J;     // per chunk: the last chunk may hold fewer valid rows, its extra items are no-ops
    const int nty       = n / I;
    const int q_items   = row_tiles * qgrid_y;
    const int group     = q_items + row_tiles * nty;
    const int total     = (chunks + 1) * group;

    for (;;) {
        __syncthreads();
        if (tid == 0) {
            item_shared = atomicAdd(sync, 1u);
        }
        __syncthreads();
        const int item = item_shared;
        if (item >= total) {
            break;
        }
        const int g = item / group, pos = item - g*group;
        if (pos < q_items) {
            // Quantize 4*threads columns of a row tile of chunk g into its slot, once the tiles of chunk g-S released it.
            const int c = g;
            if (c >= chunks) {
                continue;
            }
            if (c >= S) {
                mmq_pipe_wait(sync + 1 + S + c % S, unsigned(c / S) * row_tiles * nty);
            }
            const int start = c * step, count = min(step, rows - start);
            const int row0 = (pos / qgrid_y) * J, nrows = min(J, count - row0), cols = pos % qgrid_y;
            const int col = (cols*threads + tid)*4;
            int * dst = q8 + (c % S) * slot;
            for (int base = 0; base < nrows; base += prefetch) {
                float4 values[prefetch];
#pragma unroll
                for (int p = 0; p < prefetch; ++p) {
                    const scalar_t * src = input + int64_t(start + row0 + base + p)*k + col;
                    values[p] = base + p < nrows && col < k ? make_float4(
                        ggml_cuda_cast<float>(src[0]), ggml_cuda_cast<float>(src[1]),
                        ggml_cuda_cast<float>(src[2]), ggml_cuda_cast<float>(src[3])) : make_float4(0, 0, 0, 0);
                }
#pragma unroll
                for (int p = 0; p < prefetch; ++p) {
                    if (base + p < nrows) {
                        quantize_mmq_q8_1_block<scalar_t, layout, true>(
                            dim3(row0 + base + p, cols, 0), dim3(count, qgrid_y, 1), dim3(tid, 0, 0), dim3(threads, 1, 1),
                            input + int64_t(start)*k, nullptr, dst, k, k, int64_t(count)*k, int64_t(count)*k, padded, count, 1, values[p]);
                    }
                }
            }
            mmq_pipe_signal(sync + 1 + c % S);
        } else {
            // Output tile of chunk g-1 (it-major like llama.cpp, so concurrent tiles share weight tiles), once its chunk is quantized.
            if (g == 0) {
                continue;
            }
            const int c = g - 1, tile = pos - q_items;
            const int it = tile / row_tiles, jt = tile - it*row_tiles;
            const int start = c * step, count = min(step, rows - start);
            if (jt * J < count) {
                mmq_pipe_wait(sync + 1 + c % S, unsigned(c / S + 1) * q_items);
                mul_mat_q_process_tile<type, J, false, false, scalar_t>(
                    weight, it*I*(k/qk), q8 + (c % S) * slot + jt*J*sz, ids, nullptr, nullptr, nullptr,
                    k/qk, count, n, n - it*I - 1, count - jt*J - 1, 0, k/qk, output + (int64_t(start) + jt*J)*n + it*I,
                    biased ? bias + it*I : nullptr);
            }
            mmq_pipe_signal(sync + 1 + S + c % S);
        }
    }
}
