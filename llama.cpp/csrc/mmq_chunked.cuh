#pragma once

#include <cooperative_groups.h>

// A resident grid reuses the Q8 input buffer for each row chunk. Keep the same
// virtual stream-K partitions as separate llama.cpp launches, including their
// reverse-order fixup, so FP32 accumulation and the final rounding are unchanged.
static __device__ __forceinline__ uint3 mmq_chunk_fastdiv(uint32_t d) {
    uint32_t l = 0;
    while (l < 32 && (uint32_t{1} << l) < d) ++l;
    const uint32_t m = uint32_t((uint64_t{1} << 32) * ((uint64_t{1} << l) - d) / d + 1);
    return make_uint3(m, l, d);
}

template <ggml_type type, typename scalar_t>
static __device__ __forceinline__ void mmq_chunk_mat(
        const char * weight, const int * q8, scalar_t * output, float * completed, float * partial,
        int b, int blocks, int rows, int k, int n, uint3 nk, uint3 ntx) {
    constexpr int J = 128;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
    constexpr int threads = ggml_cuda_mmq_get_nthreads(type, J, false);
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int per_iter = ggml_cuda_mmq_get_K_vram(type, J, false) / qk;
    extern __shared__ int ids[];
    for (int j = threadIdx.y*32 + threadIdx.x; j < J; j += threads) ids[j] = j;
    __syncthreads();
    const int total = ntx.z * ((n + I - 1) / I) * nk.z;
    int begin = int64_t(b) * total / blocks;
    int end = int64_t(b + 1) * total / blocks;
    begin -= fastmodulo(begin, nk) % per_iter;
    end -= fastmodulo(end, nk) % per_iter;
    while (begin < end) {
        const uint2 ij = fast_div_modulo(fastdiv(begin, nk), ntx);
        const int it = ij.x, jt = ij.y;
        const int lo = fastmodulo(begin, nk);
        const int hi = min(int(nk.z), lo + end - begin);
        const int * y = q8 + jt*J*(sizeof(block_q8_1_mmq) / sizeof(int));
        scalar_t * dst = output + int64_t(jt)*J*n + it*I;
        if (hi == int(nk.z)) {
            mul_mat_q_process_tile<type, J, false, false, scalar_t>(
                weight, it*I*(k/qk), y, ids, completed, partial, nullptr,
                k/qk, rows, n, n-it*I-1, rows-jt*J-1, lo, hi, dst, b);
        } else {
            mul_mat_q_process_tile<type, J, false, true, scalar_t>(
                weight, it*I*(k/qk), y, ids, completed, partial, nullptr,
                k/qk, rows, n, n-it*I-1, rows-jt*J-1, lo, hi, dst, b);
        }
        begin += hi - lo;
    }
}

template <ggml_type type, typename scalar_t>
static __device__ __forceinline__ void mmq_chunk_fixup(
        scalar_t * output, const float * completed, const float * partial,
        int b, int stripe, int blocks, int rows, int n, uint3 nk, uint3 ntx) {
    constexpr int J = 128;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
    constexpr int warps = ggml_cuda_mmq_get_nthreads(type, J, false) / 32;
    constexpr int per_iter = ggml_cuda_mmq_get_K_vram(type, J, false) / ggml_cuda_type_traits<type>::qk;
    const int total = ntx.z * ((n + I - 1) / I) * nk.z;
    int begin = int64_t(b) * total / blocks;
    int end = int64_t(b + 1) * total / blocks;
    begin -= fastmodulo(begin, nk) % per_iter;
    end -= fastmodulo(end, nk) % per_iter;
    if (begin == end || fastmodulo(begin, nk) == 0 ||
            (fastdiv(begin, nk) == fastdiv(end, nk) && fastmodulo(end, nk) != 0)) return;
    const int i = stripe*32 + threadIdx.x;
    float sum[J / warps] = {};
    int stop = begin;
    for (int prev = b - 1; ; --prev) {
        int start = int64_t(prev) * total / blocks;
        start -= fastmodulo(start, nk) % per_iter;
        if (start != stop) {
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += warps) {
                const int j = j0 + threadIdx.y;
                sum[j0/warps] += partial[prev*(J*I) + j*I + i];
            }
            if (fastmodulo(start, nk) == 0 || fastdiv(start, nk) < fastdiv(begin, nk)) break;
        }
        stop = start;
    }
    const uint2 ij = fast_div_modulo(fastdiv(begin, nk), ntx);
    scalar_t * dst = output + int64_t(ij.y)*J*n + ij.x*I;
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += warps) {
        const int j = j0 + threadIdx.y;
        if (j >= rows - int(ij.y)*J) return;
        dst[j*n + i] = ggml_cuda_cast<scalar_t>(completed[b*(J*I) + j*I + i] + sum[j0/warps]);
    }
}

template <ggml_type type, typename scalar_t>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, 128, false), 1)
static __global__ void mmq_chunked(
        const char * weight, const scalar_t * input, scalar_t * output,
        int * q8, float * completed, float * partial, int rows, int k, int n, int step, int nsm) {
    constexpr int J = 128;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, false);
    constexpr int threads = ggml_cuda_mmq_get_nthreads(type, J, false);
    constexpr auto layout = (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
        ? MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
    const auto grid = cooperative_groups::this_grid();
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    const uint3 nk = mmq_chunk_fastdiv(k / ggml_cuda_type_traits<type>::qk);
    for (int start = 0; start < rows; start += step) {
        const int count = min(step, rows - start);
        const int qgrid_y = (padded + 4*threads - 1) / (4*threads);
        // The matrix tile limits resident warps. Load independent input groups
        // ahead of their shuffle/quantization work to hide memory latency without
        // adding a global buffer or changing any quantization arithmetic.
        constexpr int prefetch = 8;
        for (int base = blockIdx.x; base < count*qgrid_y; base += gridDim.x*prefetch) {
            float4 values[prefetch];
#pragma unroll
            for (int p = 0; p < prefetch; ++p) {
                const int b = base + p*gridDim.x;
                const int col = ((b/count)*threads + tid)*4;
                const scalar_t * src = input + int64_t(start+b%count)*k + col;
                values[p] = b < count*qgrid_y && col < k ? make_float4(
                    ggml_cuda_cast<float>(src[0]), ggml_cuda_cast<float>(src[1]),
                    ggml_cuda_cast<float>(src[2]), ggml_cuda_cast<float>(src[3])) : make_float4(0, 0, 0, 0);
            }
#pragma unroll
            for (int p = 0; p < prefetch; ++p) {
                const int b = base + p*gridDim.x;
                if (b < count*qgrid_y) {
                    quantize_mmq_q8_1_block<scalar_t, layout, true>(
                        dim3(b % count, b / count, 0), dim3(count, qgrid_y, 1), dim3(tid, 0, 0), dim3(threads, 1, 1),
                        input + int64_t(start)*k, nullptr, q8, k, k, int64_t(count)*k, int64_t(count)*k, padded, count, 1, values[p]);
                }
            }
        }
        grid.sync();
        const uint3 ntx = mmq_chunk_fastdiv((count + J - 1) / J);
        const int tiles = ntx.z * ((n + I - 1) / I);
        const int efficiency = 100 * tiles / (nsm*((tiles + nsm - 1) / nsm));
        const int blocks = efficiency >= 90 ? tiles : nsm;
        scalar_t * dst = output + int64_t(start)*n;
        for (int b = blockIdx.x; b < blocks; b += gridDim.x) {
            mmq_chunk_mat<type>(weight, q8, dst, completed, partial, b, blocks, count, k, n, nk, ntx);
            __syncthreads();
        }
        if (tiles % blocks != 0) {
            grid.sync();
            for (int b = blockIdx.x; b < blocks*(I/32); b += gridDim.x) {
                mmq_chunk_fixup<type>(dst, completed, partial, b % blocks, b / blocks, blocks, count, n, nk, ntx);
            }
            // Fixup reads only the compact partials, so the next chunk may
            // quantize Q8 immediately. Its quantization barrier also completes
            // every preceding fixup before any partial storage is reused.
        } else if (start + step < rows) {
            grid.sync();
        }
    }
}
