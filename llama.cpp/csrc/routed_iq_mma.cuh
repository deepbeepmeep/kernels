// Copyright (c) 2026 DeepBeepMeep. See LICENSE-WanGP.txt.
// GGML's dequantize.cuh supplies the IQ grid and scale decoding (MIT).
#include "dequantize.cuh"
#if defined(__HIP_PLATFORM_AMD__)
#include <rocwmma/rocwmma.hpp>
namespace iq_matrix = rocwmma;
using iq_half = rocwmma::float16_t;
#else
#include <mma.h>
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
namespace iq_matrix = nvcuda::wmma;
using iq_half = half;
#endif
#endif

template<ggml_type T>
__device__ __forceinline__ void unpack_iq_tile(const void * bank, int block, half * values, int lane) {
    if constexpr (T == GGML_TYPE_IQ1_S) dequantize_iq1_s(bank, block, values, lane);
    if constexpr (T == GGML_TYPE_IQ1_M) dequantize_iq1_m(bank, block, values, lane);
    if constexpr (T == GGML_TYPE_IQ2_XXS) dequantize_iq2_xxs(bank, block, values, lane);
    if constexpr (T == GGML_TYPE_IQ2_XS) dequantize_iq2_xs(bank, block, values, lane);
    if constexpr (T == GGML_TYPE_IQ2_S) dequantize_iq2_s(bank, block, values, lane);
    if constexpr (T == GGML_TYPE_IQ3_XXS) dequantize_iq3_xxs(bank, block, values, lane);
}

// One CTA produces 32 capped gate/up columns for up to 16 tokens routed to one
// expert. Each warp expands its IQ grid once per 256-column tile. The matrix
// instructions accumulate in float32. A padded routing lane never reads x.
template<ggml_type T>
__global__ void grouped_gate_mma(const half * x, const int * ids, const int * pairs, const int2 * tiles,
                                const uint64_t * ptr, float * mid, int pair_base, int columns, int rows,
                                int topk, float capg, float capu) {
#if defined(__HIP_PLATFORM_AMD__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700)
    using namespace iq_matrix;
    constexpr int K = 256, stride = K + 8;
    __shared__ __align__(32) half input[16 * stride];
    __shared__ __align__(32) union { half w[4][16 * stride]; float result[4][16 * 16]; } scratch;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int2 tile = tiles[blockIdx.y];
    const int start = tile.x - pair_base;
    const int expert = ids[pairs[start]];
    const int row = 32 * blockIdx.x + 16 * (warp % 2);
    const void * weight = (const void *)ptr[3 * expert + warp / 2];
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);
    if (weight) {
        for (int k = 0; k < columns; k += K) {
            for (int index = threadIdx.x; index < 16 * K / 4; index += blockDim.x) {
                const int token = index / (K / 4), col = 4 * (index % (K / 4));
                uint64_t value = 0;
                if (token < tile.y) {
                    const int src = pairs[start + token] / topk;
                    value = *(const uint64_t *)(x + int64_t(src) * columns + k + col);
                }
                *(uint64_t *)(input + token * stride + col) = value;
            }
            for (int r = 0; r < 16; ++r)
                unpack_iq_tile<T>(weight, (row + r) * (columns / K) + k / K, scratch.w[warp] + r * stride, lane);
            __syncthreads();
            for (int c = 0; c < K; c += 16) {
                fragment<matrix_a, 16, 16, 16, iq_half, row_major> a;
                fragment<matrix_b, 16, 16, 16, iq_half, col_major> b;
                load_matrix_sync(a, (const iq_half *)(input + c), stride);
                load_matrix_sync(b, (const iq_half *)(scratch.w[warp] + c), stride);
                mma_sync(acc, a, b, acc);
            }
            __syncthreads();
        }
    }
    store_matrix_sync(scratch.result[warp], acc, 16, mem_row_major);
    __syncthreads();
    for (int index = threadIdx.x; index < tile.y * 32; index += blockDim.x) {
        const int token = index / 32, col = index % 32, part = col / 16, i = token * 16 + col % 16;
        const float gate = scratch.result[part][i], up = scratch.result[2 + part][i];
        mid[int64_t(start + token) * rows + 32 * blockIdx.x + col] =
            (capg * tanhf(gate / capg)) / (1.0f + expf(-gate)) * (capu * tanhf(up / capu));
    }
#endif
}
