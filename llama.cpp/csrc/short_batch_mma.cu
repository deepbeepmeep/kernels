// Q4_K and PTQ1_0 short-batch linear on INT8 tensor cores (compute capability 8.0+).
//
// Speculative verification multiplies each weight row by 2-8 activation rows.
// The dp4a MMVQ kernels read each weight row with 4-8 byte loads spread over
// many rows, which caps cache-cold throughput near 1.1 TB/s on an RTX 5090.
// Here each block streams a 32-row weight tile into shared memory with
// coalesced 16-byte cp.async copies (double buffered), then INT8 tensor cores
// (mma.m16n8k32.s8) form one exact integer dot product per 32-value Q4_K
// sub-block, as the dp4a path does. Four warps split the super-blocks and
// each activation fragment is reused for two 16-row MMA tiles.
//
// Activation quantization matches quantize_mmvq_q8_1_typed: d = amax / 127,
// q = roundf(x / d), including the fused SiLU-multiply rounding. The Q4_K
// minimum term uses d * sum(q) exactly, like MMVQ's dp4a sum of q.
//
// PTQ1_0 (Prism ternary, 128 base-3 values in 28 bytes) uses the same
// staging and MMA shape. Each thread decodes its packed trits once for all
// activation rows, instead of once per row as the dp4a kernels must.
//
// Tiles were tuned on an RTX 5090 (compute capability 12.0), where the path is
// on by default. Other architectures keep MMVQ unless the caller measured this
// path faster for a shape (short_batch_set_decision) or forces a mode.
#include "short_batch_mma.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <map>
#include <mutex>
#include <tuple>

namespace {

constexpr int kWarps = 4;      // split-K warps per block
constexpr int kTiles = 2;      // 16-row MMA tiles per block
constexpr int kStages = 2;     // cp.async pipeline depth
constexpr int kRows = 16 * kTiles;
constexpr int kPiece = kWarps * 144;                                   // bytes per row per chunk
constexpr int kStride = kPiece + ((32 - kPiece % 128) + 128) % 128;    // = 32 mod 128: conflict-free fragment reads
constexpr int kStageBytes = kRows * kStride;
constexpr int kSharedBytes = kStages * kStageBytes > kWarps * kTiles * 32 * 4 * 4 ? kStages * kStageBytes : kWarps * kTiles * 32 * 4 * 4;

template <typename T> __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
template <> __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_float(float v) { return static_cast<T>(v); }
template <> __device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }

template <typename src_t, bool SILU_MUL>
__global__ void quantize_rows_kernel(const src_t * __restrict__ x, int8_t * __restrict__ xq, float2 * __restrict__ xds, const int cols) {
    const int blocks = cols / 32;
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int row = blockIdx.y, lane = threadIdx.x & 31;
    if (b >= blocks) {
        return;
    }
    const int col = b * 32 + lane;
    float xi;
    if constexpr (SILU_MUL) {
        const size_t source = static_cast<size_t>(row) * (2 * cols) + col;
        const float gate = to_float(x[source]);
        const float value = to_float(x[source + cols]);
        const float activated = to_float(from_float<src_t>(__fdiv_rn(gate, 1.0f + expf(-gate))));
        xi = to_float(from_float<src_t>(activated * value));
    } else {
        xi = to_float(x[static_cast<size_t>(row) * cols + col]);
    }
    float amax = fabsf(xi);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : static_cast<int>(roundf(xi / d));
    int sum = q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    xq[static_cast<size_t>(row) * cols + col] = static_cast<int8_t>(q);
    if (lane == 0) {
        xds[static_cast<size_t>(row) * blocks + b] = make_float2(d, d * static_cast<float>(sum));
    }
}

// cp.async and the s8 MMA need compute capability 8.0; older targets compile
// trapping stubs that dispatch never selects.
__device__ __forceinline__ void cp_async16(void * smem, const void * gmem, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(gmem), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    __trap();
#endif
}

__device__ __forceinline__ int scale_byte(const uint32_t (&w)[3], int k) { return (w[k >> 2] >> (8 * (k & 3))) & 0xFF; }

__device__ __forceinline__ void q4k_scale_min(const uint32_t (&w)[3], int j, float & sc, float & m) {
    if (j < 4) {
        sc = scale_byte(w, j) & 63;
        m = scale_byte(w, j + 4) & 63;
    } else {
        const int a = scale_byte(w, j + 4);
        sc = (a & 0xF) | ((scale_byte(w, j - 4) >> 6) << 4);
        m = (a >> 4) | ((scale_byte(w, j) >> 6) << 4);
    }
}

// MMA fragments use a permuted k order: quad thread t owns the contiguous
// values 8t..8t+7 of a sub-block for both A (weights) and B (activations).
// A dot product is invariant to that shared permutation, and each Q4_K byte
// pair supplies sub-block 2p (low nibbles) and 2p + 1 (high nibbles).
template <int NC, typename out_t>
__global__ void __launch_bounds__(kWarps * 32) q4k_mma_kernel(const uint8_t * __restrict__ W, const int8_t * __restrict__ xq, const float2 * __restrict__ xds,
                                                             out_t * __restrict__ out, const int K, const int rows) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * kRows, nsb = K / 256, kb = K / 32, nchunks = (nsb + kWarps - 1) / kWarps;
    const size_t row_bytes = static_cast<size_t>(nsb) * 144;
    constexpr int chunks16 = kPiece / 16;
    auto load_chunk = [&](int chunk, int stage) {
        uint8_t * dst = smem + stage * kStageBytes;
        const size_t col0 = static_cast<size_t>(chunk) * kPiece;
        for (int i = tid; i < kRows * chunks16; i += kWarps * 32) {
            const int r = i / chunks16, o = i - r * chunks16;
            const bool valid = row0 + r < rows && col0 + o * 16 < row_bytes;
            cp_async16(dst + r * kStride + o * 16, W + (valid ? static_cast<size_t>(row0 + r) * row_bytes + col0 + o * 16 : 0), valid);
        }
    };
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < nchunks) {
            load_chunk(s, s);
        }
        cp_async_commit();
    }
    const bool col_ok = g < NC;
    const int8_t * xcol = xq + static_cast<size_t>(col_ok ? g : 0) * K + 8 * t;
    const int c0 = min(2 * t, NC - 1), c1 = min(2 * t + 1, NC - 1);
    float acc[kTiles][4] = {};
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (chunk + kStages - 1 < nchunks) {
            load_chunk(chunk + kStages - 1, (chunk + kStages - 1) % kStages);
        }
        cp_async_commit();
        const int sb = chunk * kWarps + warp;
        if (sb >= nsb) {
            continue;
        }
        uint2 bv[8];
        float2 x0[8], x1[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            bv[j] = col_ok ? __ldg(reinterpret_cast<const uint2 *>(xcol + sb * 256 + 32 * j)) : make_uint2(0, 0);
            x0[j] = __ldg(xds + c0 * kb + sb * 8 + j);
            x1[j] = __ldg(xds + c1 * kb + sb * 8 + j);
        }
#pragma unroll
        for (int m = 0; m < kTiles; ++m) {
            const uint8_t * ba = smem + (chunk % kStages) * kStageBytes + (16 * m + g) * kStride + warp * 144;
            const uint8_t * bb = ba + 8 * kStride;
            const uint4 ha = *reinterpret_cast<const uint4 *>(ba), hb = *reinterpret_cast<const uint4 *>(bb);
            uint2 qa[4], qb[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                qa[p] = *reinterpret_cast<const uint2 *>(ba + 16 + 32 * p + 8 * t);
                qb[p] = *reinterpret_cast<const uint2 *>(bb + 16 + 32 * p + 8 * t);
            }
            const float2 dma = __half22float2(*reinterpret_cast<const half2 *>(&ha.x)), dmb = __half22float2(*reinterpret_cast<const half2 *>(&hb.x));
            const uint32_t sa[3] = {ha.y, ha.z, ha.w}, sbv[3] = {hb.y, hb.z, hb.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int p = j >> 1, shift = 4 * (j & 1);
                int c[4];
                mma_s8(c, (qa[p].x >> shift) & 0x0F0F0F0F, (qb[p].x >> shift) & 0x0F0F0F0F, (qa[p].y >> shift) & 0x0F0F0F0F, (qb[p].y >> shift) & 0x0F0F0F0F, bv[j].x, bv[j].y);
                float sca, ma, scb, mb;
                q4k_scale_min(sa, j, sca, ma);
                q4k_scale_min(sbv, j, scb, mb);
                const float da = dma.x * sca, mna = dma.y * ma, db = dmb.x * scb, mnb = dmb.y * mb;
                acc[m][0] += da * x0[j].x * static_cast<float>(c[0]) - mna * x0[j].y;
                acc[m][1] += da * x1[j].x * static_cast<float>(c[1]) - mna * x1[j].y;
                acc[m][2] += db * x0[j].x * static_cast<float>(c[2]) - mnb * x0[j].y;
                acc[m][3] += db * x1[j].x * static_cast<float>(c[3]) - mnb * x1[j].y;
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    float * partial = reinterpret_cast<float *>(smem);
#pragma unroll
    for (int m = 0; m < kTiles; ++m) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            partial[((warp * kTiles + m) * 32 + lane) * 4 + i] = acc[m][i];
        }
    }
    __syncthreads();
    if (warp >= kTiles) {
        return;
    }
    // Warp m sums tile m over the split-K warps in a fixed order.
    float v[4] = {};
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] += partial[((w * kTiles + warp) * 32 + lane) * 4 + i];
        }
    }
    const int r0 = row0 + 16 * warp + g, r1 = r0 + 8;
    if (2 * t < NC) {
        if (r0 < rows) out[static_cast<size_t>(2 * t) * rows + r0] = from_float<out_t>(v[0]);
        if (r1 < rows) out[static_cast<size_t>(2 * t) * rows + r1] = from_float<out_t>(v[2]);
    }
    if (2 * t + 1 < NC) {
        if (r0 < rows) out[static_cast<size_t>(2 * t + 1) * rows + r0] = from_float<out_t>(v[1]);
        if (r1 < rows) out[static_cast<size_t>(2 * t + 1) * rows + r1] = from_float<out_t>(v[3]);
    }
}

// ---- PTQ1_0 ----------------------------------------------------------------------
// Byte i < 16 holds values i + 16n (n = 0..4); byte 16 + i' holds 80 + i' + 8n;
// qh byte b holds 120 + b + 2n (n = 0..3). Quad thread t decodes qs word t once
// and supplies for each 32-value activation block j the k-slots {4t..4t+3} (X)
// and {16+4t..16+4t+3} (Y); activations are gathered in the same order.
__device__ __forceinline__ uint32_t trit_step(uint32_t & lo, uint32_t & hi) {
    const uint32_t wl = lo * 3, wh = hi * 3;
    lo = wl & 0x00FF00FF;
    hi = wh & 0x00FF00FF;
    return __vsub4(__byte_perm(wl, wh, 0x7531), 0x01010101);  // four signed trits
}

template <int WARPS> struct PtqTile {
    static constexpr int PIECE = WARPS * 56;                                    // two 128-value blocks per warp per chunk
    static constexpr int STRIDE = PIECE + ((16 - PIECE % 128) + 128) % 128;     // = 16 mod 128: conflict-free 4 B reads
    static constexpr int BYTES = 16 * STRIDE;
    static constexpr int SHARED = kStages * BYTES > WARPS * 32 * 4 * 4 ? kStages * BYTES : WARPS * 32 * 4 * 4;
};

template <int NC, int WARPS, typename out_t>
__global__ void __launch_bounds__(WARPS * 32) ptq1_mma_kernel(const uint8_t * __restrict__ W, const int8_t * __restrict__ xq, const float2 * __restrict__ xds,
                                                             out_t * __restrict__ out, const int K, const int rows) {
    using T = PtqTile<WARPS>;
    extern __shared__ __align__(16) uint8_t smem[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 16, nunits = K / 256, kb = K / 32, nchunks = (nunits + WARPS - 1) / WARPS;
    const size_t row_bytes = static_cast<size_t>(K / 128) * 28;
    constexpr int chunks16 = T::PIECE / 16;
    auto load_chunk = [&](int chunk, int stage) {
        uint8_t * dst = smem + stage * T::BYTES;
        const size_t col0 = static_cast<size_t>(chunk) * T::PIECE;
        for (int i = tid; i < 16 * chunks16; i += WARPS * 32) {
            const int r = i / chunks16, o = i - r * chunks16;
            const bool valid = row0 + r < rows && col0 + o * 16 < row_bytes;
            cp_async16(dst + r * T::STRIDE + o * 16, W + (valid ? static_cast<size_t>(row0 + r) * row_bytes + col0 + o * 16 : 0), valid);
        }
    };
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < nchunks) {
            load_chunk(s, s);
        }
        cp_async_commit();
    }
    const bool col_ok = g < NC;
    const int8_t * xcol = xq + static_cast<size_t>(col_ok ? g : 0) * K;
    const int c0 = min(2 * t, NC - 1), c1 = min(2 * t + 1, NC - 1);
    const int ox2 = 64 + 4 * t, oy2 = 80 + 4 * (t >> 1) + 8 * (t & 1), ox3 = 96 + 8 * t, oy3 = 100 + 8 * t;
    float acc[4] = {};
    for (int chunk = 0; chunk < nchunks; ++chunk) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (chunk + kStages - 1 < nchunks) {
            load_chunk(chunk + kStages - 1, (chunk + kStages - 1) % kStages);
        }
        cp_async_commit();
        const int unit = chunk * WARPS + warp;
        if (unit >= nunits) {
            continue;
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int blk = unit * 2 + half;
            const int8_t * xb = xcol + blk * 128;
            uint32_t bx[4] = {}, by[4] = {};
            if (col_ok) {
                bx[0] = __ldg(reinterpret_cast<const uint32_t *>(xb + 4 * t));
                by[0] = __ldg(reinterpret_cast<const uint32_t *>(xb + 16 + 4 * t));
                bx[1] = __ldg(reinterpret_cast<const uint32_t *>(xb + 32 + 4 * t));
                by[1] = __ldg(reinterpret_cast<const uint32_t *>(xb + 48 + 4 * t));
                bx[2] = __ldg(reinterpret_cast<const uint32_t *>(xb + ox2));
                by[2] = __ldg(reinterpret_cast<const uint32_t *>(xb + oy2));
                if (t < 3) {
                    bx[3] = __ldg(reinterpret_cast<const uint32_t *>(xb + ox3));
                    by[3] = __ldg(reinterpret_cast<const uint32_t *>(xb + oy3));
                } else {
                    const uint32_t lo = __ldg(reinterpret_cast<const uint32_t *>(xb + 120)), hi = __ldg(reinterpret_cast<const uint32_t *>(xb + 124));
                    bx[3] = __byte_perm(lo, hi, 0x6420);
                    by[3] = __byte_perm(lo, hi, 0x7531);
                }
            }
            float d0[4], d1[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                d0[j] = __ldg(xds + c0 * kb + blk * 4 + j).x;
                d1[j] = __ldg(xds + c1 * kb + blk * 4 + j).x;
            }
            uint32_t fx[2][4], fy[2][4];
            float dw[2];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const uint8_t * b = smem + (chunk % kStages) * T::BYTES + (g + 8 * rr) * T::STRIDE + warp * 56 + half * 28;
                const uint32_t a = *reinterpret_cast<const uint32_t *>(b + 4 * t);
                const uint32_t w0 = *reinterpret_cast<const uint32_t *>(b + 16), w1 = *reinterpret_cast<const uint32_t *>(b + 20);
                const uint32_t tail = *reinterpret_cast<const uint32_t *>(b + 24);  // qh[0], qh[1], d
                dw[rr] = __half2float(__ushort_as_half(static_cast<unsigned short>(tail >> 16)));
                uint32_t lo = __byte_perm(a, 0, 0x4140), hi = __byte_perm(a, 0, 0x4342);
                uint32_t qa[5];
#pragma unroll
                for (int n = 0; n < 5; ++n) {
                    qa[n] = trit_step(lo, hi);
                }
                uint32_t l0 = __byte_perm(w0, 0, 0x4140), h0 = __byte_perm(w0, 0, 0x4342), l1 = __byte_perm(w1, 0, 0x4140), h1 = __byte_perm(w1, 0, 0x4342);
                uint32_t q0[5], q1[5];
#pragma unroll
                for (int n = 0; n < 5; ++n) {
                    q0[n] = trit_step(l0, h0);
                    q1[n] = trit_step(l1, h1);
                }
                fx[rr][0] = qa[0];
                fy[rr][0] = qa[1];
                fx[rr][1] = qa[2];
                fy[rr][1] = qa[3];
                fx[rr][2] = qa[4];
                fy[rr][2] = (t >> 1) ? ((t & 1) ? q1[1] : q1[0]) : ((t & 1) ? q0[1] : q0[0]);
                if (t < 3) {
                    fx[rr][3] = t == 0 ? q0[2] : (t == 1 ? q0[3] : q0[4]);
                    fy[rr][3] = t == 0 ? q1[2] : (t == 1 ? q1[3] : q1[4]);
                } else {
                    uint32_t v = (tail & 0xFF) | ((tail & 0xFF00) << 8), even = 0, odd = 0;
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const uint32_t w = v * 3;
                        v = w & 0x00FF00FF;
                        even |= ((w >> 8) & 0xFF) << (8 * n);
                        odd |= ((w >> 24) & 0xFF) << (8 * n);
                    }
                    fx[rr][3] = __vsub4(even, 0x01010101);
                    fy[rr][3] = __vsub4(odd, 0x01010101);
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int c[4];
                mma_s8(c, fx[0][j], fx[1][j], fy[0][j], fy[1][j], bx[j], by[j]);
                acc[0] += dw[0] * d0[j] * static_cast<float>(c[0]);
                acc[1] += dw[0] * d1[j] * static_cast<float>(c[1]);
                acc[2] += dw[1] * d0[j] * static_cast<float>(c[2]);
                acc[3] += dw[1] * d1[j] * static_cast<float>(c[3]);
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();
    float * partial = reinterpret_cast<float *>(smem);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        partial[(warp * 32 + lane) * 4 + i] = acc[i];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    float v[4] = {};
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] += partial[(w * 32 + lane) * 4 + i];
        }
    }
    const int r0 = row0 + g, r1 = r0 + 8;
    if (2 * t < NC) {
        if (r0 < rows) out[static_cast<size_t>(2 * t) * rows + r0] = from_float<out_t>(v[0]);
        if (r1 < rows) out[static_cast<size_t>(2 * t) * rows + r1] = from_float<out_t>(v[2]);
    }
    if (2 * t + 1 < NC) {
        if (r0 < rows) out[static_cast<size_t>(2 * t + 1) * rows + r0] = from_float<out_t>(v[1]);
        if (r1 < rows) out[static_cast<size_t>(2 * t + 1) * rows + r1] = from_float<out_t>(v[3]);
    }
}

template <typename src_t>
void quantize_rows(const at::Tensor & input, int8_t * xq, float2 * xds, int rows, int cols, bool silu_mul, cudaStream_t stream) {
    const dim3 grid((cols / 32 + 7) / 8, rows);
    const src_t * x = reinterpret_cast<const src_t *>(input.data_ptr());
    if (silu_mul) {
        quantize_rows_kernel<src_t, true><<<grid, 256, 0, stream>>>(x, xq, xds, cols);
    } else {
        quantize_rows_kernel<src_t, false><<<grid, 256, 0, stream>>>(x, xq, xds, cols);
    }
}

template <int NC, typename out_t>
void launch_columns(bool ptq1, const uint8_t * W, const int8_t * xq, const float2 * xds, void * out, int K, int rows, cudaStream_t stream) {
    static_assert(kSharedBytes <= 48 * 1024 && PtqTile<8>::SHARED <= 48 * 1024, "dynamic shared memory above the default limit");
    if (!ptq1) {
        q4k_mma_kernel<NC, out_t><<<(rows + kRows - 1) / kRows, kWarps * 32, kSharedBytes, stream>>>(W, xq, xds, static_cast<out_t *>(out), K, rows);
    } else if (rows <= 6144 || K >= 16384) {
        // Fewer output rows or longer rows: more split-K warps keep all SMs busy.
        ptq1_mma_kernel<NC, 8, out_t><<<(rows + 15) / 16, 8 * 32, PtqTile<8>::SHARED, stream>>>(W, xq, xds, static_cast<out_t *>(out), K, rows);
    } else {
        ptq1_mma_kernel<NC, 4, out_t><<<(rows + 15) / 16, 4 * 32, PtqTile<4>::SHARED, stream>>>(W, xq, xds, static_cast<out_t *>(out), K, rows);
    }
}

template <typename out_t>
void dispatch_columns(int columns, bool ptq1, const uint8_t * W, const int8_t * xq, const float2 * xds, void * out, int K, int rows, cudaStream_t stream) {
    switch (columns) {
        case 2: launch_columns<2, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        case 3: launch_columns<3, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        case 4: launch_columns<4, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        case 5: launch_columns<5, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        case 6: launch_columns<6, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        case 7: launch_columns<7, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
        default: launch_columns<8, out_t>(ptq1, W, xq, xds, out, K, rows, stream); break;
    }
}

enum class Mode { Auto, Native, Mma };
std::mutex policy_mutex;
Mode policy_mode = Mode::Auto;
std::map<std::tuple<bool, int64_t, int64_t, int64_t>, bool> decisions;  // (ptq1, rows, N, K)

}  // namespace

bool short_batch_mma_selected(int cc, bool ptq1, int64_t batch_rows, int64_t out_features, int64_t in_features) {
    if (cc < 800 || batch_rows < 2 || batch_rows > 8 || in_features % 256 != 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(policy_mutex);
    if (policy_mode != Mode::Auto) {
        return policy_mode == Mode::Mma;
    }
    const auto it = decisions.find({ptq1, batch_rows, out_features, in_features});
    return it != decisions.end() ? it->second : cc == 1200;
}

void short_batch_set_mode(const std::string & mode) {
    TORCH_CHECK(mode == "auto" || mode == "native" || mode == "mma", "short-batch mode must be auto, native or mma, got ", mode);
    std::lock_guard<std::mutex> lock(policy_mutex);
    policy_mode = mode == "mma" ? Mode::Mma : mode == "native" ? Mode::Native : Mode::Auto;
}

std::string short_batch_mode() {
    std::lock_guard<std::mutex> lock(policy_mutex);
    return policy_mode == Mode::Mma ? "mma" : policy_mode == Mode::Native ? "native" : "auto";
}

void short_batch_set_decision(const std::string & qtype_name, int64_t batch_rows, int64_t out_features, int64_t in_features, bool enabled) {
    TORCH_CHECK(qtype_name == "Q4_K" || qtype_name == "PTQ1_0", "short-batch tensor-core path supports Q4_K and PTQ1_0, got ", qtype_name);
    std::lock_guard<std::mutex> lock(policy_mutex);
    decisions[{qtype_name == "PTQ1_0", batch_rows, out_features, in_features}] = enabled;
}

void short_batch_clear_decisions() {
    std::lock_guard<std::mutex> lock(policy_mutex);
    decisions.clear();
}

void short_batch_mma_linear(const at::Tensor & raw_weight, bool ptq1, int64_t out_features, int64_t in_features, const at::Tensor & input, bool silu_mul, at::Tensor & output) {
    const int rows = static_cast<int>(input.size(0)), K = static_cast<int>(in_features), N = static_cast<int>(out_features);
    const int64_t row_bytes = ptq1 ? static_cast<int64_t>(K / 128) * 28 : static_cast<int64_t>(K / 256) * 144;
    TORCH_CHECK(raw_weight.numel() * raw_weight.element_size() >= static_cast<int64_t>(N) * row_bytes, "GGUF weight is smaller than its logical shape");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.device().index()).stream();
    at::Tensor xq = at::empty({rows, K}, input.options().dtype(at::kChar));
    at::Tensor xds = at::empty({rows, K / 32, 2}, input.options().dtype(at::kFloat));
    int8_t * q = xq.data_ptr<int8_t>();
    float2 * ds = reinterpret_cast<float2 *>(xds.data_ptr<float>());
    switch (input.scalar_type()) {
        case at::kFloat: quantize_rows<float>(input, q, ds, rows, K, silu_mul, stream); break;
        case at::kHalf: quantize_rows<__half>(input, q, ds, rows, K, silu_mul, stream); break;
        case at::kBFloat16: quantize_rows<__nv_bfloat16>(input, q, ds, rows, K, silu_mul, stream); break;
        default: TORCH_CHECK(false, "Unsupported GGUF CUDA input dtype for linear: ", input.scalar_type());
    }
    const uint8_t * W = static_cast<const uint8_t *>(raw_weight.data_ptr());
    switch (output.scalar_type()) {
        case at::kFloat: dispatch_columns<float>(rows, ptq1, W, q, ds, output.data_ptr(), K, N, stream); break;
        case at::kHalf: dispatch_columns<__half>(rows, ptq1, W, q, ds, output.data_ptr(), K, N, stream); break;
        case at::kBFloat16: dispatch_columns<__nv_bfloat16>(rows, ptq1, W, q, ds, output.data_ptr(), K, N, stream); break;
        default: TORCH_CHECK(false, "Unsupported GGUF CUDA output dtype for linear: ", output.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
