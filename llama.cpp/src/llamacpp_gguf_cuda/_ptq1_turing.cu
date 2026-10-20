// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
// Portions adapted from llama.cpp (PrismML PTQ1_0 port of ggml-cuda vecdotq.cuh and quantize.cu), MIT License.
// PTQ1_0 linear for 1-8 activation rows on SM75: llama.cpp's PTQ1 x Q8_1 arithmetic (Q8_1 activations with FP16 scales, SIMD base-3
// trit decoding, DP4A, the same float operations per 128-value block), with each weight block decoded once for all rows, the
// activations staged in shared memory and read 16 bytes at a time, several output features per warp. Every output row is computed in
// the same order whatever the number of rows: a row of a speculative verification equals the single-row result bit for bit.
// Header-free CUDA: NVRTC ships with PyTorch. Defines from the host: NCOLS (rows), R (outputs per warp), WARPS, XTYPE, DTYPE, HAS_BIAS;
// the matrix size is a launch argument: one compilation per configuration, whatever the shapes.
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;

__device__ float half_float(u16 value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}
__device__ u16 float_half(float value) {
    u16 result;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(value));
    return result;
}
__device__ float bf16_float(u16 value) {
    union { u32 bits; float value; } data;
    data.bits = u32(value) << 16;
    return data.value;
}
__device__ u16 float_bf16(float value) {
    union { u32 bits; float value; } data;
    data.value = value;
    if ((data.bits & 0x7fffffffU) > 0x7f800000U) return u16((data.bits >> 16) | 64);
    return u16((data.bits + 0x7fffU + ((data.bits >> 16) & 1)) >> 16);
}
template<int TYPE> __device__ float read_value(const void* data, long long index) {
    if constexpr (TYPE == 0) return half_float(((const u16*)data)[index]);
    if constexpr (TYPE == 1) return bf16_float(((const u16*)data)[index]);
    return ((const float*)data)[index];
}
template<int TYPE> __device__ void write_value(void* data, long long index, float value) {
    if constexpr (TYPE == 0) ((u16*)data)[index] = float_half(value);
    else if constexpr (TYPE == 1) ((u16*)data)[index] = float_bf16(value);
    else ((float*)data)[index] = value;
}

#define QSTRIDE 36             // words of a staged 128-value activation block: 32 + 4 of padding, conflict-free 16-byte reads

// Q8_1 activations as llama.cpp's MMVQ entry point: per 32 values, d = amax / 127, q = round(x / d), d kept in FP16.
extern "C" __global__ void __launch_bounds__(256) ptq1_quantize(const void* x, long long x_stride, signed char* q, float* d, int K) {
    const int col = blockIdx.x * 256 + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= K) return;  // K is a multiple of 128: whole warps return
    const float xi = read_value<XTYPE>(x, row * x_stride + col);
    float amax = fabsf(xi);
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    const float scale = amax / 127.0f;
    q[(long long)row * K + col] = amax == 0.0f ? 0 : (signed char)roundf(xi / scale);
    if ((col & 31) == 0) d[(long long)row * (K / 32) + col / 32] = half_float(float_half(scale));
}

// Five trits of each byte of a 32-bit word, two bytes at a time in 16-bit lanes (no carry between bytes): trit = (v * 3) >> 8.
__device__ __forceinline__ int trits(u32& lo, u32& hi) {
    const u32 wl = lo * 3, wh = hi * 3;
    lo = wl & 0x00FF00FF;
    hi = wh & 0x00FF00FF;
    return __vsub4(__byte_perm(wl, wh, 0x7531), 0x01010101);
}

// weight: N rows of K / 128 PTQ1_0 blocks (28 bytes: 24 bytes of five trits, 2 bytes of four trits, FP16 scale)
extern "C" __global__ void __launch_bounds__(WARPS * 32) ptq1_mm(const u8* weight, const signed char* q, const float* d,
                                                                   const void* bias, void* out, int N, int K) {
    const int BLOCKS = K / 128;
    __shared__ __align__(16) int sq[NCOLS][32][QSTRIDE];
    __shared__ __align__(16) float sd[NCOLS][32][4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int first = (blockIdx.x * WARPS + warp) * R;
    float total[R][NCOLS];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) total[r][c] = 0.0f;

    for (int chunk = 0; chunk < BLOCKS; chunk += 32) {
        const int count = min(32, BLOCKS - chunk);
        __syncthreads();
        for (int i = threadIdx.x; i < NCOLS * count * 8; i += WARPS * 32) {
            const int c = i / (count * 8), rest = i % (count * 8), b = rest >> 3, w = (rest & 7) * 4;
            *(int4*)&sq[c][b][w] = *(const int4*)(q + (long long)c * K + (long long)(chunk + b) * 128 + w * 4);
        }
        for (int i = threadIdx.x; i < NCOLS * count; i += WARPS * 32) {
            const int c = i / count, b = i % count;
            *(float4*)sd[c][b] = *(const float4*)(d + (long long)c * (K / 32) + (long long)(chunk + b) * 4);
        }
        __syncthreads();
        if (lane >= count) continue;
        const int block = chunk + lane;
        u32 words[R][7];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = min(first + r, N - 1);  // rows past N repeat the last one, their results are not written
            const u32* p = (const u32*)(weight + ((long long)n * BLOCKS + block) * 28);
#pragma unroll
            for (int i = 0; i < 7; ++i) words[r][i] = p[i];
        }
        int sumi[R][NCOLS];
        float acc[R][NCOLS];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) { sumi[r][c] = 0; acc[r][c] = 0.0f; }
        // llama.cpp's float order per block: acc = sum_k ds_k * sumi_k for k = 0..3 (contracted to FMA by nvcc), then d * acc
#define FLUSH(k) \
        _Pragma("unroll") for (int r = 0; r < R; ++r) \
        _Pragma("unroll") for (int c = 0; c < NCOLS; ++c) { acc[r][c] = __fmaf_rn(sd[c][lane][k], (float)sumi[r][c], acc[r][c]); sumi[r][c] = 0; }

        // qs[0..15]: element 16 t + 4 g + i is trit t of byte 4 g + i
        u32 lo[R][4], hi[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int g = 0; g < 4; ++g) { lo[r][g] = __byte_perm(words[r][g], 0, 0x4140); hi[r][g] = __byte_perm(words[r][g], 0, 0x4342); }
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            int qv[R][4];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < 4; ++g) qv[r][g] = trits(lo[r][g], hi[r][g]);
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                const int4 u = *(const int4*)&sq[c][lane][4 * t];
#pragma unroll
                for (int r = 0; r < R; ++r)
                    sumi[r][c] = __dp4a(qv[r][3], u.w, __dp4a(qv[r][2], u.z, __dp4a(qv[r][1], u.y, __dp4a(qv[r][0], u.x, sumi[r][c]))));
            }
            if (t == 1) { FLUSH(0) }
            if (t == 3) { FLUSH(1) }
        }
        // qs[16..23]: element 80 + 8 t + 4 g + i is trit t of byte 16 + 4 g + i
#pragma unroll
        for (int r = 0; r < R; ++r)
#pragma unroll
            for (int g = 0; g < 2; ++g) { lo[r][g] = __byte_perm(words[r][4 + g], 0, 0x4140); hi[r][g] = __byte_perm(words[r][4 + g], 0, 0x4342); }
#pragma unroll
        for (int t = 0; t < 5; ++t) {
            int qv[R][2];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int g = 0; g < 2; ++g) qv[r][g] = trits(lo[r][g], hi[r][g]);
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                const int2 u = *(const int2*)&sq[c][lane][20 + 2 * t];
#pragma unroll
                for (int r = 0; r < R; ++r) sumi[r][c] = __dp4a(qv[r][1], u.y, __dp4a(qv[r][0], u.x, sumi[r][c]));
            }
            if (t == 1) { FLUSH(2) }
        }
        // qh[0..1]: element 120 + 4 s + i, four trits per byte
#pragma unroll
        for (int r = 0; r < R; ++r) lo[r][0] = (words[r][6] & 0xFF) | ((words[r][6] & 0xFF00) << 8);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int qv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const u32 w0 = lo[r][0] * 3;
                lo[r][0] = w0 & 0x00FF00FF;
                const u32 w1 = lo[r][0] * 3;
                lo[r][0] = w1 & 0x00FF00FF;
                qv[r] = __vsub4(__byte_perm(w0, w1, 0x7531), 0x01010101);
            }
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                const int u = sq[c][lane][30 + s];
#pragma unroll
                for (int r = 0; r < R; ++r) sumi[r][c] = __dp4a(qv[r], u, sumi[r][c]);
            }
        }
        FLUSH(3)
#undef FLUSH
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float scale = half_float(u16(words[r][6] >> 16));
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) total[r][c] += scale * acc[r][c];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int c = 0; c < NCOLS; ++c)
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) total[r][c] += __shfl_xor_sync(0xffffffffu, total[r][c], mask);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = first + r;
            if (n >= N) break;
#pragma unroll
            for (int c = 0; c < NCOLS; ++c) {
                float value = total[r][c];
                if constexpr (HAS_BIAS) value += read_value<DTYPE>(bias, n);
                write_value<DTYPE>(out, (long long)c * N + n, value);
            }
        }
    }
}
