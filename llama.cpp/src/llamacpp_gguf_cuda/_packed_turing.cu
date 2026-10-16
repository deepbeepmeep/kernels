// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
// Header-free CUDA: NVRTC ships with PyTorch; no toolkit/host compiler required.
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
template<int TYPE> __device__ float rounded(float value) {
    if constexpr (TYPE == 0) return half_float(float_half(value));
    if constexpr (TYPE == 1) return bf16_float(float_bf16(value));
    return value;
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
__device__ float scale(const u8* p) { return half_float(*(const u16*)p); }

// QTYPE: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
template<int Q> __device__ float decode(const u8* raw, long long position) {
    if constexpr (Q < 5) {
        constexpr int size = Q == 0 ? 84 : Q == 1 ? 110 : Q == 2 ? 144 : Q == 3 ? 176 : 210;
        const u8* p = raw + position / 256 * size;
        int lane = position % 256, group = lane / 16;
        if constexpr (Q == 0) {
            int s = p[group], q = (p[16 + lane / 128 * 32 + lane % 32] >> (lane % 128 / 32 * 2)) & 3;
            return (scale(p + 80) * (s & 15)) * q - scale(p + 82) * (s >> 4);
        } else if constexpr (Q == 1) {
            int s = ((p[96 + group % 8] >> (group / 8 * 4)) & 15) |
                    (((p[104 + group % 4] >> (group / 4 * 2)) & 3) << 4);
            int q = ((p[32 + lane / 128 * 32 + lane % 32] >> (lane % 128 / 32 * 2)) & 3)
                    - ((((p[lane % 32] >> (lane / 32)) & 1) ^ 1) * 4);
            return (scale(p + 108) * (s - 32)) * q;
        } else if constexpr (Q == 4) {
            int low = p[lane / 128 * 64 + lane % 64], high = p[128 + lane / 128 * 32 + lane % 32];
            int q = ((low >> (lane % 128 / 64 * 4)) & 15) | (((high >> (lane % 128 / 32 * 2)) & 3) << 4);
            return (scale(p + 208) * ((const signed char*)p)[192 + group]) * (q - 32);
        } else {
            group = lane / 32;
            int a = p[4 + group % 4], b = p[8 + group % 4], c = p[12 + group % 4];
            int s = group < 4 ? a & 63 : (c & 15) | ((a >> 2) & 48);
            int m = group < 4 ? b & 63 : (c >> 4) | ((b >> 2) & 48);
            int q = (p[(Q == 3 ? 48 : 16) + lane / 64 * 32 + lane % 32] >> (group % 2 * 4)) & 15;
            if constexpr (Q == 3) q |= ((p[16 + lane % 32] >> group) & 1) << 4;
            return (scale(p) * s) * q - scale(p + 2) * m;
        }
    } else {
        constexpr int size = Q == 5 ? 18 : Q == 6 ? 20 : Q == 7 ? 22 : Q == 8 ? 24 : 34;
        const u8* p = raw + position / 32 * size;
        int lane = position % 32;
        if constexpr (Q == 9) return scale(p) * ((const signed char*)p)[2 + lane];
        constexpr int offset = Q == 5 ? 2 : Q == 6 ? 4 : Q == 7 ? 6 : 8;
        int q = (p[offset + lane % 16] >> (lane / 16 * 4)) & 15;
        if constexpr (Q == 7 || Q == 8) q |= ((p[offset - 4 + lane / 8] >> (lane % 8)) & 1) << 4;
        if constexpr (Q == 5 || Q == 7) return scale(p) * (q - (Q == 5 ? 8 : 16));
        return scale(p) * q + scale(p + 2);
    }
}

template<int Q> __device__ void decode_four(const u8* raw, unsigned long long position, float* values) {
    if constexpr (Q == 0 || Q == 2 || Q == 3) {
        constexpr int size = Q == 0 ? 84 : Q == 2 ? 144 : 176;
        const u8* p = raw + position / 256 * size;
        unsigned lane = position % 256;
        float d, minimum;
        unsigned offset, shift;
        if constexpr (Q == 0) {
            unsigned s = p[lane / 16];
            d = scale(p + 80) * (s & 15); minimum = scale(p + 82) * (s >> 4);
            offset = 16 + lane / 128 * 32 + lane % 32; shift = lane % 128 / 32 * 2;
        } else {
            unsigned group = lane / 32;
            unsigned a = p[4 + group % 4], b = p[8 + group % 4], c = p[12 + group % 4];
            d = scale(p) * (group < 4 ? a & 63 : (c & 15) | ((a >> 2) & 48));
            minimum = scale(p + 2) * (group < 4 ? b & 63 : (c >> 4) | ((b >> 2) & 48));
            offset = (Q == 3 ? 48 : 16) + lane / 64 * 32 + lane % 32; shift = lane / 32 % 2 * 4;
        }
        u32 low = *(const u32*)(p + offset);
        u32 high = 0;
        if constexpr (Q == 3) high = *(const u32*)(p + 16 + lane % 32);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned q = (low >> (j * 8 + shift)) & (Q == 0 ? 3 : 15);
            if constexpr (Q == 3) q |= ((high >> (j * 8 + lane / 32)) & 1) << 4;
            values[j] = d * q - minimum;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < 4; ++j) values[j] = decode<Q>(raw, position + j);
    }
}

__device__ u32 shared_address(const void* p) {
    u32 address;
    asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }" : "=r"(address) : "l"(p));
    return address;
}
__device__ void load_a(u32* a, const u16* p, int stride) {
    asm volatile("wmma.load.a.sync.aligned.row.m16n16k16.shared.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]),"=r"(a[4]),"=r"(a[5]),"=r"(a[6]),"=r"(a[7]) : "r"(shared_address(p)),"r"(stride) : "memory");
}
__device__ void load_b(u32* b, const u16* p, int stride) {
    asm volatile("wmma.load.b.sync.aligned.col.m16n16k16.shared.f16 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;"
        : "=r"(b[0]),"=r"(b[1]),"=r"(b[2]),"=r"(b[3]),"=r"(b[4]),"=r"(b[5]),"=r"(b[6]),"=r"(b[7]) : "r"(shared_address(p)),"r"(stride) : "memory");
}
__device__ void mma(float* c, const u32* a, const u32* b) {
    asm volatile("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, "
                 "{%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23}, {%0,%1,%2,%3,%4,%5,%6,%7};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),
          "r"(b[0]),"r"(b[1]),"r"(b[2]),"r"(b[3]),"r"(b[4]),"r"(b[5]),"r"(b[6]),"r"(b[7]));
}
__device__ void store_c(float* p, const float* c, int stride) {
    asm volatile("wmma.store.d.sync.aligned.row.m16n16k16.shared.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8}, %9;"
        :: "r"(shared_address(p)),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),"f"(c[4]),"f"(c[5]),"f"(c[6]),"f"(c[7]),"r"(stride) : "memory");
}

extern "C" __global__ void packed_mm(const void* X, const u8* W, const void* Bias, void* Y,
                                     int M, long long stride_m, long long stride_k) {
    constexpr int WM = BM >= 64 ? 32 : 16, WN = BN >= 64 ? 32 : 16;
    constexpr int SK = BK + 8; // Skew shared rows to avoid WMMA bank conflicts.
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32, VALUES = WM * WN / 32;
    // Input tiles and the output staging tile have disjoint lifetimes.
    __shared__ __align__(32) union {
        struct { u16 sx[BM * SK], sw[BN * SK]; } packed;
        struct { float fx[BM * BK], fw[BN * BK]; } exact;
        float result[BM * BN];
    } storage;
    u16* sx = storage.packed.sx; u16* sw = storage.packed.sw;
    float* fx = storage.exact.fx; float* fw = storage.exact.fw;
    float* result = storage.result;
    int tid = threadIdx.x, warp = tid / 32;
    int wm = warp / (BN / WN) * WM, wn = warp % (BN / WN) * WN;
    int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    float accum[WM / 16][WN / 16][8] = {}, exact[VALUES] = {};
    // BF16 has fewer mantissa bits but a wider exponent range than FP16.
    // An exact power-of-two scale keeps small activations (including GELU's
    // negative tail) representable on FP16 tensor cores. Check the round trip
    // for every value; tiles outside this range keep their original FP32 values.
    // The two factors of 256 are removed once, from the FP32 accumulator.
    for (int start = 0; start < K; start += BK) {
        bool outside_half = false;
        for (int i = tid; i < BM * BK; i += THREADS) {
            int m = m0 + i / BK, k = start + i % BK;
            float value = m < M && k < K ? rounded<DTYPE>(read_value<XTYPE>(X, m * stride_m + k * stride_k)) : 0;
            if constexpr (DTYPE == 2) fx[i] = value;
            else {
                u16 half = float_half(DTYPE == 1 ? value * 256.0f : value);
                sx[i / BK * SK + i % BK] = half;
                if constexpr (DTYPE == 1) outside_half |= half_float(half) * (1.0f / 256.0f) != value;
            }
        }
        for (int i = tid * 4; i < BN * BK; i += THREADS * 4) {
            int n = n0 + i / BK, k = start + i % BK;
            float values[4] = {};
            if constexpr (K % 4 == 0) {
                if (n < N && k < K) decode_four<QTYPE>(W, (unsigned long long)n * K + k, values);
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j) if (n < N && k + j < K) values[j] = decode<QTYPE>(W, (long long)n * K + k + j);
            }
            unsigned long long halves = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float value = rounded<DTYPE>(values[j]);
                if constexpr (DTYPE == 2) fw[i + j] = value;
                else {
                    u16 half = float_half(DTYPE == 1 ? value * 256.0f : value);
                    halves |= (unsigned long long)half << (j * 16);
                    if constexpr (DTYPE == 1) outside_half |= half_float(half) * (1.0f / 256.0f) != value;
                }
            }
            if constexpr (DTYPE != 2) *(unsigned long long*)(sw + i / BK * SK + i % BK) = halves;
        }
        // Preserve BF16's exponent range. Exceptional tiles use FP32 arithmetic
        // here, without a dense weight buffer or a host-side retry.
        bool use_exact = __syncthreads_or(outside_half || DTYPE == 2);
        if (use_exact) {
            if constexpr (DTYPE == 1) {
                // Only exceptional BF16 tiles need FP32 staging. Reuse the
                // shared-memory allocation; the normal MMA loop stores halves.
                for (int i = tid; i < BM * BK; i += THREADS) {
                    int m = m0 + i / BK, k = start + i % BK;
                    fx[i] = m < M && k < K ? rounded<DTYPE>(read_value<XTYPE>(X, m * stride_m + k * stride_k)) : 0;
                }
                for (int i = tid; i < BN * BK; i += THREADS) {
                    int n = n0 + i / BK, k = start + i % BK;
                    fw[i] = n < N && k < K ? rounded<DTYPE>(decode<QTYPE>(W, (long long)n * K + k)) : 0;
                }
                __syncthreads();
            }
            #pragma unroll
            for (int j = 0; j < VALUES; ++j) {
                int index = tid + j * THREADS, row = index / BN, col = index % BN;
                #pragma unroll 1
                for (int k = 0; k < BK; ++k) exact[j] = __fmaf_rn(fx[row * BK + k], fw[col * BK + k], exact[j]);
            }
        } else {
            #pragma unroll
            for (int k = 0; k < BK; k += 16) {
                u32 a[WM / 16][8], b[WN / 16][8];
                #pragma unroll
                for (int i = 0; i < WM / 16; ++i) load_a(a[i], sx + (wm + i * 16) * SK + k, SK);
                #pragma unroll
                for (int j = 0; j < WN / 16; ++j) load_b(b[j], sw + (wn + j * 16) * SK + k, SK);
                #pragma unroll
                for (int i = 0; i < WM / 16; ++i) {
                    #pragma unroll
                    for (int j = 0; j < WN / 16; ++j) mma(accum[i][j], a[i], b[j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) store_c(result + (wm + i * 16) * BN + wn + j * 16, accum[i][j], BN);
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < VALUES; ++j) {
        int index = tid + j * THREADS, m = m0 + index / BN, n = n0 + index % BN;
        if (m < M && n < N) {
            float value = result[index] * (DTYPE == 1 ? 1.0f / 65536.0f : 1.0f) + exact[j];
            if constexpr (HAS_BIAS) value += read_value<DTYPE>(Bias, n);
            write_value<DTYPE>(Y, (long long)m * N + n, value);
        }
    }
}
