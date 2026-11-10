// Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE-WanGP.txt)
// Candidates of an output head from a one-bit copy of it. A drafter's head scores its whole vocabulary to keep a handful of tokens:
// here each row of the head is its signs (one bit per weight) and one scale, 26 times less to read than Q6_K, and the kernel keeps the
// KEEP best rows of each group of rows. The caller scores those rows exactly with the head's own kernel.
// Row score = scale * sum(sign * x), x as a sign and PLANES magnitude bits per value (d = amax / (2^PLANES - 1) per 64 values, FP32):
// the 64 values of a block are 2 PLANES words of 64 bits (positive and negative values that have magnitude bit p), so that the sum of
// the x whose weight bit is set is PLANES pairs of population counts, P; sum(sign * x) = 2 P - sum(x). Three magnitude bits and a scale
// per 64 values keep the same candidates as int8 activations with a scale per 32 (the head's best token in 99.9 % of 6,000 real cases
// of a drafter). Integer sums are exact and the float sums run in a fixed order: a row's score does not depend on the launch.
// The copy is stored by group and by 16 bytes of a row (128 weights): bits[(group * K / 128 + j) * R + r] is part j of row r, so that the
// threads of a warp, a row each, load one contiguous run (rows stored one after the other took twice the time: 94 us against 47 for
// 98,304 rows of 5,120 weights on an RTX 5090, the time of reading 60 MB once).
// Header-free CUDA (NVRTC ships with PyTorch), any compute capability. Defines from the host: XTYPE (0 FP16, 1 BF16, 2 FP32), KEEP,
// THREADS (of a block, a multiple of 32: one row per thread when a group has no more rows than that).
using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

#define KMAX 8192   // input features at most
#define RMAX 1024   // rows of a group at most
#define PLANES 3
#define WORDS (2 * PLANES)

__device__ float half_float(u16 value) {
    float result;
    asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value));
    return result;
}
__device__ float bf16_float(u16 value) {
    union { u32 bits; float value; } data;
    data.bits = u32(value) << 16;
    return data.value;
}
__device__ __forceinline__ float lowest() {  // minus infinity
    union { u32 bits; float value; } data;
    data.bits = 0xff800000u;
    return data.value;
}
template<int TYPE> __device__ float read_value(const void* data, long long index) {
    if constexpr (TYPE == 0) return half_float(((const u16*)data)[index]);
    if constexpr (TYPE == 1) return bf16_float(((const u16*)data)[index]);
    return ((const float*)data)[index];
}

// Activations per 64 values (a warp per block, two values per lane): v = round(x / d) in -(2^PLANES - 1) .. 2^PLANES - 1,
// d = amax / (2^PLANES - 1). planes[block][2 p], [2 p + 1]: the values > 0 and < 0 whose magnitude has bit p; d[block] = d,
// d[K / 64 + block] = d * sum(v), what a row of positive signs would get.
extern "C" __global__ void __launch_bounds__(256) head_quantize(const void* x, long long x_stride, u64* planes, float* d, int K) {
    const int block = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31, row = blockIdx.y, blocks = K / 64;
    if (block >= blocks) return;
    const float low = read_value<XTYPE>(x, row * x_stride + block * 64 + lane), high = read_value<XTYPE>(x, row * x_stride + block * 64 + 32 + lane);
    float amax = fmaxf(fabsf(low), fabsf(high));
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    const float scale = amax / (float)((1 << PLANES) - 1);
    const int v0 = amax == 0.0f ? 0 : (int)roundf(low / scale), v1 = amax == 0.0f ? 0 : (int)roundf(high / scale);
    const int m0 = v0 < 0 ? -v0 : v0, m1 = v1 < 0 ? -v1 : v1;
    int sum = v0 + v1;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    u64* words = planes + ((long long)row * blocks + block) * WORDS;
#pragma unroll
    for (int p = 0; p < PLANES; ++p) {
        u32 positive0 = (v0 > 0 && ((m0 >> p) & 1)) ? 1u << lane : 0u, negative0 = (v0 < 0 && ((m0 >> p) & 1)) ? 1u << lane : 0u;
        u32 positive1 = (v1 > 0 && ((m1 >> p) & 1)) ? 1u << lane : 0u, negative1 = (v1 < 0 && ((m1 >> p) & 1)) ? 1u << lane : 0u;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            positive0 |= __shfl_xor_sync(0xffffffffu, positive0, mask);
            negative0 |= __shfl_xor_sync(0xffffffffu, negative0, mask);
            positive1 |= __shfl_xor_sync(0xffffffffu, positive1, mask);
            negative1 |= __shfl_xor_sync(0xffffffffu, negative1, mask);
        }
        if (lane == 0) {
            words[2 * p] = (u64)positive0 | ((u64)positive1 << 32);
            words[2 * p + 1] = (u64)negative0 | ((u64)negative1 << 32);
        }
    }
    if (lane == 0) {
        d[(long long)row * 2 * blocks + block] = scale;
        d[(long long)row * 2 * blocks + blocks + block] = scale * (float)sum;
    }
}

__device__ __forceinline__ int selected(u64 mask, const u64* words) {  // the sum of the 64 activation values whose weight bit is set
    int sum = 0;
#pragma unroll
    for (int p = PLANES - 1; p >= 0; --p) sum = 2 * sum + __popcll(mask & words[2 * p]) - __popcll(mask & words[2 * p + 1]);
    return sum;
}

// One block per group of R rows and per activation row: out[(row * groups + group) * KEEP ..] = ids of its KEEP best rows.
extern "C" __global__ void __launch_bounds__(THREADS) head_candidates(const u32* bits, const float* row_scale, const int* ids, const u64* planes, const float* d, int* out, int K, int R) {
    __shared__ __align__(16) u64 xw[KMAX / 64 * WORDS];
    __shared__ float xs[KMAX / 64];
    __shared__ float score[RMAX];
    __shared__ float best_value[THREADS / 32];
    __shared__ int best_row[THREADS / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, group = blockIdx.x, row = blockIdx.y, blocks = K / 64;
    for (int i = tid; i < blocks * WORDS; i += THREADS) xw[i] = planes[(long long)row * blocks * WORDS + i];
    for (int i = tid; i < blocks; i += THREADS) xs[i] = d[(long long)row * 2 * blocks + i];
    __syncthreads();
    float all = 0.0f;  // sum(x), the same sum in the same order in every thread
    for (int b = 0; b < blocks; ++b) all += d[(long long)row * 2 * blocks + blocks + b];
    for (int r = tid; r < R; r += THREADS) {
        const long long index = (long long)group * R + r;
        const uint4* words = (const uint4*)bits + (long long)group * (blocks >> 1) * R + r;
        float set = 0.0f;
        for (int b = 0; b < blocks; b += 2) {
            const uint4 m = __ldg(words + (long long)(b >> 1) * R);
            set = __fmaf_rn(xs[b], (float)selected((u64)m.x | ((u64)m.y << 32), xw + WORDS * b), set);
            set = __fmaf_rn(xs[b + 1], (float)selected((u64)m.z | ((u64)m.w << 32), xw + WORDS * (b + 1)), set);
        }
        score[r] = row_scale[index] * (2.0f * set - all);
    }
    __syncthreads();
#pragma unroll
    for (int round = 0; round < KEEP; ++round) {  // the best row not taken yet; of equal scores, the first
        float value = lowest();
        int at = 0x7fffffff;
        for (int r = tid; r < R; r += THREADS) {
            const float s = score[r];
            if (s > value || (s == value && r < at)) { value = s; at = r; }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other_value = __shfl_xor_sync(0xffffffffu, value, mask);
            const int other_at = __shfl_xor_sync(0xffffffffu, at, mask);
            if (other_value > value || (other_value == value && other_at < at)) { value = other_value; at = other_at; }
        }
        if (lane == 0) { best_value[warp] = value; best_row[warp] = at; }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int w = 1; w < THREADS / 32; ++w)
                if (best_value[w] > value || (best_value[w] == value && best_row[w] < at)) { value = best_value[w]; at = best_row[w]; }
            at = at < R ? at : round;  // scores that are not numbers: any row
            out[((long long)row * gridDim.x + group) * KEEP + round] = ids[(long long)group * R + at];
            score[at] = lowest();
        }
        __syncthreads();
    }
}
