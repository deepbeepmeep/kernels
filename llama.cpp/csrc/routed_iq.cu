// Copyright (c) 2026 DeepBeepMeep. See LICENSE-WanGP.txt.
// Quantized dot products and block definitions are from ggml (MIT, _vendor/llama.cpp/ggml/LICENSE).
#include "vecdotq.cuh"
#include "gpu_compat.h"
#include "routed_iq_mma.cuh"
#include "routed_iq_rdna.cuh"

#ifdef _WIN32
#define IQ_API extern "C" __declspec(dllexport)
#else
#define IQ_API extern "C" __attribute__((visibility("default")))
#endif

// Divide the CTA into 32-lane groups on NVIDIA and RDNA. Packed integer dot products
// below use ggml's CUDA DP4A or AMD dot4 paths, specialized at compile time for the ISA.
template<int WIDTH>
__device__ __forceinline__ float sum_lanes(float x) {
    for (int d = WIDTH / 2; d; d /= 2) x += __shfl_xor_sync(WGP_WARP_MASK, x, d, WIDTH);
    return x;
}

template<bool BF16>
__global__ void pack_input(const void * x, block_q8_1 * y, int blocks) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx >> 5) >= blocks) return;
    float value;
    if constexpr (BF16) value = __uint_as_float(uint32_t(((const uint16_t *)x)[idx]) << 16);
    else value = ((const float *)x)[idx];
    float a = fabsf(value);
    for (int d = 16; d; d >>= 1) a = fmaxf(a, __shfl_xor_sync(WGP_WARP_MASK, a, d, 32));
    const float scale = a / 127.0f;
    const int lane = idx & 31;
    block_q8_1 & out = y[idx >> 5];
    out.qs[lane] = a > 0 ? int8_t(roundf(value * (127.0f / a))) : 0;
    const float sum = sum_lanes<32>(value);
    if (lane == 0) out.ds = __floats2half2_rn(scale, sum);
}

template<ggml_type TYPE> struct dot_traits;
#define IQ_DOT(type, name, stride) template<> struct dot_traits<GGML_TYPE_##type> { \
    using block = block_##name; static constexpr int vdr = stride; \
    __device__ static float dot(const void * w, const block_q8_1 * a, int b, int i) { return vec_dot_##name##_q8_1(w, a, b, i); } };
IQ_DOT(IQ1_S, iq1_s, VDR_IQ1_S_Q8_1_MMVQ)
IQ_DOT(IQ1_M, iq1_m, VDR_IQ1_M_Q8_1_MMVQ)
#if defined(__HIP_PLATFORM_AMD__)
#define RDNA_DOT(t) template<> struct dot_traits<GGML_TYPE_##t> { \
    using block = rdna_block<GGML_TYPE_##t>::type; \
    static constexpr int vdr = ggml_cuda_type_traits<GGML_TYPE_##t>::qi / 8; \
    __device__ static float dot(const void * w, const block_q8_1 * a, int b, int i) { return rdna_iq_dot<GGML_TYPE_##t>(w, a, b, i / vdr); } };
RDNA_DOT(IQ2_XXS) RDNA_DOT(IQ2_XS) RDNA_DOT(IQ2_S) RDNA_DOT(IQ3_XXS)
#undef RDNA_DOT
#else
IQ_DOT(IQ2_XXS, iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ)
IQ_DOT(IQ2_XS, iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ)
IQ_DOT(IQ2_S, iq2_s, VDR_IQ2_S_Q8_1_MMVQ)
IQ_DOT(IQ3_XXS, iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ)
#endif
IQ_DOT(IQ4_NL, iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ)
IQ_DOT(Q2_0, q2_0, VDR_Q2_0_Q8_1_MMVQ)
#undef IQ_DOT

// The rows of a single expert share activation blocks. Each group reduces its own
// bank row, with narrower groups for short rows. The gate and up dot products
// reuse every activation load and a single write holds their capped activation.
template<ggml_type TYPE, bool GATE, int WIDTH>
__global__ void routed_rows(const block_q8_1 * x, const int * ids, const int * pairs, const uint64_t * ptr,
                            const float * probs, float * out, int columns, int rows, int topk, float capg, float capu, int count) {
    const int local = int(blockIdx.y) + int(gridDim.y) * int(blockIdx.z);
    if (local >= count) return;
    const int pair = pairs ? pairs[local] : local;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / WIDTH;
    if (row >= rows) return;
    const int expert = ids[pair], lane = threadIdx.x % WIDTH;
    const uint64_t address = expert >= 0 ? ptr[3 * expert + (GATE ? 0 : 2)] : 0;
    float total = 0.0f, up = 0.0f;
    if (address) {
        using fmt = dot_traits<TYPE>;
        constexpr int qk = ggml_cuda_type_traits<TYPE>::qk;
        constexpr int pieces = ggml_cuda_type_traits<TYPE>::qi / fmt::vdr;
        const int nb = columns / qk;
        const auto * w = ((const typename fmt::block *)address) + row * nb;
        const typename fmt::block * u = nullptr;
        if constexpr (GATE) u = ((const typename fmt::block *)ptr[3 * expert + 1]) + row * nb;
        const block_q8_1 * input = x + int64_t(GATE ? pair / topk : local) * (columns / 32);
        for (int i = lane; i < nb * pieces; i += WIDTH) {
            const int b = i / pieces, part = (i % pieces) * fmt::vdr;
            const auto * a = input + b * (qk / 32);
            total += fmt::dot(w, a, b, part);
            if constexpr (GATE) up += fmt::dot(u, a, b, part);
        }
        total = sum_lanes<WIDTH>(total);
        if constexpr (GATE) {
            up = sum_lanes<WIDTH>(up);
            total = (capg * tanhf(total / capg)) / (1.0f + expf(-total)) * (capu * tanhf(up / capu));
        } else total *= probs[pair];
    }
    if (!lane) out[int64_t(GATE ? local : pair) * rows + row] = total;
}

IQ_API int iq_pack(const void * x, void * y, int blocks, int bf16, cudaStream_t stream) {
    const dim3 grid((blocks + 7) / 8);
    if (bf16) pack_input<true><<<grid,256,0,stream>>>(x,(block_q8_1 *)y,blocks);
    else pack_input<false><<<grid,256,0,stream>>>(x,(block_q8_1 *)y,blocks);
    return (int)cudaGetLastError();
}

IQ_API int iq_rows(const void * x, const int * ids, const int * pairs, const uint64_t * ptr, const float * probs, float * out,
                   int count, int columns, int rows, int topk, int type, int width, float capg, float capu, cudaStream_t stream) {
    if (!count) return 0;
    constexpr int threads = 128;
    const int y = count < 65535 ? count : 65535;
    const dim3 grid((rows * width + threads - 1) / threads, y, (count - 1) / y + 1);
#define LAUNCH(t, g, w) routed_rows<GGML_TYPE_##t, g, w><<<grid,threads,0,stream>>>((const block_q8_1 *)x,ids,pairs,ptr,probs,out,columns,rows,topk,capg,capu,count)
#define CASE(t, g) case GGML_TYPE_##t: if (width == 16) { LAUNCH(t,g,16); } else if(width == 32) { LAUNCH(t,g,32); } else return -1; break;
    switch (type) {
        CASE(IQ1_S,true) CASE(IQ1_M,true) CASE(IQ2_XS,true) CASE(IQ2_XXS,true) CASE(IQ2_S,true) CASE(IQ3_XXS,true)
        CASE(Q2_0,false) CASE(IQ4_NL,false)
        default: return -1;
    }
#undef CASE
#undef LAUNCH
    return (int)cudaGetLastError();
}

IQ_API int iq_grouped_gate(const void * x, const int * ids, const int * pairs, const int2 * tiles,
                           const uint64_t * ptr, float * mid, int pair_base, int count, int columns, int rows,
                           int topk, int type, float capg, float capu, cudaStream_t stream) {
    if (!count) return 0;
    if (rows % 32 || columns % 256 || count > 65535) return -1;
    const dim3 grid(rows / 32, count);
#define GROUP(t) case GGML_TYPE_##t: grouped_gate_mma<GGML_TYPE_##t><<<grid,128,0,stream>>>((const half *)x,ids,pairs,tiles,ptr,mid,pair_base,columns,rows,topk,capg,capu); break;
    switch (type) {
        GROUP(IQ1_S) GROUP(IQ1_M) GROUP(IQ2_XXS) GROUP(IQ2_XS) GROUP(IQ2_S) GROUP(IQ3_XXS)
        default: return -1;
    }
#undef GROUP
    return (int)cudaGetLastError();
}
