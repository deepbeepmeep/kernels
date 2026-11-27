// Copyright (c) 2026 DeepBeepMeep. See LICENSE-WanGP.txt.
// IQ codebooks and block layouts from ggml (MIT). One lane processes 32 values.
#if defined(__HIP_PLATFORM_AMD__)
template<ggml_type T> struct rdna_block;
template<> struct rdna_block<GGML_TYPE_IQ2_XXS> { using type = block_iq2_xxs; };
template<> struct rdna_block<GGML_TYPE_IQ2_XS> { using type = block_iq2_xs; };
template<> struct rdna_block<GGML_TYPE_IQ2_S> { using type = block_iq2_s; };
template<> struct rdna_block<GGML_TYPE_IQ3_XXS> { using type = block_iq3_xxs; };

__device__ __forceinline__ uint32_t iq_signed4(uint32_t grid, uint32_t bits) {
    // All IQ2/IQ3 grid magnitudes are nonzero. Adding one after the XOR
    // completes each selected byte's two's complement without a carry.
    uint32_t ones = (bits & 1) | ((bits & 2) << 7) | ((bits & 4) << 14) | ((bits & 8) << 21);
    return (grid ^ (ones * 255u)) + ones;
}

template<ggml_type T>
__device__ __forceinline__ float rdna_iq_dot(const void * w, const block_q8_1 * a, int block, int ib) {
    const auto & q = ((const typename rdna_block<T>::type *)w)[block];
    const int * input = (const int *)a[ib].qs;
    int sums[2] = {};
    uint32_t aux = 0;
    if constexpr (T == GGML_TYPE_IQ2_XXS) aux = q.qs[4*ib+2] | (uint32_t(q.qs[4*ib+3]) << 16);
    if constexpr (T == GGML_TYPE_IQ3_XXS) {
        const auto * p = (const uint16_t *)(q.qs + QK_K/4) + 2*ib;
        aux = p[0] | (uint32_t(p[1]) << 16);
    }
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        uint2 grid;
        uint32_t signs;
        if constexpr (T == GGML_TYPE_IQ2_XXS) {
            const uint8_t code = ((const uint8_t *)(q.qs + ib * 4))[l];
            grid = ((const uint2 *)iq2xxs_grid)[code];
            signs = ksigns_iq2xs[(aux >> (7*l)) & 127];
        } else if constexpr (T == GGML_TYPE_IQ2_XS) {
            const uint16_t code = q.qs[4*ib+l];
            grid = ((const uint2 *)iq2xs_grid)[code & 511];
            signs = ksigns_iq2xs[code >> 9];
        } else if constexpr (T == GGML_TYPE_IQ2_S) {
            const unsigned code = q.qs[4*ib+l] | ((uint32_t(q.qh[ib]) << (8-2*l)) & 0x300);
            grid = ((const uint2 *)iq2s_grid)[code];
            signs = q.qs[QK_K/8+4*ib+l];
        } else {
            grid = make_uint2(iq3xxs_grid[q.qs[8*ib+2*l]], iq3xxs_grid[q.qs[8*ib+2*l+1]]);
            signs = ksigns_iq2xs[(aux >> (7*l)) & 127];
        }
        int sum = ggml_cuda_dp4a(iq_signed4(grid.x, signs), input[2*l], sums[l/2]);
        sums[l/2] = ggml_cuda_dp4a(iq_signed4(grid.y, signs >> 4), input[2*l+1], sum);
    }
    const float scale = __half2float(q.d) * __low2float(a[ib].ds);
    if constexpr (T == GGML_TYPE_IQ2_XS || T == GGML_TYPE_IQ2_S)
        return scale * .125f * ((2*(q.scales[ib]&15)+1)*sums[0] + (2*(q.scales[ib]>>4)+1)*sums[1]);
    else
        return scale * (T == GGML_TYPE_IQ3_XXS ? .25f : .125f) * (2*(aux>>28)+1) * (sums[0]+sums[1]);
}
#endif
