#pragma once

// The block helper is shared by ordinary MMQ launches and the cooperative row loop.
template <typename src_t, mmq_q8_1_ds_layout ds_layout, bool prefetched = false>
static __device__ __forceinline__ void quantize_mmq_q8_1_block(
        const dim3 block, const dim3 grid, const dim3 thread, const dim3 dimensions,
        const src_t * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const float4 values = make_float4(0, 0, 0, 0)) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) dimensions.x * block.y + thread.x) * 4;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = block.x;
    const int64_t i2 = block.z % ne2;
    const int64_t i3 = block.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib0 = block.z * ((int64_t) grid.x * grid.y * dimensions.x / QK8_1);
    const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * ne1 + block.x;
    const int64_t iqs = i0 % (4 * QK8_1);

    const float4 xi = prefetched ? values : i0 < ne00
        ? make_float4(
            ggml_cuda_cast<float>(x[i03 * s03 + i02 * s02 + i01 * s01 + i00 + 0]),
            ggml_cuda_cast<float>(x[i03 * s03 + i02 * s02 + i01 * s01 + i00 + 1]),
            ggml_cuda_cast<float>(x[i03 * s03 + i02 * s02 + i01 * s01 + i00 + 2]),
            ggml_cuda_cast<float>(x[i03 * s03 + i02 * s02 + i01 * s01 + i00 + 3]))
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(WGP_WARP_MASK, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(WGP_WARP_MASK, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    char4 q;
    q.x = amax > 0.0f ? roundf(xi.x * d_inv) : 0;
    q.y = amax > 0.0f ? roundf(xi.y * d_inv) : 0;
    q.z = amax > 0.0f ? roundf(xi.z * d_inv) : 0;
    q.w = amax > 0.0f ? roundf(xi.w * d_inv) : 0;

    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs / 4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }
        y[ib].d2s6[2 + iqs / 16] = sum;
        if (iqs % 64 != 0) {
            return;
        }
        const float d = amax > 0.0f ? 1.0f / d_inv : 0.0f;
        y[ib].d2s6[iqs / 64] = d;
        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = amax > 0.0f ? 1.0f / d_inv : 0.0f;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs / 32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs / 32] = d;
    }
}
