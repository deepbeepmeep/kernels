#include "gpu_compat.h"
// PTQ1 decode: existing Prism FWHT and llama.cpp DP4A arithmetic, fused I/O.
// See THIRD_PARTY_NOTICES.md and the vendored llama.cpp MIT license.
#include <ATen/ATen.h>
#ifdef small
#undef small
#endif
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "vecdotq.cuh"

template <typename T>
__global__ void prism_prepare_decode_kernel(const T * x, const int8_t * signs, block_q8_1 * out,
                                     int64_t width, bool inverse, int nk, int rep, int hd) {
    __shared__ float scratch[1024];
    const int t = threadIdx.x;
    const int64_t chunk = int64_t(blockIdx.x) * 1024;
    const int64_t row = chunk / width;
    const int offset = chunk % width;
    float v[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = offset + t + 256*j;
        const int src = nk ? ((col / hd % rep) * nk + col / hd / rep) * hd + col % hd : col;
        v[j] = float(x[row * width + src]);
        if (!inverse) v[j] *= float(signs[col]);
    }
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(WGP_WARP_MASK, v[j], stride, 32);
            v[j] = (t & stride) ? other - v[j] : v[j] + other;
        }
    }
    #pragma unroll
    for (int stride = 32; stride < 256; stride *= 2) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) scratch[t + 256*j] = v[j];
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = scratch[(t + 256*j) ^ stride];
            v[j] = (t & stride) ? other - v[j] : v[j] + other;
        }
        __syncthreads();
    }
    float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = (a+b)+(c+d); v[1] = (a-b)+(c-d);
    v[2] = (a+b)-(c+d); v[3] = (a-b)-(c-d);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = offset + t + 256*j;
        float result = v[j] * (1.0f / 32.0f);
        if (inverse) result *= float(signs[col]);
        // Preserve the original Hadamard output rounding before Q8 quantization.
        const float xi = float(T(result));
        const float amax = warp_reduce_max<32>(fabsf(xi));
        const float sum = warp_reduce_sum<32>(xi);
        const float scale = amax / 127.0f;
        const int64_t index = chunk + t + 256*j;
        out[index / 32].qs[index % 32] = amax == 0.0f ? 0 : roundf(xi / scale);
        if (index % 32 == 0) out[index / 32].ds = make_half2(scale, sum);
    }
}

template <typename T, int NW, int NR, bool TAIL>
__global__ __launch_bounds__(32 * NW, 1) void prism_decode_kernel(
        const block_ptq1_0 * __restrict__ weights, const block_q8_1 * __restrict__ x, T * __restrict__ output,
        const T * __restrict__ bias, int rows, int blocks) {
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int row = blockIdx.x * NR;
    float sums[NR] = {};
    for (int k = tid; k < blocks; k += 32 * NW) {
        #pragma unroll
        for (int i = 0; i < NR; ++i) {
            if (!TAIL || row + i < rows)
                sums[i] += vec_dot_ptq1_0_q8_1(weights, x + k * 4, (row+i)*blocks+k, 0);
        }
    }
    __shared__ float partial[NW > 1 ? NW-1 : 1][NR][32];
    if (threadIdx.y > 0) {
        #pragma unroll
        for (int i = 0; i < NR; ++i) partial[threadIdx.y-1][i][threadIdx.x] = sums[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
    #pragma unroll
    for (int i = 0; i < NR; ++i) {
        #pragma unroll
        for (int w = 0; w < NW-1; ++w) sums[i] += partial[w][i][threadIdx.x];
        const float sum = warp_reduce_sum<32>(sums[i]);
        if (threadIdx.x == i && row+i < rows) {
            // The established API casts the dot product before adding bias.
            const T rounded = T(sum);
            output[row+i] = bias ? T(float(rounded) + float(bias[row+i])) : rounded;
        }
    }
}

at::Tensor prism_decode(at::Tensor input, at::Tensor raw, at::Tensor signs,
                        const c10::optional<at::Tensor> & bias, int64_t rows,
                        int64_t nk, int64_t rep, int64_t hd, int64_t warps, int64_t row_tile) {
    TORCH_CHECK(input.is_cuda() && raw.device() == input.device() && signs.device() == input.device(), "PTQ1 decode inputs must share a CUDA device.");
    TORCH_CHECK(input.dim() > 0 && input.size(-1) > 0 && input.size(-1) % 1024 == 0 && input.numel() == input.size(-1), "PTQ1 decode expects one token with width divisible by 1024.");
    const int64_t width = input.size(-1);
    TORCH_CHECK(raw.scalar_type() == at::kByte && raw.is_contiguous() && rows > 0 && raw.numel() == rows * (width / 128) * sizeof(block_ptq1_0), "Invalid packed PTQ1 weight shape.");
    TORCH_CHECK(signs.scalar_type() == at::kChar && signs.is_contiguous() && signs.numel() == width, "Expected one int8 sign per channel.");
    TORCH_CHECK(nk == 0 || (nk > 0 && rep > 0 && hd > 0 && nk * rep * hd == width), "Invalid grouped GDN dimensions.");
    TORCH_CHECK(!bias || (bias->device() == input.device() && bias->scalar_type() == input.scalar_type() && bias->is_contiguous() && bias->numel() == rows), "Invalid PTQ1 bias.");
    TORCH_CHECK((warps == 1 || warps == 2 || warps == 4 || warps == 8) && (row_tile == 1 || row_tile == 2 || row_tile == 4), "Invalid PTQ1 launch configuration.");
    const c10::cuda::CUDAGuard guard(input.device());
    input = input.contiguous();
    auto shape = input.sizes().vec();
    shape.back() = rows;
    auto out = at::empty(shape, input.options());
    auto quantized = at::empty({width / 32 * int64_t(sizeof(block_q8_1))}, input.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream();
        #define LAUNCH(NW, NR) if (rows % NR) { prism_decode_kernel<scalar_t, NW, NR, true><<<(rows+NR-1)/NR, dim3(32,NW), 0, stream>>>(w, q, out.data_ptr<scalar_t>(), b, rows, width/128); } else { prism_decode_kernel<scalar_t, NW, NR, false><<<rows/NR, dim3(32,NW), 0, stream>>>(w, q, out.data_ptr<scalar_t>(), b, rows, width/128); }
        #define ROWS(NW) switch (row_tile) { case 1: LAUNCH(NW,1); break; case 2: LAUNCH(NW,2); break; case 4: LAUNCH(NW,4); break; }
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "prism_decode", [&] {
        auto q = reinterpret_cast<block_q8_1 *>(quantized.data_ptr());
        prism_prepare_decode_kernel<scalar_t><<<width/1024, 256, 0, stream>>>(input.data_ptr<scalar_t>(), signs.data_ptr<int8_t>(), q, width, false, nk, rep, hd);
        const auto w = reinterpret_cast<const block_ptq1_0 *>(raw.data_ptr());
        const scalar_t * b = bias ? bias->data_ptr<scalar_t>() : nullptr;
        switch (warps) { case 1: ROWS(1); break; case 2: ROWS(2); break; case 4: ROWS(4); break; case 8: ROWS(8); break; }
    });
    #undef ROWS
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
