// Normalized Sylvester FWHT used by Prism Hadamard-folded checkpoints.
// The transform is block-local (1024 values), with FP32 accumulation.
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

template <typename T>
__global__ void prism_hadamard_kernel(const T * x, const int8_t * signs, T * out,
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
            const float other = __shfl_xor_sync(0xffffffff, v[j], stride);
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
        out[chunk + t + 256*j] = T(result);
    }
}

at::Tensor prism_hadamard(at::Tensor input, at::Tensor signs, bool inverse,
                         int64_t nk, int64_t rep, int64_t hd) {
    TORCH_CHECK(input.is_cuda() && signs.device() == input.device(), "Hadamard inputs must be on the same CUDA device.");
    TORCH_CHECK(input.dim() > 0 && input.size(-1) > 0 && input.size(-1) % 1024 == 0, "Prism Hadamard requires a width divisible by 1024.");
    TORCH_CHECK(signs.scalar_type() == at::kChar && signs.is_contiguous() && signs.numel() == input.size(-1), "Expected one contiguous int8 sign per input channel.");
    TORCH_CHECK(nk == 0 || (!inverse && nk > 0 && rep > 0 && hd > 0 && nk * rep * hd == input.size(-1)), "Invalid grouped GDN dimensions.");
    const c10::cuda::CUDAGuard guard(input.device());
    input = input.contiguous();
    auto out = at::empty_like(input);
    if (input.numel() == 0) return out;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "prism_hadamard", [&] {
        prism_hadamard_kernel<scalar_t><<<input.numel()/1024, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(), signs.data_ptr<int8_t>(), out.data_ptr<scalar_t>(), input.size(-1), inverse, nk, rep, hd);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
