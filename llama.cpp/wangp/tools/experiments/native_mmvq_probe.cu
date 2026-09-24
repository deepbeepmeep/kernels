// Test-only MMVQ scheduling probe. Derived from llama.cpp's MIT-licensed
// ggml-cuda/mmvq.cu and WanGP's typed Q8 quantizer. The build records and copies
// the corresponding upstream LICENSE files alongside this standalone DLL.
// No extension installation, production dispatch, or global scratch allocation.
#include "convert.cuh"
#include "vecdotq.cuh"
#include <cstdint>

#ifdef _WIN32
#define PROBE_EXPORT extern "C" __declspec(dllexport)
#else
#define PROBE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

template <typename T>
__launch_bounds__(256, 1)
static __global__ void probe_quantize(const T * x, block_q8_1 * y, int64_t cols, int64_t padded_cols) {
    // Same operations, rounding and warp reductions as the production typed
    // MMVQ quantizer (without the unrelated SiLU specialization or PDL hints).
    const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= padded_cols) return;
    const int64_t index = static_cast<int64_t>(blockIdx.y) * padded_cols + col;
    const int64_t ib = index / QK8_1;
    const int iqs = index % QK8_1;
    const float xi = col < cols ? ggml_cuda_cast<float>(x[static_cast<int64_t>(blockIdx.y) * cols + col]) : 0.0f;
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum = warp_reduce_sum<QK8_1>(xi);
    const float d = amax / 127.0f;
    y[ib].qs[iqs] = amax == 0.0f ? 0 : roundf(xi / d);
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

template <ggml_type TYPE, int ROWS, bool DISTRIBUTED>
__launch_bounds__(128, 1)
static __global__ void probe_mmvq(const void * __restrict__ weight,
        const block_q8_1 * __restrict__ y, float * __restrict__ output,
        int out_features, int in_features, int q8_stride) {
    constexpr int COLS = 3;
    constexpr int WARPS = 4; // Must not change: fixes each lane's K subsequence.
    constexpr int QI = ggml_cuda_type_traits<TYPE>::qi;
    constexpr int VDR = TYPE == GGML_TYPE_Q4_K ? VDR_Q4_K_Q8_1_MMVQ : VDR_Q6_K_Q8_1_MMVQ;
    constexpr int BLOCKS_PER_ITER = VDR * WARPS * 32 / QI;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int row0 = blockIdx.x * ROWS;
    const int blocks_per_row = in_features / QK_K;
    const int kqs = VDR * (tid % (QI / VDR));
    float sums[COLS][ROWS] = {{0.0f}};

    // Keep the production loop order and the original vecdot implementation.
    // ROWS changes output tiling only, never the integer dots or FP32 K tree.
    for (int kbx = tid / (QI / VDR); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kby = kbx * (QK_K / QK8_1);
#pragma unroll
        for (int row = 0; row < ROWS; ++row) {
#pragma unroll
            for (int col = 0; col < COLS; ++col) {
                const int weight_block = (row0 + row) * blocks_per_row + kbx;
                if constexpr (TYPE == GGML_TYPE_Q4_K) {
                    sums[col][row] += vec_dot_q4_K_q8_1(weight, y + col * q8_stride + kby, weight_block, kqs);
                } else {
                    sums[col][row] += vec_dot_q6_K_q8_1(weight, y + col * q8_stride + kby, weight_block, kqs);
                }
            }
        }
    }

    if constexpr (!DISTRIBUTED) {
        __shared__ float partial[WARPS - 1][COLS][ROWS][32];
        if (warp > 0) {
#pragma unroll
            for (int col = 0; col < COLS; ++col)
#pragma unroll
                for (int row = 0; row < ROWS; ++row)
                    partial[warp - 1][col][row][lane] = sums[col][row];
        }
        __syncthreads();
        if (warp > 0) return;
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
#pragma unroll
                for (int other = 0; other < WARPS - 1; ++other)
                    sums[col][row] += partial[other][col][row][lane];
                const float value = warp_reduce_sum<32>(sums[col][row]);
                // Preserve even the production row2 kernel's selected lane.
                if (lane == ((row0 + row) & 1)) output[col * out_features + row0 + row] = value;
            }
        }
    } else {
        __shared__ float partial[WARPS][COLS][ROWS][32];
#pragma unroll
        for (int col = 0; col < COLS; ++col)
#pragma unroll
            for (int row = 0; row < ROWS; ++row)
                partial[warp][col][row][lane] = sums[col][row];
        __syncthreads();
#pragma unroll
        for (int col = 0; col < COLS; ++col) {
#pragma unroll
            for (int row = 0; row < ROWS; ++row) {
                if (warp == (col * ROWS + row) % WARPS) {
                    float value = partial[0][col][row][lane];
#pragma unroll
                    for (int other = 1; other < WARPS; ++other)
                        value += partial[other][col][row][lane];
                    value = warp_reduce_sum<32>(value);
                    if (lane == ((row0 + row) & 1)) output[col * out_features + row0 + row] = value;
                }
            }
        }
    }
}

// dtype: 0=FP32, 1=FP16, 2=BF16. Exactly three activation rows.
PROBE_EXPORT int mmvq_probe_quantize(const void * x, void * y, int dtype, int cols, int padded_cols, void * stream_ptr) {
    if (!x || !y || cols <= 0 || cols % 256 || padded_cols < cols || padded_cols % 512) return cudaErrorInvalidValue;
    const dim3 grid((padded_cols + 255) / 256, 3, 1);
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    switch (dtype) {
        case 0: probe_quantize<<<grid, 256, 0, stream>>>(static_cast<const float *>(x), static_cast<block_q8_1 *>(y), cols, padded_cols); break;
        case 1: probe_quantize<<<grid, 256, 0, stream>>>(static_cast<const half *>(x), static_cast<block_q8_1 *>(y), cols, padded_cols); break;
        case 2: probe_quantize<<<grid, 256, 0, stream>>>(static_cast<const nv_bfloat16 *>(x), static_cast<block_q8_1 *>(y), cols, padded_cols); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

template <ggml_type TYPE, int ROWS, bool DISTRIBUTED>
static int launch(const void * weight, const void * q8, void * output, int m, int k, int q8_stride, cudaStream_t stream) {
    if (m % ROWS) return cudaErrorInvalidValue;
    probe_mmvq<TYPE, ROWS, DISTRIBUTED><<<m / ROWS, dim3(32, 4), 0, stream>>>(
        weight, static_cast<const block_q8_1 *>(q8), static_cast<float *>(output), m, k, q8_stride);
    return cudaGetLastError();
}

template <ggml_type TYPE>
static int dispatch(int variant, const void * w, const void * q, void * o, int m, int k, int stride, cudaStream_t s) {
    switch (variant) {
        case 0: return launch<TYPE, 2, false>(w, q, o, m, k, stride, s); // Control.
        case 1: return launch<TYPE, 2, true >(w, q, o, m, k, stride, s);
        case 2: return launch<TYPE, 1, false>(w, q, o, m, k, stride, s);
        case 3: return launch<TYPE, 4, false>(w, q, o, m, k, stride, s);
        case 4: return launch<TYPE, 4, true >(w, q, o, m, k, stride, s);
        default: return cudaErrorInvalidValue;
    }
}

PROBE_EXPORT int mmvq_probe_launch(const void * weight, const void * q8, void * output,
        int qtype, int variant, int m, int k, int q8_stride, void * stream_ptr) {
    if (!weight || !q8 || !output || m <= 0 || k <= 0 || k % 256 || q8_stride < k / 32) return cudaErrorInvalidValue;
    const auto stream = static_cast<cudaStream_t>(stream_ptr);
    if (qtype == GGML_TYPE_Q4_K) return dispatch<GGML_TYPE_Q4_K>(variant, weight, q8, output, m, k, q8_stride, stream);
    if (qtype == GGML_TYPE_Q6_K) return dispatch<GGML_TYPE_Q6_K>(variant, weight, q8, output, m, k, q8_stride, stream);
    return cudaErrorInvalidValue;
}
