#include "gguf_llamacpp_ops.h"

#ifdef small
#undef small
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdarg>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "_vendor/llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "_vendor/llama.cpp/ggml/src/ggml-cuda/convert.cuh"
#include "_vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cuh"
#include "_vendor/llama.cpp/ggml/src/ggml-cuda/mmvq.cuh"
#include "_vendor/llama.cpp/ggml/src/ggml-cuda/quantize.cuh"

namespace {

size_t gguf_type_size_local(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return sizeof(float);
        case GGML_TYPE_F16:
            return sizeof(ggml_fp16_t);
        case GGML_TYPE_BF16:
            return sizeof(ggml_bf16_t);
        case GGML_TYPE_Q4_0:
            return sizeof(block_q4_0);
        case GGML_TYPE_Q4_1:
            return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0:
            return sizeof(block_q5_0);
        case GGML_TYPE_Q5_1:
            return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0:
            return sizeof(block_q8_0);
        case GGML_TYPE_Q2_K:
            return sizeof(block_q2_K);
        case GGML_TYPE_Q3_K:
            return sizeof(block_q3_K);
        case GGML_TYPE_Q4_K:
            return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K:
            return sizeof(block_q5_K);
        case GGML_TYPE_Q6_K:
            return sizeof(block_q6_K);
        case GGML_TYPE_IQ2_XXS:
            return sizeof(block_iq2_xxs);
        case GGML_TYPE_IQ2_XS:
            return sizeof(block_iq2_xs);
        case GGML_TYPE_IQ3_XXS:
            return sizeof(block_iq3_xxs);
        case GGML_TYPE_IQ1_S:
            return sizeof(block_iq1_s);
        case GGML_TYPE_IQ4_NL:
            return sizeof(block_iq4_nl);
        case GGML_TYPE_IQ3_S:
            return sizeof(block_iq3_s);
        case GGML_TYPE_IQ2_S:
            return sizeof(block_iq2_s);
        case GGML_TYPE_IQ4_XS:
            return sizeof(block_iq4_xs);
        case GGML_TYPE_Q8_1:
            return sizeof(block_q8_1);
        default:
            throw std::runtime_error("Unsupported ggml type in local size helper");
    }
}

int64_t gguf_blck_size_local(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 1;
        case GGML_TYPE_Q4_0:
            return QK4_0;
        case GGML_TYPE_Q4_1:
            return QK4_1;
        case GGML_TYPE_Q5_0:
            return QK5_0;
        case GGML_TYPE_Q5_1:
            return QK5_1;
        case GGML_TYPE_Q8_0:
            return QK8_0;
        case GGML_TYPE_IQ4_NL:
            return QK4_NL;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
            return QK_K;
        case GGML_TYPE_Q8_1:
            return QK8_1;
        default:
            throw std::runtime_error("Unsupported ggml type in local block-size helper");
    }
}

struct gguf_cuda_pool_simple : public ggml_cuda_pool {
    int device;

    explicit gguf_cuda_pool_simple(int device_id) : device(device_id) {
    }

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
        void * ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        if (actual_size != nullptr) {
            *actual_size = size;
        }
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        GGML_UNUSED(size);
        if (ptr == nullptr) {
            return;
        }
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
    }
};

std::mutex g_cuda_info_mutex;
std::unique_ptr<ggml_cuda_device_info> g_cuda_info;
std::mutex g_backend_ctx_mutex;
std::unique_ptr<ggml_backend_cuda_context> g_backend_contexts[GGML_CUDA_MAX_DEVICES];

std::string format_message(const char * fmt, va_list args) {
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    return std::string(buffer);
}

at::ScalarType scalar_type_from_name(const std::string & name) {
    if (name == "float16" || name == "half") {
        return at::kHalf;
    }
    if (name == "bfloat16") {
        return at::kBFloat16;
    }
    if (name == "float32" || name == "float") {
        return at::kFloat;
    }
    throw std::runtime_error("Unsupported GGUF CUDA output dtype: " + name);
}

ggml_backend_cuda_context & get_backend_ctx(int device, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_backend_ctx_mutex);
    auto & holder = g_backend_contexts[device];
    if (holder == nullptr) {
        holder = std::make_unique<ggml_backend_cuda_context>(device);
    }
    holder->curr_stream_no = 0;
    holder->streams[device][0] = stream;
    return *holder;
}

ggml_type ggml_type_from_qtype_name(const std::string & qtype_name) {
    if (qtype_name == "Q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (qtype_name == "Q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (qtype_name == "Q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (qtype_name == "Q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (qtype_name == "Q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (qtype_name == "Q2_K") {
        return GGML_TYPE_Q2_K;
    }
    if (qtype_name == "Q3_K") {
        return GGML_TYPE_Q3_K;
    }
    if (qtype_name == "Q4_K") {
        return GGML_TYPE_Q4_K;
    }
    if (qtype_name == "Q5_K") {
        return GGML_TYPE_Q5_K;
    }
    if (qtype_name == "Q6_K") {
        return GGML_TYPE_Q6_K;
    }
    if (qtype_name == "IQ2_XXS") {
        return GGML_TYPE_IQ2_XXS;
    }
    if (qtype_name == "IQ2_XS") {
        return GGML_TYPE_IQ2_XS;
    }
    if (qtype_name == "IQ3_XXS") {
        return GGML_TYPE_IQ3_XXS;
    }
    if (qtype_name == "IQ1_S") {
        return GGML_TYPE_IQ1_S;
    }
    if (qtype_name == "IQ4_NL") {
        return GGML_TYPE_IQ4_NL;
    }
    if (qtype_name == "IQ3_S") {
        return GGML_TYPE_IQ3_S;
    }
    if (qtype_name == "IQ2_S") {
        return GGML_TYPE_IQ2_S;
    }
    if (qtype_name == "IQ4_XS") {
        return GGML_TYPE_IQ4_XS;
    }
    if (qtype_name == "F16") {
        return GGML_TYPE_F16;
    }
    if (qtype_name == "BF16") {
        return GGML_TYPE_BF16;
    }
    if (qtype_name == "F32") {
        return GGML_TYPE_F32;
    }
    throw std::runtime_error("Unsupported GGUF CUDA qtype: " + qtype_name);
}

ggml_type ggml_type_from_scalar_type(at::ScalarType scalar_type) {
    switch (scalar_type) {
        case at::kFloat:
            return GGML_TYPE_F32;
        case at::kHalf:
            return GGML_TYPE_F16;
        case at::kBFloat16:
            return GGML_TYPE_BF16;
        default:
            throw std::runtime_error("Unsupported scalar dtype for GGUF CUDA conversion");
    }
}

void check_cuda_tensor(const at::Tensor & tensor, const char * name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

ggml_tensor make_quantized_src0(const at::Tensor & raw_weight, ggml_type type, int64_t rows, int64_t cols) {
    ggml_tensor tensor = {};
    tensor.type = type;
    tensor.buffer = nullptr;
    tensor.ne[0] = cols;
    tensor.ne[1] = rows;
    tensor.ne[2] = 1;
    tensor.ne[3] = 1;
    tensor.nb[0] = ggml_type_size(type);
    tensor.nb[1] = ggml_row_size(type, cols);
    tensor.nb[2] = tensor.nb[1] * rows;
    tensor.nb[3] = tensor.nb[2];
    tensor.data = raw_weight.data_ptr();
    return tensor;
}

bool gguf_cuda_should_use_mmq_local(ggml_type type, int cc, int64_t batch_rows) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            break;
        default:
            return false;
    }
    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }
#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || batch_rows < MMQ_DP4A_MAX_BATCH_SIZE;
    }
    if (turing_mma_available(cc)) {
        return true;
    }
    if (amd_mfma_available(cc)) {
        if (GGML_CUDA_CC_IS_CDNA3(cc) || batch_rows <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (batch_rows <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }
    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return batch_rows <= 128;
                case GGML_TYPE_Q6_K:
                    return batch_rows <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || batch_rows <= 128;
                default:
                    return true;
            }
        }
        return true;
    }
    return (!GGML_CUDA_CC_IS_CDNA(cc)) || batch_rows < MMQ_DP4A_MAX_BATCH_SIZE;
}

at::Tensor convert_contiguous_cuda(const void * src, ggml_type src_type, at::ScalarType dst_type, int64_t ne, const at::TensorOptions & options, cudaStream_t stream) {
    at::Tensor dst = at::empty({ne}, options.dtype(dst_type));
    if (src_type == ggml_type_from_scalar_type(dst_type)) {
        const size_t itemsize = dst.element_size();
        CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src, static_cast<size_t>(ne) * itemsize, cudaMemcpyDeviceToDevice, stream));
        return dst;
    }
    if (dst_type == at::kFloat) {
        const to_fp32_cuda_t convert = ggml_get_to_fp32_cuda(src_type);
        TORCH_CHECK(convert != nullptr, "Missing fp32 converter for GGUF CUDA source type");
        convert(src, dst.data_ptr<float>(), ne, stream);
        return dst;
    }
    if (dst_type == at::kHalf) {
        const to_fp16_cuda_t convert = ggml_get_to_fp16_cuda(src_type);
        TORCH_CHECK(convert != nullptr, "Missing fp16 converter for GGUF CUDA source type");
        convert(src, reinterpret_cast<half *>(dst.data_ptr<at::Half>()), ne, stream);
        return dst;
    }
    if (dst_type == at::kBFloat16) {
        const to_bf16_cuda_t convert = ggml_get_to_bf16_cuda(src_type);
        TORCH_CHECK(convert != nullptr, "Missing bf16 converter for GGUF CUDA source type");
        convert(src, reinterpret_cast<nv_bfloat16 *>(dst.data_ptr<at::BFloat16>()), ne, stream);
        return dst;
    }
    TORCH_CHECK(false, "Unsupported GGUF CUDA destination dtype");
}

at::Tensor cast_tensor_cuda(const at::Tensor & input, at::ScalarType dst_type, cudaStream_t stream) {
    at::Tensor src = input.is_contiguous() ? input : input.contiguous();
    if (src.scalar_type() == dst_type) {
        return src;
    }
    switch (src.scalar_type()) {
        case at::kFloat:
        case at::kHalf:
        case at::kBFloat16:
            return convert_contiguous_cuda(src.data_ptr(), ggml_type_from_scalar_type(src.scalar_type()), dst_type, src.numel(), src.options(), stream).reshape(src.sizes());
        default:
            return src.to(dst_type);
    }
}

std::string normalize_linear_mode(const std::string & linear_mode_name) {
    if (linear_mode_name == "mmq" || linear_mode_name == "legacy" || linear_mode_name == "v3" || linear_mode_name == "mmq_v3" || linear_mode_name == "v4_mmq") {
        return "mmq";
    }
    if (linear_mode_name == "cublas" || linear_mode_name == "dequant" || linear_mode_name == "v4_cublas") {
        return "cublas";
    }
    return "auto";
}

void maybe_cast_bias_inplace(at::Tensor & output, const c10::optional<at::Tensor> & bias) {
    if (!bias.has_value() || !bias->defined()) {
        return;
    }
    at::Tensor bias_tensor = *bias;
    if (bias_tensor.device() != output.device() || bias_tensor.scalar_type() != output.scalar_type()) {
        bias_tensor = bias_tensor.to(output.device(), output.scalar_type());
    }
    output.add_(bias_tensor);
}

template <typename dst_t>
__global__ void gguf_dequantize_rows_q4_k_kernel(
    const block_q4_K * __restrict__ weight,
    const int32_t * __restrict__ rows,
    float * __restrict__ output,
    const int64_t num_rows,
    const int64_t row_stride,
    const int64_t num_embeddings) {
    const int row_index = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row_index >= num_rows || col >= row_stride) {
        return;
    }

    const int32_t lookup = rows[row_index];
    if (lookup < 0 || lookup >= num_embeddings) {
        output[row_index * row_stride + col] = 0.0f;
        return;
    }

    const block_q4_K * row_ptr = weight + (static_cast<int64_t>(lookup) * row_stride) / QK_K;
    const int block_index = col / QK_K;
    const int offset = col % QK_K;
    const block_q4_K & block = row_ptr[block_index];

    const float d = __half2float(block.data.d);
    const float dmin = __half2float(block.data.dmin);

    const int scale_index = offset / 32;
    uint8_t sc;
    uint8_t m;
    if (scale_index < 4) {
        sc = block.scales[scale_index] & 63;
        m = block.scales[scale_index + 4] & 63;
    } else {
        sc = (block.scales[scale_index + 4] & 0xF) | ((block.scales[scale_index - 4] >> 6) << 4);
        m = (block.scales[scale_index + 4] >> 4) | ((block.scales[scale_index] >> 6) << 4);
    }

    const int q_pair = offset / 64;
    const int q_byte = q_pair * 32 + (offset % 32);
    const uint8_t packed = block.qs[q_byte];
    const int q = (offset & 32) ? (packed >> 4) : (packed & 0xF);
    output[row_index * row_stride + col] = d * static_cast<float>(sc) * static_cast<float>(q) - dmin * static_cast<float>(m);
}

__global__ void gguf_dequantize_rows_q6_k_kernel(
    const block_q6_K * __restrict__ weight,
    const int32_t * __restrict__ rows,
    float * __restrict__ output,
    const int64_t num_rows,
    const int64_t row_stride,
    const int64_t num_embeddings) {
    const int row_index = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row_index >= num_rows || col >= row_stride) {
        return;
    }

    const int32_t lookup = rows[row_index];
    if (lookup < 0 || lookup >= num_embeddings) {
        output[row_index * row_stride + col] = 0.0f;
        return;
    }

    const block_q6_K * row_ptr = weight + (static_cast<int64_t>(lookup) * row_stride) / QK_K;
    const int block_index = col / QK_K;
    const int offset = col % QK_K;
    const block_q6_K & block = row_ptr[block_index];

    const float d = __half2float(block.d);
    const int chunk128 = offset / 128;
    const int chunk_offset = offset % 128;
    const int group = chunk_offset / 32;
    const int l = chunk_offset % 32;
    const int is = l / 16;
    const int scale_index = chunk128 * 8 + is + 2 * group;
    const int8_t scale = block.scales[scale_index];

    const uint8_t high = block.qh[chunk128 * 32 + l];
    const int ql_base = chunk128 * 64 + ((group & 1) ? 32 : 0) + l;
    const uint8_t packed = block.ql[ql_base];
    const int shift = (group >= 2) ? 4 : 0;
    const int low = (packed >> shift) & 0xF;
    const int high2 = (high >> (2 * group)) & 0x3;
    const int q = (low | (high2 << 4)) - 32;
    output[row_index * row_stride + col] = d * static_cast<float>(scale) * static_cast<float>(q);
}

void gguf_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("Unsupported GGUF CUDA qtype in mmq switch");
    }
}

template <typename src_t, mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1_typed(
        const src_t * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t ib0 = blockIdx.z * ((int64_t) gridDim.x * gridDim.y * blockDim.x / QK8_1);
    const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * ne1 + blockIdx.x;
    const int64_t iqs = i0 % (4 * QK8_1);

    const float4 xi = i0 < ne00
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
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
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

template <typename src_t>
void quantize_mmq_q8_1_typed_cuda(
        const src_t * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % (4 * QK8_1) == 0);

    const int64_t block_num_y = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1_typed<src_t, MMQ_Q8_1_DS_LAYOUT_D4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1_typed<src_t, MMQ_Q8_1_DS_LAYOUT_DS4><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1_typed<src_t, MMQ_Q8_1_DS_LAYOUT_D2S6><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

at::Tensor run_linear_cuda(at::Tensor raw_weight, ggml_type type, std::vector<int64_t> tensor_shape, at::Tensor input_2d) {
    const int64_t out_features = tensor_shape.at(0);
    const int64_t in_features = tensor_shape.at(1);
    TORCH_CHECK(input_2d.dim() == 2, "Expected 2D input matrix");
    TORCH_CHECK(input_2d.size(1) == in_features, "Input width does not match GGUF tensor shape");

    at::Tensor input = input_2d.is_contiguous() ? input_2d : input_2d.contiguous();
    const int64_t batch_rows = input.size(0);
    at::Tensor output = at::zeros({batch_rows, out_features}, input.options().dtype(at::kFloat));
    ggml_tensor src0 = make_quantized_src0(raw_weight, type, out_features, in_features);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    ggml_backend_cuda_context & ctx = get_backend_ctx(input.device().index(), stream);

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int64_t padded_row = GGML_PAD(in_features, MATRIX_ROW_PADDING);
    const size_t q8_bytes = static_cast<size_t>(batch_rows * padded_row) * sizeof(block_q8_1) / QK8_1
        + static_cast<size_t>(get_mmq_x_max_host(cc)) * sizeof(block_q8_1_mmq);
    at::Tensor quantized_input = at::zeros({static_cast<int64_t>(q8_bytes)}, input.options().dtype(at::kByte));
    switch (input.scalar_type()) {
        case at::kFloat:
            quantize_mmq_q8_1_typed_cuda<float>(input.data_ptr<float>(), nullptr, quantized_input.data_ptr(), type, in_features, in_features, batch_rows * in_features, batch_rows * in_features, padded_row, batch_rows, 1, 1, stream);
            break;
        case at::kHalf:
            quantize_mmq_q8_1_typed_cuda<half>(reinterpret_cast<const half *>(input.data_ptr<at::Half>()), nullptr, quantized_input.data_ptr(), type, in_features, in_features, batch_rows * in_features, batch_rows * in_features, padded_row, batch_rows, 1, 1, stream);
            break;
        case at::kBFloat16:
            quantize_mmq_q8_1_typed_cuda<nv_bfloat16>(reinterpret_cast<const nv_bfloat16 *>(input.data_ptr<at::BFloat16>()), nullptr, quantized_input.data_ptr(), type, in_features, in_features, batch_rows * in_features, batch_rows * in_features, padded_row, batch_rows, 1, 1, stream);
            break;
        default:
            TORCH_CHECK(false, "Unsupported GGUF CUDA input dtype for linear: ", input.scalar_type());
    }
    CUDA_CHECK(cudaGetLastError());

    const int64_t s01 = src0.nb[1] / ggml_type_size(src0.type);
    const bool use_stream_k = false;
    const mmq_args args = {
        static_cast<const char *>(src0.data),
        src0.type,
        reinterpret_cast<const int *>(quantized_input.data_ptr()),
        nullptr,
        nullptr,
        output.data_ptr<float>(),
        in_features,
        out_features,
        batch_rows,
        s01,
        batch_rows,
        out_features,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        use_stream_k,
        batch_rows,
    };
    gguf_cuda_mul_mat_q_switch_type(ctx, args, stream);
    CUDA_CHECK(cudaGetLastError());
    return output;
}

at::Tensor run_linear_cuda_cublas(const at::Tensor & raw_weight, ggml_type type, const std::vector<int64_t> & tensor_shape, const at::Tensor & input_2d, at::ScalarType output_dtype) {
    const int64_t out_features = tensor_shape.at(0);
    const int64_t in_features = tensor_shape.at(1);
    at::Tensor input = input_2d.is_contiguous() ? input_2d : input_2d.contiguous();
    const int64_t batch_rows = input.size(0);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
    ggml_backend_cuda_context & ctx = get_backend_ctx(input.device().index(), stream);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    if (fast_fp16_hardware_available(cc)) {
        at::Tensor weight_f16 = convert_contiguous_cuda(raw_weight.data_ptr(), type, at::kHalf, out_features * in_features, raw_weight.options(), stream).reshape({out_features, in_features});
        at::Tensor input_f16 = cast_tensor_cuda(input, at::kHalf, stream);
        at::Tensor output_f16 = at::empty({batch_rows, out_features}, input.options().dtype(at::kHalf));
        const half alpha = 1.0f;
        const half beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
        CUBLAS_CHECK(cublasGemmEx(
            ctx.cublas_handle(),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_features,
            batch_rows,
            in_features,
            &alpha,
            weight_f16.data_ptr<at::Half>(),
            CUDA_R_16F,
            in_features,
            input_f16.data_ptr<at::Half>(),
            CUDA_R_16F,
            in_features,
            &beta,
            output_f16.data_ptr<at::Half>(),
            CUDA_R_16F,
            out_features,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        if (output_dtype == at::kHalf) {
            return output_f16;
        }
        if (output_dtype == at::kBFloat16 || output_dtype == at::kFloat) {
            return cast_tensor_cuda(output_f16, output_dtype, stream);
        }
        return output_f16.to(output_dtype);
    }

    at::Tensor weight_f32 = convert_contiguous_cuda(raw_weight.data_ptr(), type, at::kFloat, out_features * in_features, raw_weight.options(), stream).reshape({out_features, in_features});
    at::Tensor input_f32 = cast_tensor_cuda(input, at::kFloat, stream);
    at::Tensor output_f32 = at::empty({batch_rows, out_features}, input.options().dtype(at::kFloat));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(cublasSgemm(
        ctx.cublas_handle(),
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch_rows,
        in_features,
        &alpha,
        weight_f32.data_ptr<float>(),
        in_features,
        input_f32.data_ptr<float>(),
        in_features,
        &beta,
        output_f32.data_ptr<float>(),
        out_features));
    if (output_dtype == at::kFloat) {
        return output_f32;
    }
    return output_f32.to(output_dtype);
}

at::Tensor run_embedding_cuda(at::Tensor raw_weight, ggml_type type, std::vector<int64_t> tensor_shape, at::Tensor indices) {
    const int64_t num_embeddings = tensor_shape.at(0);
    const int64_t embedding_dim = tensor_shape.at(1);
    at::Tensor flat_indices = indices.reshape({-1});
    if (flat_indices.scalar_type() != at::kInt) {
        flat_indices = flat_indices.to(at::kInt);
    }
    if (!flat_indices.is_contiguous()) {
        flat_indices = flat_indices.contiguous();
    }

    at::Tensor output = at::empty({flat_indices.numel(), embedding_dim}, raw_weight.options().dtype(at::kFloat));
    const dim3 block(256);
    const dim3 grid(flat_indices.numel(), (embedding_dim + block.x - 1) / block.x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(raw_weight.device().index()).stream();

    if (type == GGML_TYPE_Q4_K) {
        gguf_dequantize_rows_q4_k_kernel<float><<<grid, block, 0, stream>>>(
            reinterpret_cast<const block_q4_K *>(raw_weight.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            flat_indices.numel(),
            embedding_dim,
            num_embeddings);
    } else if (type == GGML_TYPE_Q6_K) {
        gguf_dequantize_rows_q6_k_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const block_q6_K *>(raw_weight.data_ptr()),
            flat_indices.data_ptr<int32_t>(),
            output.data_ptr<float>(),
            flat_indices.numel(),
            embedding_dim,
            num_embeddings);
    } else {
        GGML_ABORT("Unsupported GGUF CUDA qtype in embedding kernel");
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

} // namespace

extern "C" GGML_API ggml_abort_callback_t ggml_set_abort_callback(ggml_abort_callback_t callback) {
    GGML_UNUSED(callback);
    return nullptr;
}

extern "C" GGML_API size_t ggml_type_size(enum ggml_type type) {
    return gguf_type_size_local(type);
}

extern "C" GGML_API int64_t ggml_blck_size(enum ggml_type type) {
    return gguf_blck_size_local(type);
}

extern "C" GGML_API size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    const size_t blck_size = ggml_blck_size(tensor->type);
    size_t nbytes = 0;
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return nbytes;
    }

    nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
    return nbytes;
}

extern "C" GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    int64_t count = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
        count *= tensor->ne[i];
    }
    return count;
}

extern "C" GGML_API size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    const int64_t blck = gguf_blck_size_local(type);
    GGML_ASSERT(ne % blck == 0);
    return gguf_type_size_local(type) * ne / blck;
}

extern "C" GGML_API bool ggml_is_contiguously_allocated(const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor) == ggml_nelements(tensor) * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
}

extern "C" GGML_API bool ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t0->nb[i] != t1->nb[i]) {
            return false;
        }
    }
    return true;
}

extern "C" GGML_API size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    return ggml_nbytes(tensor);
}

extern "C" GGML_API enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return GGML_BACKEND_BUFFER_USAGE_ANY;
}

extern "C" GGML_API void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    if (text != nullptr) {
        fputs(text, stderr);
    }
}

extern "C" GGML_API void ggml_log_internal(enum ggml_log_level level, const char * format, ...) {
    GGML_UNUSED(level);
    va_list args;
    va_start(args, format);
    std::string message = format_message(format, args);
    va_end(args);
    fputs(message.c_str(), stderr);
}

extern "C" GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    std::string message = format_message(fmt, args);
    va_end(args);
    throw std::runtime_error(std::string("ggml abort at ") + file + ":" + std::to_string(line) + ": " + message);
}

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    throw std::runtime_error(
        std::string("llama.cpp CUDA error: ") + msg + " | stmt=" + stmt + " | func=" + func + " | file=" + file + ":" + std::to_string(line));
}

void ggml_cuda_set_device(int device) {
    int current_device = -1;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int ggml_cuda_get_device() {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    std::lock_guard<std::mutex> lock(g_cuda_info_mutex);
    if (g_cuda_info != nullptr) {
        return *g_cuda_info;
    }

    auto info = std::make_unique<ggml_cuda_device_info>();
    CUDA_CHECK(cudaGetDeviceCount(&info->device_count));
    int64_t total_vram = 0;
    for (int id = 0; id < info->device_count; ++id) {
        cudaDeviceProp prop = {};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        info->default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;
        info->devices[id].cc = 100 * prop.major + 10 * prop.minor;
        info->devices[id].nsm = prop.multiProcessorCount;
        info->devices[id].smpb = prop.sharedMemPerBlock;
        info->devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info->devices[id].integrated = false;
        info->devices[id].vmm = false;
        info->devices[id].vmm_granularity = 0;
        info->devices[id].total_vram = prop.totalGlobalMem;
        info->devices[id].warp_size = prop.warpSize;
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, id));
        info->devices[id].supports_cooperative_launch = !!supports_coop_launch;
    }
    if (total_vram > 0) {
        for (int id = 0; id < info->device_count; ++id) {
            info->default_tensor_split[id] /= total_vram;
        }
    }
    g_cuda_info = std::move(info);
    return *g_cuda_info;
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; ++dev) {
        if (cublas_handles[dev] != nullptr) {
            cublasDestroy(cublas_handles[dev]);
            cublas_handles[dev] = nullptr;
        }
    }
    if (copy_event != nullptr) {
        cudaEventDestroy(copy_event);
        copy_event = nullptr;
    }
}

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int stream_no) {
    GGML_UNUSED(stream_no);
    return std::make_unique<gguf_cuda_pool_simple>(device);
}

template void mul_mat_q_case<GGML_TYPE_Q2_K>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q3_K>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_0>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_1>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q5_0>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q5_1>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q5_K>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q8_0>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ1_S>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_S>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XS>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ3_S>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ4_NL>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
template void mul_mat_q_case<GGML_TYPE_IQ4_XS>(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);

bool gguf_cuda_supports_linear_qtype_name(const std::string & qtype_name) {
    return qtype_name == "Q2_K" || qtype_name == "Q3_K" || qtype_name == "Q4_0" || qtype_name == "Q4_1" || qtype_name == "Q4_K" || qtype_name == "Q5_0" || qtype_name == "Q5_1" || qtype_name == "Q5_K" || qtype_name == "Q6_K" || qtype_name == "Q8_0" || qtype_name == "IQ1_S" || qtype_name == "IQ2_S" || qtype_name == "IQ2_XS" || qtype_name == "IQ2_XXS" || qtype_name == "IQ3_S" || qtype_name == "IQ3_XXS" || qtype_name == "IQ4_NL" || qtype_name == "IQ4_XS";
}

bool gguf_cuda_supports_embedding_qtype_name(const std::string & qtype_name) {
    return qtype_name == "Q4_K" || qtype_name == "Q6_K";
}

bool gguf_cuda_supports_qtype_name(const std::string & qtype_name) {
    return gguf_cuda_supports_linear_qtype_name(qtype_name);
}

at::Tensor gguf_cuda_linear(
    at::Tensor raw_weight,
    const std::string & qtype_name,
    std::vector<int64_t> tensor_shape,
    at::Tensor input,
    c10::optional<at::Tensor> bias,
    const std::string & output_dtype_name,
    const std::string & linear_mode_name) {
    TORCH_CHECK(tensor_shape.size() == 2, "GGUF CUDA linear expects a 2D tensor shape");
    check_cuda_tensor(raw_weight, "raw_weight");
    check_cuda_tensor(input, "input");
    TORCH_CHECK(raw_weight.device() == input.device(), "raw_weight and input must be on the same CUDA device");

    const at::cuda::CUDAGuard guard(input.device());
    const ggml_type type = ggml_type_from_qtype_name(qtype_name);
    const at::ScalarType output_dtype = scalar_type_from_name(output_dtype_name);
    at::Tensor input_2d = input.reshape({-1, input.size(-1)});
    if (!input_2d.is_contiguous()) {
        input_2d = input_2d.contiguous();
    }

    const std::string linear_mode = normalize_linear_mode(linear_mode_name);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool use_mmq = linear_mode == "mmq" || (linear_mode == "auto" && gguf_cuda_should_use_mmq_local(type, cc, input_2d.size(0)));
    at::Tensor output = use_mmq ? run_linear_cuda(raw_weight, type, tensor_shape, input_2d) : run_linear_cuda_cublas(raw_weight, type, tensor_shape, input_2d, output_dtype);
    if (output.scalar_type() != output_dtype) {
        output = output.to(output_dtype);
    }
    maybe_cast_bias_inplace(output, bias);

    std::vector<int64_t> output_shape(input.sizes().begin(), input.sizes().end() - 1);
    output_shape.push_back(tensor_shape.at(0));
    return output.reshape(output_shape);
}

at::Tensor gguf_cuda_embedding(
    at::Tensor raw_weight,
    const std::string & qtype_name,
    std::vector<int64_t> tensor_shape,
    at::Tensor indices,
    const std::string & output_dtype_name) {
    TORCH_CHECK(tensor_shape.size() == 2, "GGUF CUDA embedding expects a 2D tensor shape");
    check_cuda_tensor(raw_weight, "raw_weight");
    check_cuda_tensor(indices, "indices");

    const at::cuda::CUDAGuard guard(raw_weight.device());
    const ggml_type type = ggml_type_from_qtype_name(qtype_name);
    at::Tensor output = run_embedding_cuda(raw_weight, type, tensor_shape, indices);
    const at::ScalarType output_dtype = scalar_type_from_name(output_dtype_name);
    if (output.scalar_type() != output_dtype) {
        output = output.to(output_dtype);
    }

    std::vector<int64_t> output_shape(indices.sizes().begin(), indices.sizes().end());
    output_shape.push_back(tensor_shape.at(1));
    return output.reshape(output_shape);
}
