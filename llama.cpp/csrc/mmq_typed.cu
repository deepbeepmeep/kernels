// BF16 output instances of llama.cpp's stream-k MMQ kernels (mul_mat_q_case<type, nv_bfloat16>): complete tiles are rounded once
// in the kernel and split tiles once in the fixup kernel, so a prefill-sized GGUF linear needs neither an FP32 product nor a
// conversion pass. Their own translation unit, so that they compile beside the FP32 instances of gguf_llamacpp_kernels.cu.
#include "gpu_compat.h"
#ifdef small
#undef small
#endif
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "_vendor/llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "_vendor/llama.cpp/ggml/src/ggml-cuda/mmq.cuh"

#if !defined(GGML_USE_HIP)
void gguf_cuda_mul_mat_q_bf16_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_PTQ1_0:  mul_mat_q_case<GGML_TYPE_PTQ1_0,  nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q2_K:    mul_mat_q_case<GGML_TYPE_Q2_K,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q3_K:    mul_mat_q_case<GGML_TYPE_Q3_K,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q4_0:    mul_mat_q_case<GGML_TYPE_Q4_0,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q4_1:    mul_mat_q_case<GGML_TYPE_Q4_1,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q4_K:    mul_mat_q_case<GGML_TYPE_Q4_K,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q5_0:    mul_mat_q_case<GGML_TYPE_Q5_0,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q5_1:    mul_mat_q_case<GGML_TYPE_Q5_1,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q5_K:    mul_mat_q_case<GGML_TYPE_Q5_K,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q6_K:    mul_mat_q_case<GGML_TYPE_Q6_K,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_Q8_0:    mul_mat_q_case<GGML_TYPE_Q8_0,    nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ1_S:   mul_mat_q_case<GGML_TYPE_IQ1_S,   nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_S:   mul_mat_q_case<GGML_TYPE_IQ2_S,   nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_XS:  mul_mat_q_case<GGML_TYPE_IQ2_XS,  nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS, nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ3_S:   mul_mat_q_case<GGML_TYPE_IQ3_S,   nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS, nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ4_NL:  mul_mat_q_case<GGML_TYPE_IQ4_NL,  nv_bfloat16>(ctx, args, stream); break;
        case GGML_TYPE_IQ4_XS:  mul_mat_q_case<GGML_TYPE_IQ4_XS,  nv_bfloat16>(ctx, args, stream); break;
        default:
            GGML_ABORT("Unsupported GGUF CUDA qtype in the BF16 mmq switch");
    }
}
#endif // !defined(GGML_USE_HIP)
