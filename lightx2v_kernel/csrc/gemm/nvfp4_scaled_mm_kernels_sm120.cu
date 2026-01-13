#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

#define CHECK_TYPE(x, st, m) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type:", m)
#define CHECK_TH_CUDA(x, m) TORCH_CHECK(x.is_cuda(), m, "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x, m) TORCH_CHECK(x.is_contiguous(), m, "must be contiguous")
#define CHECK_INPUT(x, st, m) \
  CHECK_TH_CUDA(x, m);        \
  CHECK_CONTIGUOUS(x, m);     \
  CHECK_TYPE(x, st, m)

namespace {

#define CUBLAS_CHECK(call)                                                            \
  do {                                                                                \
    cublasStatus_t status = call;                                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                                            \
      throw std::runtime_error(std::string("cuBLAS error: ") + std::to_string(status)); \
    }                                                                                 \
  } while (0)

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err = call;                                                           \
    if (err != cudaSuccess) {                                                         \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    }                                                                                 \
  } while (0)

__device__ __constant__ float one_device;
__device__ __constant__ float zero_device;

float* get_scalar_one() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float one = 1.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(one_device, &one, sizeof(float)));
  });
  float* dev_ptr = nullptr;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_ptr), one_device));
  return dev_ptr;
}

float* get_scalar_zero() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    float zero = 0.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(zero_device, &zero, sizeof(float)));
  });
  float* dev_ptr = nullptr;
  CUDA_CHECK(cudaGetSymbolAddress(reinterpret_cast<void**>(&dev_ptr), zero_device));
  return dev_ptr;
}

thread_local cublasLtHandle_t cached_handle = nullptr;

cublasLtHandle_t get_cublas_lt_handle() {
  if (cached_handle == nullptr) {
    cublasStatus_t status = cublasLtCreate(&cached_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("cuBLAS handle creation error: ") + std::to_string(status));
    }
  }
  return cached_handle;
}

cudaDataType_t to_cuda_type(at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return CUDA_R_32F;
    case at::ScalarType::Half:
      return CUDA_R_16F;
    case at::ScalarType::BFloat16:
      return CUDA_R_16BF;
    default:
      throw std::runtime_error("Unsupported output dtype for NVFP4 GEMM");
  }
}

void cublas_gemm_blockwise_fp4_impl(
    const void* A_ptr,
    const void* A_decode_scale_ptr,
    const void* B_ptr,
    const void* B_decode_scale_ptr,
    void* D_ptr,
    const void* bias_ptr,
    int64_t A_rows,
    int64_t A_cols,
    int64_t B_rows,
    int64_t B_cols,
    int64_t D_rows,
    int64_t D_cols,
    int64_t bias_size,
    cudaDataType_t Dtype,
    const float* alpha_ptr,
    cudaStream_t stream) {
  // Only TN layout is supported.
  const bool transa = true;
  const bool transb = false;

  if (A_rows == 0 || A_cols == 0 || B_rows == 0 || B_cols == 0) {
    return;
  }

  if (D_rows != B_rows || D_cols != A_rows) {
    throw std::runtime_error("D shape mismatch");
  }

  const int m = transa ? static_cast<int>(A_rows) : static_cast<int>(A_cols);
  const int k = (transa ? static_cast<int>(A_cols) : static_cast<int>(A_rows)) * 2;
  const int n = transb ? static_cast<int>(B_cols) : static_cast<int>(B_rows);

  int lda = k;
  int ldb = k;
  int ldc = m;
  int ldd = m;

  float* beta_ptr = get_scalar_zero();

  cublasLtHandle_t ltHandle = get_cublas_lt_handle();
  int returnedResults = 0;
  cublasLtMatmulHeuristicResult_t heuristicResult = {};

  cublasLtMatmulDesc_t operationDesc = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasLtMatmulMatrixScale_t A_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  cublasLtMatmulMatrixScale_t B_scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &A_scale_mode, sizeof(A_scale_mode)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &B_scale_mode, sizeof(B_scale_mode)));

  const cublasOperation_t transa_type = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transb_type = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa_type, sizeof(transa_type)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb_type, sizeof(transb_type)));

  const cudaDataType_t Atype = CUDA_R_4F_E2M1;
  const cudaDataType_t Btype = CUDA_R_4F_E2M1;

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &A_decode_scale_ptr,
      sizeof(A_decode_scale_ptr)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &B_decode_scale_ptr,
      sizeof(B_decode_scale_ptr)));

  cublasDataType_t scale_type = CUDA_R_32F;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

  cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

  cublasLtMatrixLayout_t Adesc = nullptr;
  cublasLtMatrixLayout_t Bdesc = nullptr;
  cublasLtMatrixLayout_t Cdesc = nullptr;
  cublasLtMatrixLayout_t Ddesc = nullptr;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Adesc, Atype, transa_type == CUBLAS_OP_N ? m : k, transa_type == CUBLAS_OP_N ? k : m, lda));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
      &Bdesc, Btype, transb_type == CUBLAS_OP_N ? k : n, transb_type == CUBLAS_OP_N ? n : k, ldb));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, Dtype, m, n, ldc));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, Dtype, m, n, ldd));

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  if (bias_ptr != nullptr && bias_size != 0) {
    if (bias_size != m) {
      throw std::runtime_error("bias must have size matching output columns");
    }
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));

  size_t workspace_size = 32 * 1024 * 1024;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

  const auto status = cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      operationDesc,
      Adesc,
      Bdesc,
      Cdesc,
      Ddesc,
      preference,
      1,
      &heuristicResult,
      &returnedResults);

  if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
    throw std::runtime_error("Unable to find suitable cuBLAS GEMM algorithm");
  }

  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  auto workspace_options =
      at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, device_id));
  auto workspace = at::empty(static_cast<int64_t>(workspace_size), workspace_options);

  CUBLAS_CHECK(cublasLtMatmul(
      ltHandle,
      operationDesc,
      alpha_ptr,
      A_ptr,
      Adesc,
      B_ptr,
      Bdesc,
      beta_ptr,
      D_ptr,
      Cdesc,
      D_ptr,
      Ddesc,
      &heuristicResult.algo,
      workspace.data_ptr(),
      workspace_size,
      stream));

  if (preference) {
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
  }
  if (Ddesc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
  }
  if (Cdesc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
  }
  if (Bdesc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
  }
  if (Adesc) {
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
  }
  if (operationDesc) {
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(operationDesc));
  }
}

void run_cublas_nvfp4(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<at::Tensor> const& bias,
    cudaStream_t stream) {
  const int64_t M = A.size(0);
  const int64_t K = A.size(1) * 2;
  const int64_t N = B.size(0);

  TORCH_CHECK(B.size(1) * 2 == K, "K mismatch between A and B");
  TORCH_CHECK(D.size(0) == M && D.size(1) == N, "Output shape mismatch");
  TORCH_CHECK(alpha.numel() == 1, "alpha must be a scalar tensor");

  const void* bias_ptr = nullptr;
  int64_t bias_size = 0;
  if (bias) {
    bias_ptr = bias->data_ptr();
    bias_size = bias->numel();
  }

  const void* A_ptr = B.data_ptr();
  const void* A_scale_ptr = B_sf.data_ptr();
  const void* B_ptr = A.data_ptr();
  const void* B_scale_ptr = A_sf.data_ptr();

  cublas_gemm_blockwise_fp4_impl(
      A_ptr,
      A_scale_ptr,
      B_ptr,
      B_scale_ptr,
      D.data_ptr(),
      bias_ptr,
      N,
      K / 2,
      M,
      K / 2,
      M,
      N,
      bias_ptr ? N : 0,
      to_cuda_type(D.scalar_type()),
      static_cast<const float*>(alpha.data_ptr()),
      stream);
}

} // namespace


using namespace cute;


struct Fp4GemmSm120 {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // A matrix configuration
    using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
    using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
    using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // C/D matrix configuration
    using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
    using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
    using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
    using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
    // Kernel functional config
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

    // Kernel Perf config
    using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
    using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

    // use per-column bias, i.e. every column has different bias
    using EVTOp = cutlass::epilogue::fusion::LinCombPerColBias<ElementD, ElementAccumulator>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,                      // Epilogue schedule policy
        EVTOp
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,                                                   // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));
};


// Populates a Gemm::Arguments structure from the given commandline options
typename Fp4GemmSm120::Gemm::Arguments args_from_options_nvfp4_nvfp4(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<at::Tensor> const& bias,
    int64_t M,
    int64_t N,
    int64_t K) {
  using Sm1xxBlkScaledConfig = typename Fp4GemmSm120::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  int m = static_cast<int>(M);
  int n = static_cast<int>(N);
  int k = static_cast<int>(K);
  auto stride_A = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideA{}, {m, k, 1});
  auto stride_B = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideB{}, {n, k, 1});
  auto stride_D = cutlass::make_cute_packed_stride(Fp4GemmSm120::StrideD{}, {m, n, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

  if (bias){
    using StrideBias = Stride<cutlass::_0, cutlass::_1, int64_t>;

    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    fusion_args.bias_ptr = static_cast<Fp4GemmSm120::Gemm::ElementC const*>(bias->data_ptr());
    fusion_args.dBias = StrideBias{};
    return arguments;
  } else {
    typename Fp4GemmSm120::Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {// Mainloop arguments
       static_cast<Fp4GemmSm120::Gemm::ElementA const*>(A.data_ptr()),
       stride_A,
       static_cast<Fp4GemmSm120::Gemm::ElementB const*>(B.data_ptr()),
       stride_B,
       static_cast<cutlass::float_ue4m3_t const*>(A_sf.data_ptr()),
       layout_SFA,
       static_cast<cutlass::float_ue4m3_t const*>(B_sf.data_ptr()),
       layout_SFB},
      {     // Epilogue arguments
       {},  // epilogue.thread
       static_cast<Fp4GemmSm120::Gemm::ElementC const*>(D.data_ptr()),
       stride_D,
       static_cast<Fp4GemmSm120::Gemm::ElementD*>(D.data_ptr()),
       stride_D}};
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr = static_cast<float const*>(alpha.data_ptr());
    static const float beta_zero = 0.0f;
    fusion_args.beta_ptr = &beta_zero;
    return arguments;
  }
}


void runGemmNvfp4Sm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<at::Tensor> const& bias,
    int64_t m,
    int64_t n,
    int64_t k,
    cudaStream_t stream) {
  typename Fp4GemmSm120::Gemm gemm;

  auto arguments = args_from_options_nvfp4_nvfp4(D, A, B, A_sf, B_sf, alpha, bias, m, n, k);
  size_t workspace_size = Fp4GemmSm120::Gemm::get_workspace_size(arguments);
  auto const workspace_options = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto workspace = at::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}


constexpr auto FLOAT4_E2M1X2 = at::ScalarType::Byte;
constexpr auto SF_DTYPE = at::ScalarType::Float8_e4m3fn;

void cutlass_scaled_nvfp4_mm_sm120(
    at::Tensor& D,
    at::Tensor const& A,
    at::Tensor const& B,
    at::Tensor const& A_sf,
    at::Tensor const& B_sf,
    at::Tensor const& alpha,
    c10::optional<at::Tensor> const& bias) {

  CHECK_INPUT(A, FLOAT4_E2M1X2, "a");
  CHECK_INPUT(B, FLOAT4_E2M1X2, "b");

  CHECK_INPUT(A_sf, SF_DTYPE, "scale_a");
  CHECK_INPUT(B_sf, SF_DTYPE, "scale_b");
  CHECK_INPUT(alpha, at::ScalarType::Float, "alpha");


  TORCH_CHECK(A.dim() == 2, "a must be a matrix");
  TORCH_CHECK(B.dim() == 2, "b must be a matrix");
  TORCH_CHECK(
      A.sizes()[1] == B.sizes()[1],
      "a and b shapes cannot be multiplied (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      " and ",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ")");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;

  constexpr int alignment = 32;
  TORCH_CHECK(
      k % alignment == 0,
      "Expected k to be divisible by ",
      alignment,
      ", but got a shape: (",
      A.sizes()[0],
      "x",
      A.sizes()[1],
      "), k: ",
      k,
      ".");
  TORCH_CHECK(
      n % alignment == 0,
      "Expected n to be divisible by ",
      alignment,
      ", but got b shape: (",
      B.sizes()[0],
      "x",
      B.sizes()[1],
      ").");

  auto round_up = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up(m, 128);
  int rounded_n = round_up(n, 128);
  // Since k is divisible by 32 (alignment), k / 16 is guaranteed to be an
  // integer.
  int rounded_k = round_up(k / 16, 4);

  TORCH_CHECK(A_sf.dim() == 2, "scale_a must be a matrix");
  TORCH_CHECK(B_sf.dim() == 2, "scale_b must be a matrix");
  TORCH_CHECK(
      A_sf.sizes()[1] == B_sf.sizes()[1],
      "scale_a and scale_b shapes cannot be multiplied (",
      A_sf.sizes()[0],
      "x",
      A_sf.sizes()[1],
      " and ",
      B_sf.sizes()[0],
      "x",
      B_sf.sizes()[1],
      ")");
  TORCH_CHECK(
      A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
      "scale_a must be padded and swizzled to a shape (",
      rounded_m,
      "x",
      rounded_k,
      "), but got a shape (",
      A_sf.sizes()[0],
      "x",
      A_sf.sizes()[1],
      ")");
  TORCH_CHECK(
      B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
      "scale_b must be padded and swizzled to a shape (",
      rounded_n,
      "x",
      rounded_k,
      "), but got a shape (",
      B_sf.sizes()[0],
      "x",
      B_sf.sizes()[1],
      ")");

  auto out_dtype = D.dtype();
  at::cuda::CUDAGuard device_guard{(char)A.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  bool use_cublas = true;
  if (const char* env = std::getenv("LIGHTX2V_NVFP4_GEMM")) {
    std::string mode(env);
    for (auto& c : mode) c = static_cast<char>(tolower(c));
    if (mode == "cutlass") {
      use_cublas = false;
    }
  }
  if (use_cublas) {
    run_cublas_nvfp4(D, A, B, A_sf, B_sf, alpha, bias, stream);
  } else {
    runGemmNvfp4Sm120(D, A, B, A_sf, B_sf, alpha, bias, m, n, k, stream);
  }
}
