#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <memory>

namespace {
void driver_check(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char * description = nullptr;
        cuGetErrorString(status, &description);
        TORCH_CHECK(false, "SM120 CUDA driver error: ", description ? description : "unknown");
    }
}

struct Kernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
    unsigned shared = 0;
    ~Kernel() {
        if (module && cuCtxPushCurrent(context) == CUDA_SUCCESS) {
            cuModuleUnload(module);
            CUcontext previous;
            cuCtxPopCurrent(&previous);
        }
    }
};

pybind11::capsule load_kernel(pybind11::bytes binary, const std::string & name, unsigned shared) {
    const auto * device = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(device->major == 12 && device->minor == 0, "Async attention binaries require SM120.");
    auto kernel = std::make_unique<Kernel>();
    driver_check(cuCtxGetCurrent(&kernel->context));
    std::string data = binary;
    driver_check(cuModuleLoadData(&kernel->module, data.data()));
    driver_check(cuModuleGetFunction(&kernel->function, kernel->module, name.c_str()));
    kernel->shared = shared;
    if (shared > 48 * 1024) driver_check(cuFuncSetAttribute(kernel->function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared));
    return pybind11::capsule(kernel.release(), "llamacpp_gguf_cuda.sm120", [](PyObject * capsule) {
        delete static_cast<Kernel *>(PyCapsule_GetPointer(capsule, "llamacpp_gguf_cuda.sm120"));
    });
}

void launch(const pybind11::capsule & handle, at::Tensor query, unsigned x, unsigned y, unsigned z, void ** arguments) {
    auto * kernel = static_cast<Kernel *>(handle.get_pointer());
    c10::cuda::CUDAGuard guard(query.device());
    auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
    driver_check(cuLaunchKernel(kernel->function, x, y, z, 128, 1, 1, kernel->shared, reinterpret_cast<CUstream>(stream.stream()), arguments, nullptr));
}

void prefill(pybind11::capsule handle, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor ks, at::Tensor vs, at::Tensor tables, at::Tensor cu_q, at::Tensor cu_k, at::Tensor output, double scaling) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == 256, "SM120 query must be CUDA [tokens, heads, 256].");
    TORCH_CHECK(k.size(1) % 32 == 0, "SM120 cache pages must be multiples of 32 tokens.");
    void * q_ptr = q.data_ptr(), * k_ptr = k.data_ptr(), * v_ptr = v.data_ptr(), * ks_ptr = ks.data_ptr(), * vs_ptr = vs.data_ptr();
    void * tables_ptr = tables.data_ptr(), * cu_q_ptr = cu_q.data_ptr(), * cu_k_ptr = cu_k.data_ptr(), * out_ptr = output.data_ptr();
    int q_stride_t = q.stride(0), q_stride_h = q.stride(1), scale_stride_block = ks.stride(0), scale_stride_token = ks.stride(1), scale_stride_head = ks.stride(2);
    int bt_stride = tables.stride(0), out_stride_t = output.stride(0), out_stride_h = output.stride(1);
    int heads = q.size(1), kv_heads = k.size(2), page = k.size(1);
    float scale = static_cast<float>(scaling);
    void * scratch = nullptr;
    void * arguments[] = {&q_ptr, &k_ptr, &v_ptr, &ks_ptr, &vs_ptr, &tables_ptr, &cu_q_ptr, &cu_k_ptr, &out_ptr,
        &q_stride_t, &q_stride_h, &scale_stride_block, &scale_stride_token, &scale_stride_head, &bt_stride, &out_stride_t, &out_stride_h,
        &scale, &heads, &kv_heads, &page, &scratch, &scratch};
    launch(handle, q, cu_q.numel() - 1, heads, (q.size(0) + 15) / 16, arguments);
}

void grouped(pybind11::capsule handle, at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor ks, at::Tensor vs, at::Tensor tables, at::Tensor lengths, at::Tensor partial, at::Tensor maximum, at::Tensor sums, int splits, double scaling) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.size(2) == 256 && q.is_contiguous(), "SM120 grouped query must be contiguous CUDA [tokens, heads, 256].");
    TORCH_CHECK(k.size(1) % 32 == 0, "SM120 cache pages must be multiples of 32 tokens.");
    void * q_ptr = q.data_ptr(), * k_ptr = k.data_ptr(), * v_ptr = v.data_ptr(), * ks_ptr = ks.data_ptr(), * vs_ptr = vs.data_ptr();
    void * tables_ptr = tables.data_ptr(), * lengths_ptr = lengths.data_ptr(), * partial_ptr = partial.data_ptr(), * maximum_ptr = maximum.data_ptr(), * sums_ptr = sums.data_ptr();
    int queries = q.size(0) / lengths.numel(), heads = q.size(1), kv_heads = k.size(2), page = k.size(1), table_width = tables.size(1);
    float scale = static_cast<float>(scaling);
    void * scratch = nullptr;
    void * arguments[] = {&q_ptr, &k_ptr, &v_ptr, &ks_ptr, &vs_ptr, &tables_ptr, &lengths_ptr, &partial_ptr, &maximum_ptr, &sums_ptr,
        &queries, &heads, &kv_heads, &page, &table_width, &splits, &scale, &scratch, &scratch};
    launch(handle, q, lengths.numel(), kv_heads, ((queries * (heads / kv_heads) + 15) / 16) * splits, arguments);
}
} // namespace

void register_sm120_bindings(pybind11::module_ & module) {
    module.def("load_sm120_kernel", &load_kernel);
    module.def("sm120_prefill", &prefill);
    module.def("sm120_grouped", &grouped);
}
