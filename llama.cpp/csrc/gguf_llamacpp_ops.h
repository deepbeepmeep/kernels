#pragma once

#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

bool gguf_cuda_supports_linear_qtype_name(const std::string & qtype_name);
bool gguf_cuda_supports_embedding_qtype_name(const std::string & qtype_name);
bool gguf_cuda_supports_qtype_name(const std::string & qtype_name);

at::Tensor gguf_cuda_linear(
    at::Tensor raw_weight,
    const std::string & qtype_name,
    std::vector<int64_t> tensor_shape,
    at::Tensor input,
    c10::optional<at::Tensor> bias,
    const std::string & output_dtype_name,
    const std::string & linear_mode_name);

at::Tensor gguf_cuda_embedding(
    at::Tensor raw_weight,
    const std::string & qtype_name,
    std::vector<int64_t> tensor_shape,
    at::Tensor indices,
    const std::string & output_dtype_name);
