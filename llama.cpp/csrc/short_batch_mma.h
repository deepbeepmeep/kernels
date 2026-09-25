#pragma once

#include <ATen/ATen.h>

#include <string>

// Q4_K / PTQ1_0 linear for 2-8 activation rows on INT8 tensor cores (compute capability 8.0+).
// Policy: "native" keeps MMVQ, "mma" forces the tensor-core path, "auto" applies
// per-shape decisions (set by the caller after measuring) and otherwise enables
// it only on compute capability 12.0, where it was validated.
bool short_batch_mma_selected(int cc, bool ptq1, int64_t batch_rows, int64_t out_features, int64_t in_features);
void short_batch_mma_linear(const at::Tensor & raw_weight, bool ptq1, int64_t out_features, int64_t in_features, const at::Tensor & input, bool silu_mul, at::Tensor & output);
void short_batch_set_mode(const std::string & mode);
std::string short_batch_mode();
void short_batch_set_decision(const std::string & qtype_name, int64_t batch_rows, int64_t out_features, int64_t in_features, bool enabled);
void short_batch_clear_decisions();
