// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace infinity {
// Supported layouts use FlashAttention's LSE output and FlashInfer's cascade merge.
bool block_cascade_supported(const at::Tensor& q);
at::Tensor block_cascade_attention(const at::Tensor& q,
                                   const at::Tensor& k,
                                   const at::Tensor& v,
                                   const at::Tensor& prefix_k,
                                   const at::Tensor& prefix_v,
                                   const at::Tensor& cumulative,
                                   const int32_t* offsets,
                                   const std::vector<int32_t>& prefix_offsets,
                                   const std::vector<int64_t>& sources,
                                   int64_t max_suffix,
                                   double scale);
}
