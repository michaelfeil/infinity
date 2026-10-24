#pragma once
// SPDX-License-Identifier: Apache-2.0
// Original exact-copy CUDA operator; no third-party kernel source copied.
#include <ATen/ATen.h>
#include <array>

namespace tei {
// Metadata-only check: no device reads, synchronization or token-row padding.
bool radix_qkv_cuda_supported(const at::Tensor& q,
                              const at::Tensor& k,
                              const at::Tensor& v,
                              const at::Tensor& scatter);
/// Gather compact Q/K/V into three owned logical-row tensors using one int64 scatter map. The
/// caller must validate map contents before this CUDA launch; only layout is checked here.
std::array<at::Tensor, 3> radix_unfold_qkv_cuda(const at::Tensor& q,
                                                const at::Tensor& k,
                                                const at::Tensor& v,
                                                const at::Tensor& scatter);
}
