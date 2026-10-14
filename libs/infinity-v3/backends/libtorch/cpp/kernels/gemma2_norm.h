// SPDX-License-Identifier: MIT
// Adapted Candle norm kernels retain their notices in qwen35_kernels.cu and
// backends/candle/src/kernels/gemma_rms_norm.cu.
#pragma once
#include <ATen/ATen.h>
namespace tei {
at::Tensor gemma2_norm_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor gemma2_residual_norm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
}
