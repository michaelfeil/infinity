// SPDX-License-Identifier: MIT
// Adapted Candle norm kernels retain their notices in qwen35_kernels.cu and
// backends/candle/src/kernels/gemma_rms_norm.cu.
/// \file
/// Native Gemma2 RMS-normalization and residual-normalization entry points.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
/// Apply the Gemma2 RMS recipe to token rows using a prepared affine scale.
at::Tensor gemma2_norm_cuda(const at::Tensor&, const at::Tensor&, double);
/// Fuse the rounded residual addition with the Gemma2 normalization recipe.
at::Tensor
gemma2_residual_norm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
}
