// Copyright Infinity Core Contributors. See the repository LICENSE.
/// \file
/// Transformers-style RMS-normalization and residual-update interface.

#pragma once
#include <ATen/ATen.h>
#include <utility>

namespace tei {
// Transformers RMSNorm rounds the normalized activation to model dtype before
// multiplying by its affine weight. Residual additions also round before the
// FP32 statistics, unlike the unrounded-residual Candle path.
/// Return normalized rows and rounded residual sum; without
/// residual, the second result aliases input.
std::pair<at::Tensor, at::Tensor> transformers_rms_norm(const at::Tensor& input,
                                                        const at::Tensor& residual,
                                                        const at::Tensor& weight,
                                                        double epsilon);
}
