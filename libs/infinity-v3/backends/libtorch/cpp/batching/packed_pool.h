// SPDX-License-Identifier: Apache-2.0
/// \file
/// Packed CUDA mean-pooling entry points for exact sequence spans.
/// Implementations retain the model-dtype reduction contract described below.

#pragma once
#include <ATen/ATen.h>

namespace tei {
// CUDA pooling over selected exact sequence spans, preserving Candle's
// model-dtype reduction tree and reciprocal rounding.
at::Tensor packed_mean_pool(const at::Tensor& hidden, const at::Tensor& spans);
// Full batch mean from original int32 cumulative offsets; no metadata tensor.
at::Tensor packed_mean_pool_cumulative(const at::Tensor& hidden, const at::Tensor& cumulative);
}
