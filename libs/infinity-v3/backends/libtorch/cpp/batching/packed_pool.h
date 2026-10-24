// SPDX-License-Identifier: Apache-2.0
/// \file
/// Packed CUDA mean-pooling entry points for exact sequence spans.
/// Implementations retain the model-dtype reduction contract described below.

#pragma once
#include <ATen/ATen.h>

namespace tei {
// CUDA pooling over selected exact sequence spans, preserving Candle's
// model-dtype reduction tree and reciprocal rounding.
/// Reduce [tokens, width] hidden rows over int32 [selected_sequences, 2] start/length spans.
at::Tensor packed_mean_pool(const at::Tensor& hidden, const at::Tensor& spans);
// Full batch mean from original int32 cumulative offsets; no metadata tensor.
/// Return one mean row per sequence from int32 [batch + 1] packed boundaries.
at::Tensor packed_mean_pool_cumulative(const at::Tensor& hidden, const at::Tensor& cumulative);
}
