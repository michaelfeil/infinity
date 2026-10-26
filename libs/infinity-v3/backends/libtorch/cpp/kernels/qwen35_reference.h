/// \file
/// Reference-arithmetic Qwen3.5 gated-delta and normalization interfaces.

#pragma once
#include "batching/packed_input.h"
#include <utility>

namespace infinity {
/// Evaluate convolution, recurrent update, and gated normalization with the reference operation
/// order. cache_layer connects suffix execution to resident
/// recurrent/history snapshots when present.
at::Tensor qwen35_reference_delta_cuda(const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       const PackedInput&,
                                       int64_t,
                                       int64_t,
                                       double,
                                       int64_t cache_layer = -1);
/// Apply per-head RMS normalization while preserving reference intermediate rounding.
at::Tensor qwen35_reference_norm_cuda(const at::Tensor&, const at::Tensor&, double);
/// Preserve BF16 residual addition and the original norm reduction in one launch.
std::pair<at::Tensor, at::Tensor> qwen35_reference_residual_norm_cuda(const at::Tensor&,
                                                                      const at::Tensor&,
                                                                      const at::Tensor&,
                                                                      double);
}
