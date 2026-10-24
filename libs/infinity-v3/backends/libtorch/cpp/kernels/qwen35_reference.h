/// \file
/// Reference-arithmetic Qwen3.5 gated-delta and normalization interfaces.

#pragma once
#include "batching/packed_input.h"

namespace tei {
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
at::Tensor qwen35_reference_norm_cuda(const at::Tensor&, const at::Tensor&, double);
}
