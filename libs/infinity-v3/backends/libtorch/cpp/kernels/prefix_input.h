// SPDX-License-Identifier: Apache-2.0
/// \file
/// Suffix input buffers and packing interface for retained-prefix execution.

#pragma once
#include <ATen/ATen.h>
#include <span>

namespace infinity {
/// Owned packed suffix tensors for token IDs, positions, and token types.
struct PrefixInputs {
  at::Tensor ids, positions, types;
};

/// Check same-device contiguous CUDA int64 token vectors using metadata only; no device reads.
bool prefix_inputs_supported(const at::Tensor&, const at::Tensor&, const at::Tensor&);
// Select the uncached suffix of each sequence into one independently owned
// allocation. All metadata is validated before any GPU work is submitted.
PrefixInputs pack_prefix_inputs(const at::Tensor& ids,
                                const at::Tensor& positions,
                                const at::Tensor& types,
                                std::span<const int32_t> full,
                                std::span<const int32_t> suffix,
                                std::span<const int64_t> hits);
}
