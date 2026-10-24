// SPDX-License-Identifier: Apache-2.0
/// \file
/// Interface for filling packed disentangled-attention bias storage.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
// Fill only real sequence-local bias tiles. Metadata is derived from validated
// packed lengths/bucket construction; no token rows are padded or transferred.
// Indices may be packed 1D with offsets, or the actual longest real sequence
// square 2D table with undefined offsets (direct top-left shared indexing).
void packed_bias_fill(const at::Tensor& out,
                      const at::Tensor& content_key,
                      const at::Tensor& position_content,
                      const at::Tensor& indices,
                      const at::Tensor& index_offsets,
                      const at::Tensor& cumulative,
                      int64_t max_sequence);
}
