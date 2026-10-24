#pragma once
// SPDX-License-Identifier: MIT
// Compact/unfold contract adapted from Candle's RadixMLP by Michael Feil.
// Copyright (c) 2025 michaelfeil
#include "batching/packed_input.h"
#include <array>

namespace tei {
/// Whether any compact metadata was supplied; validate_radix() checks that it is complete.
bool radix_enabled(const PackedInput& input);
/// Borrow compact token IDs when sharing is enabled, otherwise the original packed IDs.
const at::Tensor& radix_ids(const PackedInput& input);
/// Borrow the position IDs corresponding to the selected compact or full token rows.
const at::Tensor& radix_positions(const PackedInput& input);
/// Borrow the token types corresponding to the selected compact or full token rows.
const at::Tensor& radix_types(const PackedInput& input);
/// Expand compact rows into logical token order using
/// scatter_unfold; disabled input passes through.
at::Tensor radix_unfold(const at::Tensor& rows, const PackedInput& input);
// Exact-copy Q/K/V unfolding; CPU and unsupported CUDA geometry retain
// the original independent gathers. No numerical operations or padding.
std::array<at::Tensor, 3> radix_unfold_qkv(const at::Tensor& q,
                                           const at::Tensor& k,
                                           const at::Tensor& v,
                                           const PackedInput& input);
/// Select compact representatives using fold_gather; disabled input passes through.
at::Tensor radix_fold(const at::Tensor& rows, const PackedInput& input);
// Validate host metadata before transfer/capture. No GPU reads or synchronization.
void validate_radix(const PackedInput& input, bool causal);
}
