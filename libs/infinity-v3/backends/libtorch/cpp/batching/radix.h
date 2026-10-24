#pragma once
// SPDX-License-Identifier: MIT
// Compact/unfold contract adapted from Candle's RadixMLP by Michael Feil.
// Copyright (c) 2025 michaelfeil
#include "batching/packed_input.h"
#include <array>

namespace tei {
bool radix_enabled(const PackedInput& input);
const at::Tensor& radix_ids(const PackedInput& input);
const at::Tensor& radix_positions(const PackedInput& input);
const at::Tensor& radix_types(const PackedInput& input);
at::Tensor radix_unfold(const at::Tensor& rows, const PackedInput& input);
// Exact-copy Q/K/V unfolding; CPU and unsupported CUDA geometry retain
// the original independent gathers. No numerical operations or padding.
std::array<at::Tensor, 3> radix_unfold_qkv(const at::Tensor& q,
                                           const at::Tensor& k,
                                           const at::Tensor& v,
                                           const PackedInput& input);
at::Tensor radix_fold(const at::Tensor& rows, const PackedInput& input);
// Validate host metadata before transfer/capture. No GPU reads or synchronization.
void validate_radix(const PackedInput& input, bool causal);
}
