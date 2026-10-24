// SPDX-License-Identifier: Apache-2.0
/// \file
/// Eligibility and dispatch interface for batched prefix-state tensor copies.

#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace tei {
// Batched, bit-preserving copies from row-strided state into contiguous owners.
// Returns false without submitting work if any layout is unsupported.
// leading_rows copies the target's row count from a possibly longer source.
bool prefix_state_copies(const std::vector<at::Tensor>& targets,
                         const std::vector<at::Tensor>& sources,
                         bool leading_rows = false);
}
