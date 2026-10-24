// Fork contribution; see LICENSE and LICENSING.md at the repository root.
/// \file
/// Laya action-feature interface over packed logits and question offsets.

#pragma once
#include <ATen/ATen.h>

namespace tei {
// Packed F32 logits and Int64 option offsets -> F32 [questions, 4] action features.
// Features retain the published top1, margin, normalized entropy and count formula.
at::Tensor laya_action_features(const at::Tensor& logits, const at::Tensor& offsets);
}
