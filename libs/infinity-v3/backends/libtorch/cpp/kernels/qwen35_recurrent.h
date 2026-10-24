/// \file
/// Qwen3.5 recurrent and cached-convolution execution interfaces.

#pragma once
#include "batching/packed_input.h"

namespace tei {
/// Check the device architecture and head geometry qualified for the register-state recurrent path.
bool qwen35_recurrent_supported(c10::Device, int64_t, int64_t);
/// Write BF16 [tokens, value_heads, 128] output from packed convolution values and F32 Q/K/gates.
/// Sequence boundaries reset state; the final boolean chooses the reference contraction recipe.
void qwen35_recurrent_cuda(const at::Tensor&,
                           const at::Tensor&,
                           const at::Tensor&,
                           const PackedInput&,
                           at::Tensor&,
                           int64_t,
                           int64_t,
                           bool);
/// Apply sequence-local convolution using retained history and per-sequence history lengths. Write
/// caller-owned output; history supplies only the tokens preceding each real suffix.
void qwen35_cached_conv_cuda(const at::Tensor&,
                             const at::Tensor&,
                             const PackedInput&,
                             const at::Tensor&,
                             const at::Tensor&,
                             at::Tensor&,
                             bool);
/// Resume recurrence from retained state pointers and write requested intermediate checkpoints.
/// Pointer owners must remain alive until the current CUDA stream finishes consuming them.
void qwen35_cached_recurrent_cuda(const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const PackedInput&,
                                  at::Tensor&,
                                  int64_t,
                                  int64_t,
                                  bool,
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  at::Tensor&);
}
