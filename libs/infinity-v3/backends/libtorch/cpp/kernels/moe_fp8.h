// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Persistent routed-expert FP8 execution for Gemma4 MoE.
#pragma once

#include <ATen/ATen.h>
#include <cstddef>
#include <cstdint>

namespace infinity {
/// Owns the one-time E4M3 expert-weight preparation for a routed Gemma MoE
/// layer. Forward calls reuse the cached matrices/scales and keep every routing
/// count, offset, row map and packed row on the device.
class GemmaMoeFp8 {
public:
  GemmaMoeFp8() = default;
  /// Quantize contiguous BF16 [E, 2I, H] gate/up and [E, H, I] down weights
  /// once. The original learned per-output-row scales are retained.
  GemmaMoeFp8(const at::Tensor& gate_up, const at::Tensor& down);

  bool defined() const;

  int64_t experts() const {
    return experts_;
  }

  int64_t hidden() const {
    return hidden_;
  }

  int64_t intermediate() const {
    return intermediate_;
  }

  /// Whether this build/device can execute the native SM100 FP8 grouped path.
  static bool supported(const c10::Device& device);

  /// Device workspace required for this many tokens, including routing,
  /// route-packed E4M3 input rows with FP32 row scales, FP32 down output,
  /// and provider scratch.
  size_t workspace_bytes(int64_t tokens) const;

  /// Execute routing, quantize the original input rows once, pack the E4M3
  /// bytes and FP32 scales by route map, then run grouped FP8 gate/up, the
  /// exact BF16 GeGLU, grouped FP8 down with FP32 output, and the existing
  /// weighted combine. Allocates the device-only scratch for the call.
  /// Requires a prepared provider and valid operands; unsupported calls throw.
  at::Tensor
  forward(const at::Tensor& input, const at::Tensor& logits, const at::Tensor& expert_scales) const;

  /// Same execution with caller-owned scratch; useful for graph capture and
  /// reuse of an existing MoE workspace.
  at::Tensor forward(const at::Tensor& input,
                     const at::Tensor& logits,
                     const at::Tensor& expert_scales,
                     void* scratch,
                     size_t scratch_bytes) const;

private:
  at::Tensor gate_up_;
  at::Tensor gate_up_scales_;
  at::Tensor down_;
  at::Tensor down_scales_;
  int64_t experts_ = 0;
  int64_t hidden_ = 0;
  int64_t intermediate_ = 0;
};
} // namespace infinity
