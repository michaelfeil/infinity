// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Persistent routed-expert MXFP8 execution for Gemma4 MoE.
#pragma once

#include <ATen/ATen.h>
#include <cstddef>
#include <cstdint>

namespace infinity {
/// Owns the one-time MXFP8 expert-weight preparation for a routed Gemma MoE
/// layer. Forward calls reuse cached E4M3 values and UE8M0 scale bytes while
/// all routing counts, padded offsets, row maps and packed rows stay on device.
class GemmaMoeMxfp8 {
public:
  GemmaMoeMxfp8() = default;
  /// Quantize contiguous BF16 [E,2I,H] gate/up and [E,H,I] down weights once.
  /// N=2I and N=H must be multiples of 128 so global [E*N,K] SF layouts have
  /// contiguous per-expert slices.
  GemmaMoeMxfp8(const at::Tensor& gate_up, const at::Tensor& down);

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

  /// Whether this build/device can execute the native SM100 MXFP8 grouped path.
  static bool supported(const c10::Device& device);

  /// Device workspace for routing, route-packed E4M3 rows with swizzled
  /// UE8M0 block scales, FP32 down output, and provider scratch. Capacity is
  /// slots + 128*experts padded rows.
  size_t workspace_bytes(int64_t tokens) const;

  /// Execute device-only routing, quantize the original input rows once, pack
  /// the E4M3 bytes and swizzled UE8M0 scale bytes by route map, then run
  /// MXFP8 gate/up, the original BF16 GeGLU, MXFP8 down with FP32 output, and
  /// the existing weighted combine.
  at::Tensor
  forward(const at::Tensor& input, const at::Tensor& logits, const at::Tensor& expert_scales) const;

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
