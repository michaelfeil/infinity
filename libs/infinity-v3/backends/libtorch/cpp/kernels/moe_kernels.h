// SPDX-License-Identifier: MIT
#pragma once
#include <ATen/ATen.h>
namespace tei {
// The captured routed path is qualified only when the Hopper implementation
// was compiled and the selected device is SM90.
bool native_hopper_moe_available(const c10::Device& device);
// Returns undefined for layouts/devices not supported by the native kernel.
// Weights are contiguous [E,2I,H]/[E,H,I], logits FP32, input/output BF16.
// Route counts and packed expert rows stay on-device; no padded token rows.
at::Tensor routed_moe_cuda(const at::Tensor& input, const at::Tensor& logits,
  const at::Tensor& gate_up, const at::Tensor& down, bool renormalize,
  const at::Tensor& expert_scales = {});
// Reuse the native ordered, weighted reduction for eight routed experts.
at::Tensor routed_moe_combine_cuda(const at::Tensor& expert_output,
  const at::Tensor& order, const at::Tensor& routing);
}
