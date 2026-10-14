// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>

namespace tei {
constexpr int64_t kNomicGroupedMaxTokens = 8192;
constexpr int64_t kNomicGroupedMaxRoutes = 2 * kNomicGroupedMaxTokens;
// Prepared at model load, without reading any request's router counts. Returns
// undefined outside the qualified model/device/library/precision profile.
at::Tensor nomic_moe_split_policy(const c10::Device& device, at::ScalarType dtype,
    int64_t hidden, int64_t width, int64_t experts, int64_t top_k);
at::Tensor nomic_grouped_experts(const at::Tensor& input, const at::Tensor& first,
    const at::Tensor& second, const at::Tensor& ids, const at::Tensor& scores,
    const at::Tensor& split_policy);
} // namespace tei
