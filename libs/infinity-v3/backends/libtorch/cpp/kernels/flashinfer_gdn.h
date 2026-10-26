// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Fresh packed GDN adapter. Returns false when the native bundle is unavailable.
#pragma once
#include <ATen/ATen.h>
#include <array>

namespace infinity {
struct PackedInput;
/// Pack provider Q/K/V and gates in one pass, preserving original casts.
std::array<at::Tensor, 5> flashinfer_gdn_pack_cuda(const at::Tensor& values,
                                                 const at::Tensor& qk,
                                                 const at::Tensor& gates,
                                                 int64_t key_heads,
                                                 int64_t value_heads);
bool flashinfer_gdn_available(const c10::Device& device);
bool flashinfer_gdn_prefill(const at::Tensor& values,
                            const at::Tensor& qk,
                            const at::Tensor& gates,
                            const PackedInput& input,
                            at::Tensor& output,
                            int64_t key_heads,
                            int64_t value_heads,
                            at::Tensor* final_state);
}
