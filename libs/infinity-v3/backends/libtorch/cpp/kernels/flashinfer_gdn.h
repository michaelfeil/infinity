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
/// Run the provider on buffers already produced by either packing route.
bool flashinfer_gdn_prefill_packed(const std::array<at::Tensor, 5>& packed,
                                   const PackedInput& input,
                                   at::Tensor& output,
                                   int64_t key_heads,
                                   int64_t value_heads,
                                   at::Tensor* final_state);
bool flashinfer_gdn_prefill(const at::Tensor& values,
                            const at::Tensor& qk,
                            const at::Tensor& gates,
                            const PackedInput& input,
                            at::Tensor& output,
                            int64_t key_heads,
                            int64_t value_heads,
                            at::Tensor* final_state);
/// Resume packed suffixes from immutable FP32 prefix states.
bool flashinfer_gdn_resume(const at::Tensor& values,
                           const at::Tensor& qk,
                           const at::Tensor& gates,
                           const PackedInput& input,
                           at::Tensor& output,
                           int64_t kh,
                           int64_t vh,
                           at::Tensor* final_state,
                           const at::Tensor& initial);
/// Resume from already packed provider buffers and immutable FP32 prefix states.
bool flashinfer_gdn_resume_packed(const std::array<at::Tensor, 5>& packed,
                                  const PackedInput& input,
                                  at::Tensor& output,
                                  int64_t kh,
                                  int64_t vh,
                                  at::Tensor* final_state,
                                  const at::Tensor& initial);
}
