// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Fresh packed GDN adapter. Returns false when the native bundle is unavailable.
#pragma once
#include <ATen/ATen.h>

namespace infinity {
struct PackedInput;
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
