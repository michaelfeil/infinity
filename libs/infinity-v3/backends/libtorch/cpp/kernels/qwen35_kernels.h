#pragma once
#include "batching/packed_input.h"

namespace tei {
at::Tensor gemma_vision_rotary_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor
gemma_rotary_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
at::Tensor gemma_audio_norm_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor gemma_vision_norm_cuda(const at::Tensor&, const at::Tensor&, double);
at::Tensor gemma_vision_pool_cuda(const at::Tensor&, int64_t, int64_t, int64_t);
std::pair<at::Tensor, at::Tensor> gemma3_qk_norm_rotary_cuda(const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             double);
at::Tensor gemma_norm_cuda(const at::Tensor&, const at::Tensor&, double, bool);
// Undefined result preserves the ordinary legacy norm-plus-add fallback.
std::pair<at::Tensor, at::Tensor> gemma3_residual_prenorm_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor
gemma3_legacy_residual_norm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor gemma_gated_cuda(const at::Tensor&, bool);
at::Tensor gemma_activation_cuda(const at::Tensor&, bool);
at::Tensor qwen35_delta_cuda(const at::Tensor& qkv,
                             const at::Tensor& z,
                             const at::Tensor& ab,
                             const at::Tensor& conv,
                             const at::Tensor& a_log,
                             const at::Tensor& dt_bias,
                             const at::Tensor& norm,
                             const PackedInput& input,
                             int64_t key_heads,
                             int64_t value_heads,
                             double epsilon,
                             const at::Tensor& initial = {},
                             at::Tensor* final_state = nullptr,
                             const at::Tensor& history = {},
                             const at::Tensor& history_cumulative = {},
                             bool chunk_prefill = true,
                             int64_t cache_layer = -1);
}
