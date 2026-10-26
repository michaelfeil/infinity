/// \file
/// Shared native CUDA primitives for normalization, activations, rotary positions,
/// and expert routing.

#pragma once
#include <ATen/ATen.h>
#include <utility>

namespace infinity {
// Prefix counts for FA4's packed query-tile scheduler; result is CUDA int32 [batch + 1].
at::Tensor attention_tile_prefix(const at::Tensor& cumulative, int tile, int group);

// Eight-expert CUDA router: pinned Candle descending bitonic ordering,
// including its strict-comparison tie behavior. FP32 scores, int64 indices.
std::pair<at::Tensor, at::Tensor> nomic_router_topk8(const at::Tensor& probabilities, int64_t k);

/// Normalize the final dimension after required residual and optional second-residual additions.
/// Inputs remain unchanged; the result owns fresh storage on the input device.
at::Tensor fused_add_layer_norm(const at::Tensor& x,
                                const at::Tensor& residual,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                double epsilon,
                                const at::Tensor& second_residual = at::Tensor());
// Sequential model-dtype word+type+position additions, followed by the
// same register LayerNorm reduction as fused_add_layer_norm. CUDA Half H768.
at::Tensor fused_embedding_layer_norm(const at::Tensor& ids,
                                      const at::Tensor& types,
                                      const at::Tensor& positions,
                                      const at::Tensor& words,
                                      const at::Tensor& type_table,
                                      const at::Tensor& position_table,
                                      const at::Tensor& weight,
                                      const at::Tensor& bias,
                                      double epsilon);
/// Return normalized rows and the model-dtype residual sum. Without a residual, the second result
/// aliases x; callers must not assume two new allocations.
std::pair<at::Tensor, at::Tensor> fused_add_rms_norm(const at::Tensor& x,
                                                     const at::Tensor& residual,
                                                     const at::Tensor& weight,
                                                     double epsilon);
// Activation codes: 0 SiLU, 1 GELU exact, 2 GELU tanh, 3 ReLU,
// 4 Candle GELU tanh, 5 Candle SiLU, with model-dtype intermediate rounding.
at::Tensor
fused_gated_activation(const at::Tensor& gate_up, int32_t activation, bool gate_first = true);
/// Rotate packed Q/K into separate outputs using per-token cosine/sine values. Outputs are disjoint
/// views of one fresh allocation; neither aliases the inputs. contract_first_product selects the
/// qualified floating-point contraction recipe.
std::pair<at::Tensor, at::Tensor> fused_rotary(const at::Tensor& q,
                                               const at::Tensor& k,
                                               const at::Tensor& cosine,
                                               const at::Tensor& sine,
                                               bool contract_first_product = false);
// Inference-only: Q/K must be disjoint views from a fresh QKV allocation.
// Writes each rotary pair in place and preserves the untouched V storage.
void encoder_rotary_inplace(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& cosine,
                            const at::Tensor& sine);
// Same arithmetic using the original position tensor and full RoPE tables.
void encoder_rotary_indexed_inplace(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& cosine,
                                    const at::Tensor& sine,
                                    const at::Tensor& positions);
at::Tensor fused_gelu(const at::Tensor& x, bool approximate_tanh);
/// Add broadcast feature bias before GELU, preserving the selected exact or tanh activation recipe.
at::Tensor fused_bias_gelu(const at::Tensor& x, const at::Tensor& bias, bool approximate_tanh);
/// Normalize Q/K per head and apply rotary phases into storage independent of the inputs. The
/// specialized CUDA path returns disjoint views of one shared output allocation.
std::pair<at::Tensor, at::Tensor> fused_qk_norm_rope(const at::Tensor& q,
                                                     const at::Tensor& k,
                                                     const at::Tensor& q_weight,
                                                     const at::Tensor& k_weight,
                                                     const at::Tensor& cosine,
                                                     const at::Tensor& sine,
                                                     double epsilon);
// Preserve Q/K arithmetic while writing normalized K and raw V into independent
// contiguous suffix buffers. Returns independently owned normalized Q rows.
// Optional output_rows must contain unique, in-range destination rows. The
// prefix graph owner validates and owns this immutable mapping before capture.
at::Tensor fused_qkv_norm_rope_into(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& v,
                                    const at::Tensor& qw,
                                    const at::Tensor& kw,
                                    const at::Tensor& cosine,
                                    const at::Tensor& sine,
                                    double epsilon,
                                    const at::Tensor& key_target,
                                    const at::Tensor& value_target,
                                    const at::Tensor& output_rows = {});
}
