#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
// Eight-expert CUDA router: pinned Candle descending bitonic ordering,
// including its strict-comparison tie behavior. FP32 scores, int64 indices.
std::pair<at::Tensor, at::Tensor> nomic_router_topk8(const at::Tensor& probabilities, int64_t k);

at::Tensor fused_add_layer_norm(const at::Tensor& x, const at::Tensor& residual,
 const at::Tensor& weight, const at::Tensor& bias, double epsilon,
 const at::Tensor& second_residual = at::Tensor());
// Sequential model-dtype word+type+position additions, followed by the
// same register LayerNorm reduction as fused_add_layer_norm. CUDA Half H768.
at::Tensor fused_embedding_layer_norm(const at::Tensor& ids, const at::Tensor& types,
 const at::Tensor& positions, const at::Tensor& words, const at::Tensor& type_table,
 const at::Tensor& position_table, const at::Tensor& weight, const at::Tensor& bias, double epsilon);
std::pair<at::Tensor, at::Tensor> fused_add_rms_norm(const at::Tensor& x, const at::Tensor& residual,
 const at::Tensor& weight, double epsilon);
// Activation codes: 0 SiLU, 1 GELU exact, 2 GELU tanh, 3 ReLU,
// 4 Candle GELU tanh, 5 Candle SiLU, with model-dtype intermediate rounding.
at::Tensor fused_gated_activation(const at::Tensor& gate_up, int32_t activation, bool gate_first = true);
std::pair<at::Tensor, at::Tensor> fused_rotary(const at::Tensor& q, const at::Tensor& k,
 const at::Tensor& cosine, const at::Tensor& sine, bool contract_first_product = false);
// Inference-only: Q/K must be disjoint views from a fresh QKV allocation.
// Writes each rotary pair in place and preserves the untouched V storage.
void encoder_rotary_inplace(const at::Tensor& q, const at::Tensor& k,
 const at::Tensor& cosine, const at::Tensor& sine);
// Same arithmetic using the original position tensor and full RoPE tables.
void encoder_rotary_indexed_inplace(const at::Tensor& q,const at::Tensor& k,
 const at::Tensor& cosine,const at::Tensor& sine,const at::Tensor& positions);
at::Tensor fused_gelu(const at::Tensor& x, bool approximate_tanh);
at::Tensor fused_bias_gelu(const at::Tensor& x, const at::Tensor& bias, bool approximate_tanh);
std::pair<at::Tensor, at::Tensor> fused_qk_norm_rope(const at::Tensor& q,
 const at::Tensor& k, const at::Tensor& q_weight, const at::Tensor& k_weight,
 const at::Tensor& cosine, const at::Tensor& sine, double epsilon);
}
