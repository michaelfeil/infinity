#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
// Reuses Candle's exact CUDA normalization kernel up to width 1536 and rounded
// residual semantics. Wider custom configurations fall back to ATen LayerNorm.
std::pair<at::Tensor, at::Tensor> encoder_layer_norm(const at::Tensor& input,
  const at::Tensor& weight, double epsilon, const at::Tensor& residual = at::Tensor(),
  const at::Tensor& bias = at::Tensor(), bool round_residual = true);
// Encoder-private rounded residual update. The caller owns this intermediate
// and must retire all earlier reads before the current stream reaches the norm.
// Widths supported by the exact kernel reuse its storage; wider norms keep the
// original allocating fallback. The normalized output always owns fresh storage.
std::pair<at::Tensor,at::Tensor> encoder_layer_norm_update_residual(
  const at::Tensor& input,const at::Tensor& weight,double epsilon,at::Tensor& residual);
// Reuses only caller-owned scratch within one synchronous forward. Prior reads
// must be ordered on the current model stream before the next overwrite. An
// invalid/overlapping destination uses the original allocating path instead.
std::pair<at::Tensor,at::Tensor> encoder_layer_norm_update_residual_scratch(
  const at::Tensor& input,const at::Tensor& weight,double epsilon,
  at::Tensor& residual,const at::Tensor& normalized_scratch);
// Same arithmetic without materializing an unused model-dtype residual output.
at::Tensor encoder_layer_norm_only(const at::Tensor& input,
  const at::Tensor& weight, double epsilon, const at::Tensor& residual = at::Tensor(),
  const at::Tensor& bias = at::Tensor(), bool round_residual = true);
// Half(word + token_type), then FP32 position addition and biased norm.
// Used only by the existing CUDA Half width-768 exact-biased embedding path.
at::Tensor encoder_embedding_layer_norm(const at::Tensor& word,
  const at::Tensor& token_type, const at::Tensor& position,
  const at::Tensor& weight, double epsilon, const at::Tensor& bias);
}
