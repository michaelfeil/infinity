#pragma once
#include <ATen/ATen.h>
#include <utility>
namespace tei {
// Exact Candle Welford reduction for BF16 widths 896 (masked1024), 1024, 2048, 3072 and 4096,
// and FP16 widths 896 (masked1024), 1024 and 2048, using each registered decoder warp layout. Statistics
// use the unrounded FP32 residual sum; the returned residual rounds to the
// input dtype. No input rows are padded.
std::pair<at::Tensor,at::Tensor> decoder_exact_rms_norm(const at::Tensor& input,
 const at::Tensor& residual,const at::Tensor& weight,double epsilon);
}

namespace tei {
// Original Candle F32 warp softmax and sequential top8 normalization for the
// eligible Half Qwen3-MoE reference path. Real rows only, fresh owned outputs.
at::Tensor decoder_half_moe_probabilities(const at::Tensor& logits);
at::Tensor decoder_half_moe_route_weights(const at::Tensor& selected,bool renormalize);
}
