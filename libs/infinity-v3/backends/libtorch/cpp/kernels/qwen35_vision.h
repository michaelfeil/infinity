/// \file
/// Vision softmax and layer-normalization interfaces using the reference reductions.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
/// Rotate interleaved vision Q/K in FP32 and copy V directly to packed attention storage.
/// Frequencies have shape [tokens, 1, head_dim / 2]; output is [3, tokens, heads, head_dim].
at::Tensor vision_rotary_qkv(const at::Tensor& qkv,
                             const at::Tensor& cosine,
                             const at::Tensor& sine,
                             at::ScalarType output_type);
// Original core warp reductions on actual tensor dimensions, without padding.
at::Tensor qwen35_vision_softmax(const at::Tensor& scores);
at::Tensor qwen35_vision_layer_norm(const at::Tensor& states,
                                    const at::Tensor& weight,
                                    const at::Tensor& bias,
                                    float epsilon);
}
