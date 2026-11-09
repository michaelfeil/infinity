/// \file
/// Vision interpolation, rotary packing and reference reductions.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
/// Interpolate each bilinear corner in model dtype, retaining the caller's summation order.
at::Tensor vision_position_corners(const at::Tensor& table,
                                   int64_t side,
                                   int64_t frames,
                                   int64_t height,
                                   int64_t width,
                                   int64_t merge);
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
