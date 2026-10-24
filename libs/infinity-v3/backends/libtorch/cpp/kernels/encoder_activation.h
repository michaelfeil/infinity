/// \file
/// Encoder GELU entry points with explicit model-dtype intermediate rounding.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
// Candle's elementwise tanh GELU, including rounding every model-dtype
// intermediate. Expert MLPs use this rather than a GEMM activation epilogue.
at::Tensor encoder_gelu(const at::Tensor& input);
// Round input+bias to model dtype before the original elementwise GELU.
at::Tensor encoder_bias_gelu(const at::Tensor& input, const at::Tensor& bias);
}
