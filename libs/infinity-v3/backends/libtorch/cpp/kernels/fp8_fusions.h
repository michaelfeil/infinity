/// \file
/// Fused activation/normalization and per-row FP8 quantization interfaces.

#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace tei {
// Native Candle SiLU rounding followed by dynamic per-row E4M3 quantization.
std::pair<at::Tensor, at::Tensor> fp8_swiglu_quantize(const at::Tensor& gate_up);
// BF16 residual/RMS rounding and reduction match fused_add_rms_norm.
// Returns quantized normalized rows, residual output, and FP32 row scales.
std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_add_rms_quantize(const at::Tensor& x,
                                                                    const at::Tensor& residual,
                                                                    const at::Tensor& weight,
                                                                    double epsilon);
}
