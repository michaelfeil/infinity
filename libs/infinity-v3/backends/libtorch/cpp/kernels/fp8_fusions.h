/// \file
/// Fused activation/normalization and per-row FP8 quantization interfaces.

#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace infinity {
inline constexpr int64_t fp8_scaled_swiglu_max_width = 20480;
// Apply separate FP32 row and column scales to FP32 GEMM values before one model-dtype rounding.
at::Tensor fp8_scale_output(const at::Tensor& values,
                            const at::Tensor& row_scales,
                            const at::Tensor& column_scales,
                            at::ScalarType dtype);
// Native Candle SiLU rounding followed by dynamic per-row E4M3 quantization.
std::pair<at::Tensor, at::Tensor> fp8_swiglu_quantize(const at::Tensor& gate_up);
// Apply FP32 row and column scales, round once to dtype, then run the same rounded SwiGLU and
// dynamic per-row E4M3 quantization. Aligned rows up to 20480 activation columns avoid the scaled
// gate/up tensor; wider or unaligned rows retain the same composed arithmetic.
std::pair<at::Tensor, at::Tensor> fp8_scaled_swiglu_quantize(const at::Tensor& values,
                                                             const at::Tensor& row_scales,
                                                             const at::Tensor& column_scales,
                                                             at::ScalarType dtype);
// BF16 residual/RMS rounding and reduction match fused_add_rms_norm.
// Returns quantized normalized rows, residual output, and FP32 row scales.
std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_add_rms_quantize(const at::Tensor& x,
                                                                    const at::Tensor& residual,
                                                                    const at::Tensor& weight,
                                                                    double epsilon);
}
