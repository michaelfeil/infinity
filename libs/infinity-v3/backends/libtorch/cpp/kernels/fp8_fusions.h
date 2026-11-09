/// \file
/// Fused activation/normalization and per-row FP8 quantization interfaces.

#pragma once
#include "helpers/fp8_activation.h"
#include <ATen/ATen.h>
#include <tuple>

namespace infinity {
inline constexpr int64_t fp8_scaled_swiglu_max_width = 20480;
// Apply separate FP32 row and column scales to FP32 GEMM values before one model-dtype rounding.
at::Tensor fp8_scale_output(const at::Tensor& values,
                            const at::Tensor& row_scales,
                            const at::Tensor& column_scales,
                            at::ScalarType dtype);
// Apply one rounded gated activation followed by dynamic per-row E4M3 quantization.
// RoundedSilu matches the original Candle SiLU staging, including BF16 gate-first aligned pairs.
// RoundedGelu matches the original native gated GELU staging: aligned vector rows use the native
// half-pair HFMA alpha, while ragged or unaligned rows retain the scalar separate staging.
// gate_first selects whether the first half is the activated gate; false activates the second half.
std::pair<at::Tensor, at::Tensor>
fp8_gated_quantize(const at::Tensor& gate_up, Fp8GatedActivation kind, bool gate_first = true);
// Apply FP32 row and column scales, round once to dtype, then run the same rounded gated
// activation and dynamic per-row E4M3 quantization. Aligned rows up to 20480 activation columns
// avoid the scaled gate/up tensor; wider or unaligned rows retain the same composed arithmetic.
std::pair<at::Tensor, at::Tensor> fp8_scaled_gated_quantize(const at::Tensor& values,
                                                            const at::Tensor& row_scales,
                                                            const at::Tensor& column_scales,
                                                            at::ScalarType dtype,
                                                            Fp8GatedActivation kind,
                                                            bool gate_first = true);

// Native Candle SiLU rounding followed by dynamic per-row E4M3 quantization. This legacy wrapper
// retains the original gate-first layout and bytes.
std::pair<at::Tensor, at::Tensor> fp8_swiglu_quantize(const at::Tensor& gate_up);
// Apply FP32 row and column scales, round once to dtype, then run the same rounded SwiGLU and
// dynamic per-row E4M3 quantization. Aligned rows up to 20480 activation columns avoid the scaled
// gate/up tensor; wider or unaligned rows retain the same composed arithmetic. This legacy wrapper
// retains the original gate-first layout and bytes.
std::pair<at::Tensor, at::Tensor> fp8_scaled_swiglu_quantize(const at::Tensor& values,
                                                             const at::Tensor& row_scales,
                                                             const at::Tensor& column_scales,
                                                             at::ScalarType dtype);

namespace detail {
// Internal entry points shared by the gated wrappers. Aligned eligible rows select the native
// aligned-pair recipe; `preserve_legacy_scalar` pins the frozen scalar staging when required.
std::pair<at::Tensor, at::Tensor> fp8_gated_quantize_impl(const at::Tensor& gate_up,
                                                          Fp8GatedActivation kind,
                                                          bool gate_first,
                                                          bool preserve_legacy_scalar);
std::pair<at::Tensor, at::Tensor> fp8_scaled_gated_quantize_impl(const at::Tensor& values,
                                                                 const at::Tensor& row_scales,
                                                                 const at::Tensor& column_scales,
                                                                 at::ScalarType dtype,
                                                                 Fp8GatedActivation kind,
                                                                 bool gate_first,
                                                                 bool preserve_legacy_scalar);
}

// Add an optional model-dtype bias, apply one rounded unary activation, then dynamically quantize
// every row to E4M3 with an F32 scale. Ragged and wide rows retain the same scalar arithmetic.
std::pair<at::Tensor, at::Tensor>
fp8_activation_quantize(const at::Tensor& input, const at::Tensor& bias, Fp8Activation kind);
// Apply FP32 row and column scales, round once to dtype, add an optional model-dtype bias, apply one
// rounded unary activation, then dynamically quantize every row to E4M3. Aligned rows up to 20480
// activation columns avoid the scaled intermediate; wider or unaligned rows compose the same steps.
std::pair<at::Tensor, at::Tensor> fp8_scaled_activation_quantize(const at::Tensor& values,
                                                                 const at::Tensor& row_scales,
                                                                 const at::Tensor& column_scales,
                                                                 at::ScalarType dtype,
                                                                 const at::Tensor& bias,
                                                                 Fp8Activation kind);
// BF16 residual/RMS rounding and reduction match fused_add_rms_norm.
// Returns quantized normalized rows, residual output, and FP32 row scales.
std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_add_rms_quantize(const at::Tensor& x,
                                                                    const at::Tensor& residual,
                                                                    const at::Tensor& weight,
                                                                    double epsilon);
}
