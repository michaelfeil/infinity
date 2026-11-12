// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include "helpers/fp8_activation.h"
#include "kernels/mxfp8_fusions.h"
#include <ATen/ATen.h>
#include <utility>

namespace infinity {
/// Whether this runtime/device can execute native MXFP8 block-scaled GEMMs.
bool mxfp8_provider_supported(const c10::Device& device);

/// Load-time cuBLASLt heuristic probe for the actual N/K at small M. This does
/// not convert weights and is called before any MXFP8 weight storage is built.
bool mxfp8_geometry_supported(const c10::Device& device, int64_t m, int64_t n, int64_t k);

/// MXFP8 MLP projection state. Weights are E4M3 with swizzled UE8M0 block
/// scales; no rowwise FP8 or model-dtype weight copy is retained after
/// construction. Forward inputs are model-dtype rows (quantized internally) or
/// E4M3 rows carrying byte block-scale metadata. Per-row FP32 RMS scales are
/// explicitly rejected.
class Mxfp8Linear {
  Mxfp8Quantized matrix_;
  at::ScalarType dtype_ = at::kBFloat16;

public:
  explicit Mxfp8Linear(const at::Tensor& weight);

  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  at::Tensor forward_quantized(const at::Tensor& input,
                               const at::Tensor& scales,
                               const at::Tensor& bias = {}) const;
  std::pair<at::Tensor, at::Tensor> forward_gated_quantized(const at::Tensor& input,
                                                            const at::Tensor& rowwise_scales,
                                                            Fp8GatedActivation kind,
                                                            bool gate_first) const;

  /// Fused model-dtype projection, gate-first gated activation and MXFP8 block
  /// quantization. Returns E4M3 values with swizzled UE8M0 block scales.
  Mxfp8Quantized forward_gated_quantize(const at::Tensor& input, Fp8GatedActivation kind) const;

  int64_t input_width() const {
    return matrix_.cols;
  }

  int64_t output_width() const {
    return matrix_.rows;
  }

  const Mxfp8Quantized& matrix() const {
    return matrix_;
  }
};
}
