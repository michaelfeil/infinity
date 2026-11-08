// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <utility>

namespace infinity {
/// Native NVFP4 rows: packed E2M1 values plus two-level E4M3 block scales.
/// Values are [rows, cols/2] Float4_e2m1fn_x2 (low nibble is the even K
/// element). `block_scales` is contiguous E4M3 metadata sized by
/// nvfp4_scale_elements(rows, cols) in the 128x4 SWIZZLE_32_4_4 layout, with
/// one scale for every 16 adjacent K elements. `global_scale` is a one-element
/// F32 tensor; dequantization multiplies value * block_scale * global_scale.
struct Nvfp4Quantized {
  at::Tensor values;
  at::Tensor block_scales;
  at::Tensor global_scale;
  int64_t rows = 0;
  int64_t cols = 0;
};

int64_t nvfp4_scale_elements(int64_t rows, int64_t cols);

/// Return a contiguous CUDA tensor whose data pointer is 16-byte aligned.
at::Tensor nvfp4_aligned_tensor(const at::Tensor& input,
                                at::ScalarType dtype,
                                int64_t dims,
                                const char* label);

/// Quantize BF16 rows with a global scale derived from the actual batch amax.
/// Inputs are expected to be finite; non-finite elements are excluded
/// from amax and encoded as zero. No calibration data is retained and no host
/// synchronization is performed.
/// `power2_global` rounds the activation global scale up to a power of two.
/// Nvfp4Linear uses this for activations and the exact amax scale for weights.
Nvfp4Quantized nvfp4_quantize_rows(const at::Tensor& input, bool power2_global = false);

/// Fuse the standard gate-first BF16-rounded SwiGLU with NVFP4 quantization.
/// The activation arithmetic matches the shared fused_gated_activation code-5
/// pair kernel. Quantization uses a power-of-two global activation scale.
Nvfp4Quantized nvfp4_swiglu_quantize(const at::Tensor& gate_up);

/// Whether this runtime/device can execute native NVFP4 block-scaled GEMMs.
bool nvfp4_provider_supported(const c10::Device& device);

/// Load-time geometry qualification for the MLP projection shapes.
bool nvfp4_geometry_supported(const c10::Device& device, int64_t m, int64_t n, int64_t k);

/// NVFP4 projection state. Weights are quantized once at preparation; inputs
/// are model-dtype rows quantized on each forward with a batch-global scale.
class Nvfp4Linear {
  Nvfp4Quantized matrix_;
  at::ScalarType dtype_ = at::kBFloat16;

public:
  explicit Nvfp4Linear(const at::Tensor& weight);

  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  at::Tensor forward_quantized(const Nvfp4Quantized& input, const at::Tensor& bias = {}) const;
  Nvfp4Quantized swiglu_quantize(const at::Tensor& gate_up) const;

  int64_t input_width() const {
    return matrix_.cols;
  }

  int64_t output_width() const {
    return matrix_.rows;
  }

  const Nvfp4Quantized& matrix() const {
    return matrix_;
  }
};

/// GEMM for quantized NVFP4 operands with BF16 output and no bias.
at::Tensor nvfp4_gemm(const Nvfp4Quantized& lhs, const Nvfp4Quantized& rhs);
}
