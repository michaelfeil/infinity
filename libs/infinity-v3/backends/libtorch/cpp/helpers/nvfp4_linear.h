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
/// Activation rows also carry `row_scales`, a contiguous F32 [rows] vector of
/// per-row power-of-two scales with `global_scale` fixed to one. Weight rows
/// leave `row_scales` undefined and use their exact tensor-global scale.
struct Nvfp4Quantized {
  at::Tensor values;
  at::Tensor block_scales;
  at::Tensor global_scale;
  int64_t rows = 0;
  int64_t cols = 0;
  at::Tensor row_scales;
};

int64_t nvfp4_scale_elements(int64_t rows, int64_t cols);

/// Return a contiguous CUDA tensor whose data pointer is 16-byte aligned.
at::Tensor nvfp4_aligned_tensor(const at::Tensor& input,
                                at::ScalarType dtype,
                                int64_t dims,
                                const char* label);

/// Quantize BF16 rows. With `rowwise` false the weights use the exact
/// tensor-global amax scale. With `rowwise` true every activation row uses its
/// own power-of-two scale S = 2^ceil(log2(row_amax / (6*448))) (S=1 for a zero
/// or all-nonfinite row), the GEMM tensor scale is one, and `row_scales`
/// carries the per-row factor applied to the GEMM output.
/// Inputs are expected to be finite; non-finite elements are excluded from
/// amax and encoded as zero. No calibration data is retained and no host
/// synchronization is performed.
Nvfp4Quantized nvfp4_quantize_rows(const at::Tensor& input, bool rowwise = false);

/// Fuse the standard gate-first BF16-rounded SwiGLU with NVFP4 quantization.
/// The activation arithmetic matches the shared fused_gated_activation code-5
/// pair kernel. Each activation row is quantized with its own power-of-two
/// scale and reports it in `row_scales`. Optional contiguous F32 [rows]
/// `gate_up_row_scales` restore per-row GEMM scales on the gate/up operands
/// before the activation, exactly matching a separate BF16 row-scale restore.
Nvfp4Quantized nvfp4_swiglu_quantize(const at::Tensor& gate_up,
                                     const at::Tensor& gate_up_row_scales = {});

/// Whether this runtime/device can execute native NVFP4 block-scaled GEMMs.
bool nvfp4_provider_supported(const c10::Device& device);

/// Load-time geometry qualification for the MLP projection shapes.
bool nvfp4_geometry_supported(const c10::Device& device, int64_t m, int64_t n, int64_t k);

/// NVFP4 projection state. Weights are quantized once at preparation; inputs
/// are model-dtype rows quantized on each forward with per-row power-of-two
/// activation scales.
class Nvfp4Linear {
  Nvfp4Quantized matrix_;
  at::ScalarType dtype_ = at::kBFloat16;

public:
  explicit Nvfp4Linear(const at::Tensor& weight);

  at::Tensor forward(const at::Tensor& input, const at::Tensor& bias = {}) const;
  at::Tensor forward_quantized(const Nvfp4Quantized& input, const at::Tensor& bias = {}) const;
  Nvfp4Quantized forward_swiglu_quantized(const at::Tensor& input) const;
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

/// Apply per-row F32 activation scales to a BF16 GEMM output in place. Uses a
/// typed TensorIterator GPU kernel (BF16 value, float scale, BF16 result) so
/// the framework does not insert per-element dynamic casting.
void nvfp4_apply_row_scales_(at::Tensor& output, const at::Tensor& row_scales);

/// GEMM for quantized NVFP4 operands with BF16 output and no bias. When the
/// left operand carries per-row activation scales, they are applied to the
/// fresh BF16 product in place unless `apply_row_scales` is false. The right
/// (weight) operand must not carry row scales.
at::Tensor
nvfp4_gemm(const Nvfp4Quantized& lhs, const Nvfp4Quantized& rhs, bool apply_row_scales = true);
}
