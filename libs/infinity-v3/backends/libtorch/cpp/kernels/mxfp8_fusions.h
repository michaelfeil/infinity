// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <utility>

namespace infinity {
/// Native MXFP8 rows: E4M3 values plus swizzled UE8M0 scales for
/// every 32 adjacent K elements. `scales` is contiguous byte metadata sized by
/// mxfp8_scale_elements(rows, cols); it is not a per-row FP32 RMS scale.
struct Mxfp8Quantized {
  at::Tensor values;
  at::Tensor scales;
  int64_t rows = 0;
  int64_t cols = 0;
};

int64_t mxfp8_scale_elements(int64_t rows, int64_t cols);

/// Return a contiguous CUDA tensor whose data pointer is 16-byte aligned.
/// Aligned contiguous inputs are returned unchanged; unaligned contiguous views
/// and non-contiguous inputs are copied into fresh Torch storage. Empty tensors
/// are returned without a copy.
at::Tensor mxfp8_aligned_tensor(const at::Tensor& input,
                                at::ScalarType dtype,
                                int64_t dims,
                                const char* label);

Mxfp8Quantized mxfp8_quantize_rows(const at::Tensor& input);

/// Fused gate-first rounded SwiGLU followed by MXFP8 block quantization.
/// `gate_up` is contiguous model-dtype [rows, 2*width] and uses the same
/// rounded pair staging as fp8_gated_quantize.
Mxfp8Quantized mxfp8_gated_silu_quantize(const at::Tensor& gate_up);
}
