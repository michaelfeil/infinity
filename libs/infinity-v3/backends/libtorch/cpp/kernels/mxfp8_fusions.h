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

/// Fused Gemma4 MoE GeGLU followed by MXFP8 block quantization for routed rows.
/// `gate_up` is the contiguous BF16 physical expert workspace [capacity, 2*I]
/// and `row_map` is the contiguous int32 device map from logical slots to those
/// physical rows. Returns E4M3 values [capacity, I] plus UE8M0 scales in the
/// 128x4x4 swizzle. One CTA quantizes each logical slot directly from its mapped
/// physical row; padding activation rows are never read and the complete scale
/// tensor (including physical padding) is zeroed. The activation recipe matches
/// gemma4_moe_gelu_mul exactly: FP32 tanh GELU with the contracted cubic FMA,
/// rounded to BF16, multiplied by the BF16 up row, then rounded to BF16 before
/// the same block-32 E4M3/UE8M0 quantization as mxfp8_quantize_rows.
Mxfp8Quantized mxfp8_moe_gelu_quantize(const at::Tensor& gate_up, const at::Tensor& row_map);
}
