/// \file
/// Gemma BF16 scoring and residual-scaling interfaces.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
/// Three pre-feedforward normalization outputs derived from one shared RMS
/// statistic of the same residual rows. `dense`/`routed` are BF16; `router` is
/// FP32 after the original unit-norm, scalar-root and learned-scale BF16 stages.
/// Any unsupported layout/device/dtype returns three undefined tensors so the
/// caller can keep the existing separate norm calls.
struct Gemma4PreFeedforwardNorm {
  at::Tensor dense;
  at::Tensor routed;
  at::Tensor router;
};

Gemma4PreFeedforwardNorm gemma4_bf16_pre_feedforward_norm(
    const at::Tensor& x,
    const at::Tensor& dense_scale,
    const at::Tensor& routed_scale,
    const at::Tensor& router_scale,
    const at::Tensor& root,
    double epsilon);

// Gather from the complete all-token score GEMM. Every capped operation rounds
// to BF16 before the next operation, preserving the declared publisher readout.
at::Tensor gemma4_bf16_sequence_score_readout(
    const at::Tensor& logits, const at::Tensor& cumulative, int64_t batch, double cap, bool capped);
// Preserve BF16 rounding after residual addition and layer-scalar multiply.
at::Tensor gemma4_bf16_residual_scale(const at::Tensor& residual,
                                      const at::Tensor& normalized,
                                      const at::Tensor& scale);
// Original native RMS tree, then BF16 norm, residual-add and optional scalar stages.
at::Tensor gemma4_bf16_norm_residual(const at::Tensor& residual,
                                     const at::Tensor& input,
                                     const at::Tensor& norm_scale,
                                     double epsilon,
                                     const at::Tensor& layer_scalar);
}
