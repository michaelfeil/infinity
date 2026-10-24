// Adapted from Candle 036f39e CPU operators (Apache-2.0).
// Sources: candle-core/src/{cpu/mod.rs,cpu/avx.rs,op.rs,cpu_backend/mod.rs};
// candle-nn/src/attention/cpu_flash/{varlen.rs,online_softmax.rs}.
// Keep the upstream LICENSE and NOTICE distributed with this library.
// Half rounding and F32 operation order are intentional. Compile this file
// without floating-point contraction; other model translation units are unchanged.
#pragma once
#include <ATen/ATen.h>
#include <utility>
#include <vector>

namespace tei::modern_cpu {
// Conservative eligibility for the independently proved AVX2 / gemm-f16 FMA
// cache profile with one ATen intra-op thread. Other CPUs or thread counts
// retain the existing ATen implementation; eligibility is checked during ready().
bool supported_profile();
/// Apply the qualified CPU Half LayerNorm recipe, preserving FP32
/// statistics and Half affine rounding.
at::Tensor norm(const at::Tensor&, const at::Tensor&, float);

/// Input-column range and prepared weights for one CPU linear reduction block.
struct LinearBlock {
  int64_t start, end;
  at::Tensor weight;
};

/// Input width and ordered weight blocks used by the reference CPU linear recipe.
struct LinearWeights {
  int64_t input_width;
  std::vector<LinearBlock> blocks;
};

/// Preconvert ordered K-dimension blocks without changing the Half checkpoint rounding.
LinearWeights prepare_linear(const at::Tensor& weight);
at::Tensor linear(const at::Tensor& x, const LinearWeights& weights, const at::Tensor& bias);
at::Tensor linear(const at::Tensor&, const at::Tensor&, const at::Tensor&);
/// Split concatenated gate/up rows and preserve Half rounding after each reference GELU operation.
at::Tensor gated_gelu(const at::Tensor&);
/// Evaluate packed sequence-local CPU attention for the independently qualified Half profile.
at::Tensor attention(const at::Tensor&,
                     const at::Tensor&,
                     const at::Tensor&,
                     const int32_t*,
                     int64_t,
                     float,
                     int64_t);
/// Evaluate scalar F32 sine/cosine before converting rotary tables to the requested model dtype.
std::pair<at::Tensor, at::Tensor> trig(const at::Tensor&, at::ScalarType);
/// Reduce one sequence to [1, width] with sequential Half additions and a Half reciprocal.
at::Tensor mean(const at::Tensor&);
} // namespace tei::modern_cpu
