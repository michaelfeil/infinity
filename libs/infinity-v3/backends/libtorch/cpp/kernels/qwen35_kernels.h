/// \file
/// Native Gemma vision/audio primitives and Qwen3.5 gated-delta entry points.

#pragma once
#include "batching/packed_input.h"
#include <utility>

namespace infinity {
/// Rotate vision Q/K rows using already selected spatial cosine/sine tables.
at::Tensor gemma_vision_rotary_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&);
/// Gather rotary phases by packed position IDs and rotate token/head rows.
at::Tensor
gemma_rotary_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&);
/// Normalize audio rows with the audio tower's reduction and affine precision recipe.
at::Tensor gemma_audio_norm_cuda(const at::Tensor&, const at::Tensor&, double);
/// Normalize vision rows with FP32 statistics and the tower's affine rounding recipe.
at::Tensor gemma_vision_norm_cuda(const at::Tensor&, const at::Tensor&, double);
/// Pool a height/width patch grid into spatial kernel groups without mixing separate images.
at::Tensor gemma_vision_pool_cuda(const at::Tensor&, int64_t, int64_t, int64_t);
/// Return per-head normalized Q/K with position-indexed rotary phases applied.
std::pair<at::Tensor, at::Tensor> gemma3_qk_norm_rotary_cuda(const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             const at::Tensor&,
                                                             double);
/// Normalize Q/K with the BF16 reference RMS tree, apply composed staged-BF16 rotary
/// products, and unit-normalize V in one launch for packed BF16 rows of width 32 through 1024.
/// K/V may share input storage; both are read before any output is written. Unsupported
/// layouts and widths return an empty result for the caller's original path.
std::array<at::Tensor, 3> gemma4_qkv_norm_rotary_cuda(const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      const at::Tensor&,
                                                      double);
/// Query-only shared-KV variant using the same reference norm, staged rotary and width limits.
/// Unsupported layouts or widths leave the result undefined.
at::Tensor gemma4_q_norm_rotary_cuda(const at::Tensor&,
                                     const at::Tensor&,
                                     const at::Tensor&,
                                     const at::Tensor&,
                                     const at::Tensor&,
                                     double);
/// Apply a prepared Gemma affine scale, selecting the reference reduction when requested.
at::Tensor gemma_norm_cuda(const at::Tensor&, const at::Tensor&, double, bool);
/// Normalize a branch in FP32, round to FP16, then add the FP16 residual.
at::Tensor gemma_half_residual_norm_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double, bool);
// Undefined result preserves the ordinary legacy norm-plus-add fallback.
std::pair<at::Tensor, at::Tensor> gemma3_residual_prenorm_cuda(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
at::Tensor
gemma3_legacy_residual_norm_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, double);
/// Split concatenated gate/up channels, activate the gate, and return their elementwise product.
at::Tensor gemma_gated_cuda(const at::Tensor&, bool);
/// Activate rows using the selected GELU or SiLU recipe without changing their shape.
at::Tensor gemma_activation_cuda(const at::Tensor&, bool);
/// Materialize the BF16 convolution/SiLU rows for convolution-route tests.
at::Tensor qwen35_conv_cuda(const at::Tensor& qkv,
                            const at::Tensor& conv,
                            const at::Tensor& cumulative,
                            int64_t sequences,
                            const at::Tensor& sources = {},
                            const at::Tensor& history = {},
                            const at::Tensor& history_cumulative = {});
/// Prepare native FP32 Q/K and gate buffers from a BF16 convolution result.
std::pair<at::Tensor, at::Tensor> qwen35_gdn_prepare_cuda(const at::Tensor& convolution,
                                                          const at::Tensor& ab,
                                                          const at::Tensor& a_log,
                                                          const at::Tensor& dt_bias,
                                                          int64_t key_heads,
                                                          int64_t value_heads);
/// Fuse convolution/SiLU, provider Q/K normalization, gate preparation, and packing.
/// Optional cached history/source mapping resumes suffixes without a BF16 convolution
/// intermediate or FP32 Q/K staging buffers.
std::array<at::Tensor, 5> qwen35_gdn_prepare_pack_cuda(const at::Tensor& qkv,
                                                       const at::Tensor& conv,
                                                       const at::Tensor& cumulative,
                                                       const at::Tensor& ab,
                                                       const at::Tensor& a_log,
                                                       const at::Tensor& dt_bias,
                                                       int64_t key_heads,
                                                       int64_t value_heads,
                                                       const at::Tensor& sources = {},
                                                       const at::Tensor& history = {},
                                                       const at::Tensor& history_cumulative = {});
/// Compose packed depthwise convolution, F32 gated-delta state updates, and gated RMS output.
/// QKV and Z may be column slices of merged projections with their original row strides.
/// Optional initial/history buffers resume block execution;
/// cache_layer enables resident checkpoints. hopper_prefill opts qualified fresh requests into
/// the existing BF16 Hopper prefill provider without retaining a final state.
/// Optional two-byte-aligned quantized destinations receive E4M3 rows and FP32 scales directly;
/// that mode returns an undefined tensor instead of materializing BF16 output.
/// Cached suffixes use the tensor-core provider automatically when it is available.
at::Tensor qwen35_delta_cuda(const at::Tensor& qkv,
                             const at::Tensor& z,
                             const at::Tensor& ab,
                             const at::Tensor& conv,
                             const at::Tensor& a_log,
                             const at::Tensor& dt_bias,
                             const at::Tensor& norm,
                             const PackedInput& input,
                             int64_t key_heads,
                             int64_t value_heads,
                             double epsilon,
                             const at::Tensor& initial = {},
                             at::Tensor* final_state = nullptr,
                             const at::Tensor& history = {},
                             const at::Tensor& history_cumulative = {},
                             bool chunk_prefill = true,
                             int64_t cache_layer = -1,
                             bool allow_prefill = false,
                             const std::pair<at::Tensor, at::Tensor>* quantized_output = nullptr);
}
