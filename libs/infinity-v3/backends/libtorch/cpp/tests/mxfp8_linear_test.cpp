// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/dense_mlp.h"
#include "helpers/mxfp8_linear.h"
#include "kernels/fp8_fusions.h"
#include "kernels/mxfp8_fusions.h"
#include "kernels/qwen35_kernels.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace {
constexpr int64_t kBlock = 32;
constexpr int64_t kTile = 128;
constexpr int64_t kTileK = 4;

int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

int64_t inner_dim(int64_t cols) {
  return round_up(cols / 32, kTileK);
}

int64_t scale_offset(int64_t outer, int64_t block, int64_t inner) {
  const int64_t sf_outer = outer / kTile;
  const int64_t inner_major = block & ~(kTileK - 1);
  return (inner_major + sf_outer * inner) * kTile + (outer % 32) * 16 + ((outer % kTile) / 32) * 4 +
         (block % kTileK);
}

float e4m3_to_float(uint8_t bits) {
  const int sign = (bits >> 7) & 1;
  const int exponent = (bits >> 3) & 0xf;
  const int mantissa = bits & 0x7;
  if (exponent == 15 && mantissa == 7)
    return NAN;
  float value = 0.0f;
  if (exponent == 0)
    value = (static_cast<float>(mantissa) / 8.0f) * std::ldexp(1.0f, -6);
  else
    value = (1.0f + static_cast<float>(mantissa) / 8.0f) * std::ldexp(1.0f, exponent - 7);
  return sign ? -value : value;
}

at::Tensor cpu_dequantize(const infinity::Mxfp8Quantized& q) {
  auto q_cpu = q.values.to(at::kCPU).contiguous();
  auto s_cpu = q.scales.to(at::kCPU).contiguous();
  auto out = at::empty({q.rows, q.cols}, at::TensorOptions().dtype(at::kFloat));
  const auto* qb = reinterpret_cast<const uint8_t*>(q_cpu.const_data_ptr());
  const auto* sb = reinterpret_cast<const uint8_t*>(s_cpu.const_data_ptr());
  auto* out_ptr = out.mutable_data_ptr<float>();
  const int64_t inner = inner_dim(q.cols);
  for (int64_t row = 0; row < q.rows; ++row) {
    for (int64_t col = 0; col < q.cols; ++col) {
      const uint8_t scale = sb[scale_offset(row, col / 32, inner)];
      const float value = e4m3_to_float(qb[row * q.cols + col]);
      out_ptr[row * q.cols + col] =
          scale == 255 ? NAN : value * std::ldexp(1.0f, static_cast<int>(scale) - 127);
    }
  }
  return out;
}

at::Tensor gated_activation_reference(const at::Tensor& gate_up) {
  const int64_t width = gate_up.size(1) / 2;
  return infinity::mlp_activation(gate_up.narrow(1, 0, width),
                                  infinity::MlpActivation::RoundedSilu) *
         gate_up.narrow(1, width, width);
}

void check_precise_gated_quantize(const at::Tensor& gate_up, int64_t width) {
  auto expected = infinity::mxfp8_quantize_rows(infinity::gemma_gated_cuda(gate_up, true));
  auto actual = infinity::mxfp8_gated_quantize(gate_up, infinity::Fp8GatedActivation::PreciseGelu);
  TORCH_CHECK(expected.values.sizes() == actual.values.sizes() &&
                  expected.scales.sizes() == actual.scales.sizes(),
              "Precise MXFP8 gated quantization changed geometry at width ",
              width);
  TORCH_CHECK(at::equal(expected.values.view(at::kByte), actual.values.view(at::kByte)) &&
                  at::equal(expected.scales, actual.scales),
              "Precise MXFP8 gated quantization differs from gemma_gated_cuda + mxfp8_quantize "
              "at width ",
              width);
}

void check_scale_layout(const at::Tensor& x, const infinity::Mxfp8Quantized& q) {
  auto x_cpu = x.to(at::kCPU).contiguous();
  auto q_cpu = q.values.to(at::kCPU).contiguous();
  auto s_cpu = q.scales.to(at::kCPU).contiguous();
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x_cpu.const_data_ptr());
  const auto* qb = reinterpret_cast<const uint8_t*>(q_cpu.const_data_ptr());
  const auto* sb = reinterpret_cast<const uint8_t*>(s_cpu.const_data_ptr());
  const int64_t rows = q.rows, cols = q.cols, inner = inner_dim(cols);
  TORCH_CHECK(q.scales.numel() == round_up(rows, kTile) * inner, "MXFP8 scale size");
  int64_t invalid_scales = 0, checked = 0;
  uint8_t distinct[256] = {};
  int distinct_count = 0;
  double l2 = 0.0, ref_l2 = 0.0, max_abs = 0.0;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t block = 0; block < cols / 32; ++block) {
      float amax = 0.0f;
      bool nonfinite = false;
      for (int lane = 0; lane < 32; ++lane) {
        const float value = __bfloat162float(xb[row * cols + block * 32 + lane]);
        amax = std::max(amax, std::fabs(value));
        nonfinite |= !std::isfinite(value);
      }
      const uint8_t actual = sb[scale_offset(row, block, inner)];
      if (!distinct[actual]) {
        distinct[actual] = 1;
        ++distinct_count;
      }
      if (nonfinite) {
        invalid_scales += actual != 255;
      } else if (amax == 0.0f) {
        invalid_scales += actual != 0;
      } else {
        if (actual == 255) {
          ++invalid_scales;
        } else {
          const int e = static_cast<int>(actual) - 127;
          const double upper = 448.0 * std::exp2(static_cast<double>(e));
          const double lower = 448.0 * std::exp2(static_cast<double>(e - 1));
          if (amax > upper || (e > -127 && amax <= lower))
            ++invalid_scales;
        }
      }
      for (int lane = 0; lane < 32; ++lane) {
        const int64_t col = block * 32 + lane;
        const float original = __bfloat162float(xb[row * cols + col]);
        if (!std::isfinite(original))
          continue;
        const float dequantized = actual == 255
                                      ? NAN
                                      : e4m3_to_float(qb[row * cols + col]) *
                                            std::ldexp(1.0f, static_cast<int>(actual) - 127);
        const double diff = dequantized - original;
        l2 += diff * diff;
        ref_l2 += static_cast<double>(original) * original;
        max_abs = std::max(max_abs, std::fabs(diff));
        ++checked;
      }
    }
  }
  for (int64_t row = rows; row < round_up(rows, kTile); ++row)
    for (int64_t block = 0; block < inner; ++block)
      invalid_scales += sb[scale_offset(row, block, inner)] != 0;
  const double rel_l2 = ref_l2 > 0 ? std::sqrt(l2 / ref_l2) : 0.0;
  std::printf("MXFP8 scale layout: rows=%lld checked=%lld invalid=%lld distinct=%d "
              "rel_l2=%.6g max_abs=%.6g\n",
              static_cast<long long>(rows),
              static_cast<long long>(checked),
              static_cast<long long>(invalid_scales),
              distinct_count,
              rel_l2,
              max_abs);
  TORCH_CHECK(invalid_scales == 0, "MXFP8 scale bytes, exponent bounds, or padding invalid");
  TORCH_CHECK(distinct_count > 1, "MXFP8 scales look rowwise-scalar rather than block-varying");
  TORCH_CHECK(rel_l2 < 0.05, "MXFP8 dequantized input error too large: ", rel_l2);
  TORCH_CHECK(max_abs < 0.5, "MXFP8 dequantized input max error too large: ", max_abs);
}

void check_special_values() {
  auto cpu_options = at::TensorOptions().dtype(at::kBFloat16);
  auto special_cpu = at::zeros({1, 5 * kBlock}, cpu_options);
  special_cpu.narrow(1, kBlock, kBlock).fill_(std::ldexp(1.0f, -133));
  special_cpu.narrow(1, 2 * kBlock, kBlock).fill_(std::numeric_limits<float>::infinity());
  special_cpu.narrow(1, 3 * kBlock, kBlock).fill_(-std::numeric_limits<float>::infinity());
  special_cpu.narrow(1, 4 * kBlock, kBlock).fill_(std::numeric_limits<float>::quiet_NaN());
  auto special = special_cpu.to(at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA));
  auto q = infinity::mxfp8_quantize_rows(special);
  auto s_cpu = q.scales.to(at::kCPU).contiguous();
  auto q_cpu = q.values.to(at::kCPU).contiguous();
  const auto* sb = reinterpret_cast<const uint8_t*>(s_cpu.const_data_ptr());
  const auto* qb = reinterpret_cast<const uint8_t*>(q_cpu.const_data_ptr());
  const int64_t inner = inner_dim(q.cols);
  TORCH_CHECK(sb[scale_offset(0, 0, inner)] == 0, "zero block must keep scale byte 0");
  TORCH_CHECK(sb[scale_offset(0, 1, inner)] <= 254, "subnormal block scale out of range");
  for (int block = 2; block < 5; ++block) {
    TORCH_CHECK(sb[scale_offset(0, block, inner)] == 255,
                "nonfinite block must use UE8M0 NaN scale 255");
    for (int lane = 0; lane < 32; ++lane)
      TORCH_CHECK(qb[block * 32 + lane] == 0x7f, "nonfinite block values must be E4M3 NaN");
  }
  auto dequantized = cpu_dequantize(q);
  TORCH_CHECK(dequantized[0][0].item<float>() == 0.0f, "zero block dequantized nonzero");
  TORCH_CHECK(std::isfinite(dequantized[0][1 * kBlock].item<float>()),
              "subnormal block dequantized nonfinite");
  for (int block = 2; block < 5; ++block)
    TORCH_CHECK(std::isnan(dequantized[0][block * kBlock].item<float>()),
                "nonfinite block must dequantize to NaN");
  infinity::Mxfp8Linear linear(at::ones({32, special.size(1)}, special.options()));
  TORCH_CHECK(at::isnan(linear.forward(special)).all().item<bool>(),
              "nonfinite MXFP8 blocks must propagate NaN through GEMM");
  auto gated = infinity::mxfp8_gated_silu_quantize(at::cat({special, at::ones_like(special)}, 1));
  TORCH_CHECK(at::isnan(linear.forward_quantized(gated.values, gated.scales)).all().item<bool>(),
              "nonfinite gated MXFP8 blocks must propagate NaN through GEMM");
  std::printf("MXFP8 special values: zero/subnormal finite; +/-Inf/NaN -> scale 255 + NaN\n");
}

void check_alignment_views() {
  const int64_t rows = 257, hidden = 256, output = 192;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto weight_base = at::randn({output * hidden + 1}, options);
  auto weight_view = weight_base.narrow(0, 1, output * hidden).view({output, hidden});
  auto input_base = at::randn({rows * hidden + 1}, options);
  auto input_view = input_base.narrow(0, 1, rows * hidden).view({rows, hidden});
  TORCH_CHECK(weight_view.is_contiguous() && input_view.is_contiguous(),
              "offset views must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(weight_view.const_data_ptr()) % 16 != 0 &&
                  reinterpret_cast<uintptr_t>(input_view.const_data_ptr()) % 16 != 0,
              "offset views must be unaligned");
  auto q = infinity::mxfp8_quantize_rows(input_view);
  auto q_aligned = infinity::mxfp8_quantize_rows(input_view.clone());
  TORCH_CHECK(at::equal(q.values, q_aligned.values) && at::equal(q.scales, q_aligned.scales),
              "unaligned quantize view changed values or scales");
  infinity::Mxfp8Linear unaligned(weight_view);
  infinity::Mxfp8Linear aligned(weight_view.clone());
  TORCH_CHECK(at::equal(unaligned.matrix().values, aligned.matrix().values) &&
                  at::equal(unaligned.matrix().scales, aligned.matrix().scales),
              "unaligned weight view changed quantized weight");
  TORCH_CHECK(at::equal(unaligned.forward(input_view), aligned.forward(input_view.clone())),
              "unaligned forward input changed output");

  const int64_t width = 128;
  auto gate_base = at::randn({rows * 2 * width + 1}, options);
  auto gate_view = gate_base.narrow(0, 1, rows * 2 * width).view({rows, 2 * width});
  auto gated = infinity::mxfp8_gated_silu_quantize(gate_view);
  auto gated_aligned = infinity::mxfp8_gated_silu_quantize(gate_view.clone());
  TORCH_CHECK(at::equal(gated.values, gated_aligned.values) &&
                  at::equal(gated.scales, gated_aligned.scales),
              "unaligned gated view changed values or scales");
  std::printf("MXFP8 alignment: unaligned contiguous BF16 views normalized\n");
}

void check_empty_rows() {
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto q = infinity::mxfp8_quantize_rows(at::empty({0, 2560}, options));
  TORCH_CHECK(q.rows == 0 && q.cols == 2560 && q.values.sizes() == at::IntArrayRef({0, 2560}) &&
                  q.scales.numel() == 0,
              "empty MXFP8 quantize");
  auto gated = infinity::mxfp8_gated_silu_quantize(at::empty({0, 256}, options));
  TORCH_CHECK(gated.rows == 0 && gated.cols == 128 &&
                  gated.values.sizes() == at::IntArrayRef({0, 128}) && gated.scales.numel() == 0,
              "empty MXFP8 gated quantize");

  auto weight = (at::randn({192, 256}, options) * 0.05).to(at::kBFloat16);
  infinity::Mxfp8Linear linear(weight);
  auto output = linear.forward(at::empty({0, 256}, options));
  TORCH_CHECK(output.sizes() == at::IntArrayRef({0, 192}) && output.scalar_type() == at::kBFloat16,
              "empty MXFP8 linear output");
  auto empty_values = at::empty({0, 256}, options.dtype(at::kFloat8_e4m3fn));
  auto empty_scales = at::zeros({0}, options.dtype(at::kByte));
  auto quantized_output = linear.forward_quantized(empty_values, empty_scales);
  TORCH_CHECK(quantized_output.sizes() == at::IntArrayRef({0, 192}),
              "empty MXFP8 quantized output");
  auto gated_pair = linear.forward_gated_quantized(
      at::empty({0, 256}, options), at::Tensor(), infinity::Fp8GatedActivation::RoundedSilu, true);
  TORCH_CHECK(gated_pair.first.sizes() == at::IntArrayRef({0, 96}) &&
                  gated_pair.second.numel() == 0,
              "empty MXFP8 gated linear output");

  auto gate_weight = (at::randn({256, 256}, options) * 0.05).to(at::kBFloat16);
  auto down_weight = (at::randn({256, 128}, options) * 0.05).to(at::kBFloat16);
  auto recipe = infinity::MlpActivationRecipe::numerical(
      [](const at::Tensor& both) {
        const int64_t width = both.size(1) / 2;
        return infinity::mlp_activation(both.narrow(1, 0, width),
                                        infinity::MlpActivation::RoundedSilu) *
               both.narrow(1, width, width);
      },
      true);
  recipe.fusion = infinity::MlpFusion::RoundedSwiGlu;
  infinity::DenseMlp mlp({gate_weight}, {down_weight}, recipe);
  TORCH_CHECK(mlp.supports_mxfp8(), "empty-row test MLP should qualify");
  mlp.prepare(infinity::Quantization::Mxfp8);
  auto mlp_output = mlp.forward(at::empty({0, 256}, options));
  TORCH_CHECK(mlp_output.sizes() == at::IntArrayRef({0, 256}), "empty MXFP8 DenseMlp output");
  std::printf("MXFP8 empty rows: quantize/gated/linear/DenseMlp return empties\n");
}

infinity::MlpActivationRecipe qwen_gated_recipe() {
  auto recipe = infinity::MlpActivationRecipe::numerical(
      [](const at::Tensor& both) {
        const int64_t width = both.size(-1) / 2;
        return infinity::mlp_activation(both.narrow(-1, 0, width),
                                        infinity::MlpActivation::RoundedSilu) *
               both.narrow(-1, width, width);
      },
      true);
  recipe.fusion = infinity::MlpFusion::RoundedSwiGlu;
  return recipe;
}

void check_mxfp8_mlp_path() {
  const int64_t rows = 257, hidden = 256, intermediate = 128;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto x = (at::randn({rows, hidden}, options) * 0.5).to(at::kBFloat16);
  auto gate_up_weight = (at::randn({2 * intermediate, hidden}, options) * 0.05).to(at::kBFloat16);
  auto down_weight = (at::randn({hidden, intermediate}, options) * 0.05).to(at::kBFloat16);
  infinity::DenseMlp mlp({gate_up_weight}, {down_weight}, qwen_gated_recipe());
  TORCH_CHECK(mlp.supports_mxfp8(), "qualified packed SwiGLU MLP was not MXFP8-qualified");
  mlp.prepare(infinity::Quantization::Mxfp8);
  TORCH_CHECK(mlp.is_quantized() && !mlp.supports_rowwise_rms_input(),
              "MXFP8 DenseMlp capability contract");
  auto output = mlp.forward(x);
  TORCH_CHECK(output.sizes() == at::IntArrayRef({rows, hidden}) &&
                  output.scalar_type() == at::kBFloat16 && at::isfinite(output).all().item<bool>(),
              "MXFP8 DenseMlp output contract");

  auto gate_up_ref =
      at::matmul(x.to(at::kFloat), gate_up_weight.to(at::kFloat).t()).to(at::kBFloat16);
  auto activated_ref = gated_activation_reference(gate_up_ref);
  auto output_ref =
      at::matmul(activated_ref.to(at::kFloat), down_weight.to(at::kFloat).t()).to(at::kBFloat16);
  auto diff = (output.to(at::kFloat) - output_ref.to(at::kFloat)).abs();
  const double l2 = std::sqrt(at::sum(diff * diff).item<double>());
  const double ref_l2 =
      std::sqrt(at::sum(output_ref.to(at::kFloat) * output_ref.to(at::kFloat)).item<double>());
  const double rel_l2 = ref_l2 > 0 ? l2 / ref_l2 : 0.0;
  std::printf("MXFP8 DenseMlp vs model-dtype gated reference: rel_l2=%.6g max_abs=%.6g\n",
              rel_l2,
              diff.max().item<double>());
  TORCH_CHECK(rel_l2 < 0.15, "MXFP8 DenseMlp path diverged: rel_l2=", rel_l2);
}

void check_mxfp8_qualification_rejections() {
  const int64_t hidden = 256, intermediate = 128;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  auto gate_up_weight = (at::randn({2 * intermediate, hidden}, options) * 0.05).to(at::kBFloat16);
  auto down_weight = (at::randn({hidden, intermediate}, options) * 0.05).to(at::kBFloat16);
  auto unary_weight = (at::randn({intermediate, hidden}, options) * 0.05).to(at::kBFloat16);

  auto rowwise = [](infinity::DenseMlp& mlp, const char* label) {
    TORCH_CHECK(!mlp.supports_mxfp8(), label, " must not select MXFP8");
    mlp.prepare(infinity::Quantization::Fp8);
    TORCH_CHECK(mlp.is_quantized(), label, " must select rowwise FP8");
  };

  auto gelu = infinity::MlpActivationRecipe::gated(infinity::MlpActivation::RoundedGelu);
  gelu.fusion = infinity::MlpFusion::RoundedBiasGelu;
  infinity::DenseMlp gelu_mlp({gate_up_weight}, {down_weight}, gelu);
  rowwise(gelu_mlp, "GELU MLP");

  auto reversed = qwen_gated_recipe();
  reversed.gate_first = false;
  infinity::DenseMlp reversed_mlp({gate_up_weight}, {down_weight}, reversed);
  rowwise(reversed_mlp, "reversed gate MLP");

  infinity::MlpActivationRecipe unary(infinity::MlpActivation::RoundedSilu);
  infinity::DenseMlp unary_mlp({unary_weight}, {down_weight}, unary);
  rowwise(unary_mlp, "unary MLP");

  auto gate_bias = (at::randn({2 * intermediate}, options) * 0.05).to(at::kBFloat16);
  infinity::DenseMlp bias_mlp({gate_up_weight, gate_bias}, {down_weight}, qwen_gated_recipe());
  rowwise(bias_mlp, "gate/up bias MLP");

  infinity::ProjectionRecipe clipped;
  clipped.input_clip = std::array<double, 2>{-1.0, 1.0};
  infinity::DenseMlp clipped_mlp(
      {gate_up_weight, at::Tensor(), clipped}, {down_weight}, qwen_gated_recipe());
  rowwise(clipped_mlp, "clipped MLP");

  infinity::ProjectionRecipe cast;
  cast.cast_input = true;
  infinity::DenseMlp cast_mlp(
      {gate_up_weight, at::Tensor(), cast}, {down_weight}, qwen_gated_recipe());
  rowwise(cast_mlp, "cast-input MLP");

  auto wide_gate = (at::randn({41024, 32}, options) * 0.05).to(at::kBFloat16);
  auto wide_down = (at::randn({32, 20512}, options) * 0.05).to(at::kBFloat16);
  infinity::DenseMlp wide_mlp({wide_gate}, {wide_down}, qwen_gated_recipe());
  rowwise(wide_mlp, "wide packed MLP");

  auto down_bias = (at::randn({hidden}, options) * 0.05).to(at::kBFloat16);
  infinity::DenseMlp down_bias_mlp({gate_up_weight}, {down_weight, down_bias}, qwen_gated_recipe());
  TORCH_CHECK(down_bias_mlp.supports_mxfp8(), "down-projection bias is supported after rounding");
  std::printf(
      "MXFP8 qualification: packed SwiGLU accepted; GELU/reversed/unary/bias/clip/cast/wide select rowwise\n");
}

void check_k_padding() {
  const int64_t rows = 257, output = 192;
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  for (const int64_t k : {1056, 2592}) {
    auto x = (at::randn({rows, k}, options) * 0.5).to(at::kBFloat16);
    auto w = (at::randn({output, k}, options) * 0.05).to(at::kBFloat16);
    auto qx = infinity::mxfp8_quantize_rows(x);
    auto qw = infinity::mxfp8_quantize_rows(w);
    const int64_t inner = inner_dim(k);
    TORCH_CHECK(qx.scales.numel() == round_up(rows, kTile) * inner, "K-padding scale size");
    auto s_cpu = qx.scales.to(at::kCPU).contiguous();
    const auto* sb = reinterpret_cast<const uint8_t*>(s_cpu.const_data_ptr());
    int64_t invalid_padding = 0;
    for (int64_t row = 0; row < rows; ++row)
      for (int64_t block = k / 32; block < inner; ++block)
        invalid_padding += sb[scale_offset(row, block, inner)] != 0;
    for (int64_t row = rows; row < round_up(rows, kTile); ++row)
      for (int64_t block = 0; block < inner; ++block)
        invalid_padding += sb[scale_offset(row, block, inner)] != 0;
    TORCH_CHECK(invalid_padding == 0, "K-padding metadata must be zero");
    if (!infinity::mxfp8_geometry_supported(at::Device(at::kCUDA, 0), rows, output, k)) {
      std::printf("MXFP8 K=%lld layout verified; provider GEMM probe skipped\n",
                  static_cast<long long>(k));
      continue;
    }
    infinity::Mxfp8Linear linear(w);
    auto output_tensor = linear.forward(x);
    auto x_deq = cpu_dequantize(qx).to(at::kCUDA);
    auto w_deq = cpu_dequantize(qw).to(at::kCUDA);
    auto ref = at::matmul(x_deq, w_deq.t()).to(at::kBFloat16);
    auto diff = (output_tensor.to(at::kFloat) - ref.to(at::kFloat)).abs();
    const double l2 = std::sqrt(at::sum(diff * diff).item<double>());
    const double ref_l2 =
        std::sqrt(at::sum(ref.to(at::kFloat) * ref.to(at::kFloat)).item<double>());
    const double rel_l2 = ref_l2 > 0 ? l2 / ref_l2 : 0.0;
    std::printf("MXFP8 K=%lld padding: rel_l2=%.6g max_abs=%.6g\n",
                static_cast<long long>(k),
                rel_l2,
                diff.max().item<double>());
    TORCH_CHECK(rel_l2 < 1e-3, "K-padding GEMM reference mismatch: ", rel_l2);
    auto gate_up_weight = (at::randn({256, k}, options) * 0.05).to(at::kBFloat16);
    auto down_weight = (at::randn({256, 128}, options) * 0.05).to(at::kBFloat16);
    infinity::DenseMlp mlp({gate_up_weight}, {down_weight}, qwen_gated_recipe());
    TORCH_CHECK(mlp.supports_mxfp8(), "K divisible by 32 must qualify with padded metadata");
    mlp.prepare(infinity::Quantization::Mxfp8);
    auto padded_output = mlp.forward(x);
    TORCH_CHECK(padded_output.sizes() == at::IntArrayRef({rows, 256}) &&
                    at::isfinite(padded_output).all().item<bool>(),
                "K-padding MXFP8 MLP changed actual rows or produced nonfinite output");
  }
}
} // namespace

int main(int argc, char** argv) {
  bool large_probe = false;
  for (int i = 1; i < argc; ++i)
    large_probe |= std::strcmp(argv[i], "--large-probe") == 0;
  if (at::cuda::device_count() == 0) {
    std::printf("SKIP: no CUDA device\n");
    return 77;
  }
  at::globalContext().setAllowTF32CuBLAS(false);
  c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
  if (!infinity::mxfp8_provider_supported(at::Device(at::kCUDA, 0))) {
    std::printf("SKIP: MXFP8 provider unsupported on this device/runtime\n");
    return 77;
  }
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
  if (large_probe) {
    const int64_t rows = 700000, cols = 2560;
    auto large = at::randn({rows, cols}, options);
    auto q = infinity::mxfp8_quantize_rows(large);
    TORCH_CHECK(q.rows == rows && q.cols == cols &&
                    q.values.sizes() == at::IntArrayRef({rows, cols}) &&
                    q.scales.numel() == round_up(rows, kTile) * inner_dim(cols),
                "700k-row MXFP8 quantize geometry");
    const auto device = at::Device(at::kCUDA, 0);
    const bool up_ok = infinity::mxfp8_geometry_supported(device, rows, 19456, 2560);
    const bool down_ok = infinity::mxfp8_geometry_supported(device, rows, 2560, 9728);
    TORCH_CHECK(up_ok && down_ok, "700k-row MXFP8 provider probe rejected actual N/K");
    auto down_input = at::randn({rows, 9728}, options);
    auto down_q = infinity::mxfp8_quantize_rows(down_input);
    auto down_weight = (at::randn({2560, 9728}, options) * 0.05).to(at::kBFloat16);
    infinity::Mxfp8Linear down_linear(down_weight);
    auto down_out = down_linear.forward_quantized(down_q.values, down_q.scales);
    TORCH_CHECK(down_out.sizes() == at::IntArrayRef({rows, 2560}),
                "700k-row MXFP8 provider output shape");
    TORCH_CHECK(at::isfinite(down_out.narrow(0, 0, 1)).all().item<bool>(),
                "700k-row MXFP8 provider produced nonfinite output");
    std::printf("MXFP8 700k-row probe: scales=%lld up=%d down=%d provider=SUCCESS\n",
                static_cast<long long>(q.scales.numel()),
                up_ok ? 1 : 0,
                down_ok ? 1 : 0);
    return 0;
  }
  auto x = (at::randn({257, 2560}, options) * 0.5).to(at::kBFloat16);
  auto qx = infinity::mxfp8_quantize_rows(x);
  check_scale_layout(x, qx);

  auto weight = (at::randn({192, 256}, options) * 0.05).to(at::kBFloat16);
  auto input = (at::randn({257, 256}, options) * 0.5).to(at::kBFloat16);
  infinity::Mxfp8Linear linear(weight);
  auto output = linear.forward(input);
  auto qi = infinity::mxfp8_quantize_rows(input);
  auto qw = infinity::mxfp8_quantize_rows(weight);
  auto ref = at::matmul(cpu_dequantize(qi).to(at::kCUDA), cpu_dequantize(qw).to(at::kCUDA).t())
                 .to(at::kBFloat16);
  auto diff = (output.to(at::kFloat) - ref.to(at::kFloat)).abs();
  const double l2 = std::sqrt(at::sum(diff * diff).item<double>());
  const double ref_l2 = std::sqrt(at::sum(ref.to(at::kFloat) * ref.to(at::kFloat)).item<double>());
  const double rel_l2 = ref_l2 > 0 ? l2 / ref_l2 : 0.0;
  std::printf("MXFP8 linear vs dequantized reference: rel_l2=%.6g max_abs=%.6g\n",
              rel_l2,
              diff.max().item<double>());
  TORCH_CHECK(rel_l2 < 1e-3, "MXFP8 linear GEMM reference mismatch: ", rel_l2);
  TORCH_CHECK(diff.max().item<double>() < 0.25, "MXFP8 linear max error too large");

  auto bias = (at::randn({192}, options) * 0.05).to(at::kBFloat16);
  TORCH_CHECK(at::equal(linear.forward(input, bias), output + bias),
              "MXFP8 bias must be added after model-dtype GEMM rounding");

  auto rowwise = infinity::fp8_quantize_rows(input);
  bool rejected = false;
  try {
    (void)linear.forward_quantized(rowwise.first, rowwise.second);
  } catch (const c10::Error& error) {
    rejected = std::string(error.what_without_backtrace()).find("byte UE8M0") != std::string::npos;
  }
  TORCH_CHECK(rejected, "MXFP8 method accepted per-row FP32 RMS scales");
  auto cpu_scales = at::zeros({infinity::mxfp8_scale_elements(qi.rows, qi.cols)},
                              at::TensorOptions().dtype(at::kByte));
  bool cpu_scales_rejected = false;
  try {
    (void)linear.forward_quantized(qi.values, cpu_scales);
  } catch (const c10::Error& error) {
    cpu_scales_rejected =
        std::string(error.what_without_backtrace()).find("device") != std::string::npos;
  }
  TORCH_CHECK(cpu_scales_rejected, "MXFP8 accepted CPU block scales for CUDA values");

  auto gate_up = (at::randn({257, 256}, options) * 0.5).to(at::kBFloat16);
  auto gated = infinity::mxfp8_gated_silu_quantize(gate_up);
  auto gated_ref = gated_activation_reference(gate_up).to(at::kCPU);
  auto gated_deq = cpu_dequantize(gated);
  auto gated_diff = (gated_deq - gated_ref.to(at::kFloat)).abs();
  const double gated_l2 = std::sqrt(at::sum(gated_diff * gated_diff).item<double>());
  const double gated_ref_l2 =
      std::sqrt(at::sum(gated_ref.to(at::kFloat) * gated_ref.to(at::kFloat)).item<double>());
  const double gated_rel = gated_ref_l2 > 0 ? gated_l2 / gated_ref_l2 : 0.0;
  std::printf("MXFP8 fused gated quantization: rel_l2=%.6g max_abs=%.6g\n",
              gated_rel,
              gated_diff.max().item<double>());
  TORCH_CHECK(gated_rel < 0.05, "MXFP8 fused gated quantization error too large: ", gated_rel);

  for (const int64_t width : {2112, 21504}) {
    constexpr int64_t rows = 129;
    auto storage = at::randn({rows * 2 * width + 1}, options) * 2;
    auto gate_up = storage.narrow(0, 1, rows * 2 * width).view({rows, 2 * width});
    gate_up.select(0, 0).zero_();
    gate_up.select(0, 1).narrow(0, 0, kBlock).fill_(std::numeric_limits<float>::infinity());
    gate_up.select(0, 2).narrow(0, 0, kBlock).fill_(std::numeric_limits<float>::quiet_NaN());
    check_precise_gated_quantize(gate_up, width);
  }
  {
    constexpr int64_t width = 65536, rows = 6;
    auto gates =
        at::arange(width, at::TensorOptions().dtype(at::kInt)).to(at::kShort).view(at::kBFloat16);
    auto exhaustive = at::empty({rows, 2 * width}, at::TensorOptions().dtype(at::kBFloat16));
    for (int64_t row = 0; row < rows; ++row)
      exhaustive.select(0, row).narrow(0, 0, width).copy_(gates);
    exhaustive.select(0, 0).narrow(0, width, width).fill_(1.f);
    exhaustive.select(0, 1).narrow(0, width, width).fill_(-1.5f);
    exhaustive.select(0, 2).narrow(0, width, width).fill_(std::ldexp(1.f, -133));
    exhaustive.select(0, 3).narrow(0, width, width).fill_(std::ldexp(1.f, 100));
    exhaustive.select(0, 4).narrow(0, width, width).zero_();
    exhaustive.select(0, 5).narrow(0, width, width).fill_(-0.f);
    check_precise_gated_quantize(exhaustive.to(options.device()), width);
    // Homogeneous packs force zero and each guard endpoint through BF16x2.
    auto bounds = at::tensor({0.f,
                              -0.f,
                              std::ldexp(1.f, -32),
                              -std::ldexp(1.f, -32),
                              std::ldexp(1.f, 32),
                              -std::ldexp(1.f, 32),
                              std::ldexp(1.f, -133),
                              INFINITY})
                      .to(options)
                      .unsqueeze(1)
                      .expand({8, 32});
    check_precise_gated_quantize(at::cat({bounds, at::full_like(bounds, 1.5)}, 1), 32);
  }
  std::printf("MXFP8 precise GeGLU: fused bytes match gemma_gated_cuda for I=2112/21504 and all "
              "65536 BF16 gates\n");

  check_special_values();
  check_alignment_views();
  check_empty_rows();
  check_mxfp8_mlp_path();
  check_mxfp8_qualification_rejections();
  check_k_padding();
  std::printf("mxfp8_linear_test PASS\n");
  return 0;
}
