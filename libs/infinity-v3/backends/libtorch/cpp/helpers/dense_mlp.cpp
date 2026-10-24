// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/dense_mlp.h"
#include <ATen/ops/_addmm_activation.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/encoder_activation.h"
#include "kernels/fp8_fusions.h"
#endif

namespace tei {
at::Tensor mlp_activation(const at::Tensor& x, MlpActivation kind) {
  switch (kind) {
  case MlpActivation::Identity:
    return x;
  case MlpActivation::Relu:
    return at::relu(x);
  case MlpActivation::Silu:
    return at::silu(x);
  case MlpActivation::Gelu:
    return at::gelu(x, "none");
  case MlpActivation::GeluTanh:
    return at::gelu(x, "tanh");
  case MlpActivation::Tanh:
    return at::tanh(x);
  case MlpActivation::RoundedSilu:
    return x / (at::ones({}, x.options()) + at::exp(-x));
  case MlpActivation::RoundedGelu: {
#ifdef TEI_TORCH_CUDA_KERNELS
    if (x.is_cuda())
      return encoder_gelu(x);
#endif
    auto cube = (x * x) * x;
    auto alpha = x + at::scalar_tensor(.044715, x.options()) * cube;
    auto tangent = at::tanh(at::scalar_tensor(.7978845608028654, x.options()) * alpha);
    return (at::scalar_tensor(.5, x.options()) * x) * (at::ones({}, x.options()) + tangent);
  }
  }
  TORCH_CHECK(false, "Unsupported MLP activation");
}

MlpProjection::MlpProjection(at::Tensor weight,
                             at::Tensor bias,
                             ProjectionRecipe recipe,
                             std::string name)
    : weight_(std::move(weight)), bias_(std::move(bias)), recipe_(recipe), name_(std::move(name)) {
  TORCH_CHECK(weight_.dim() == 2 && weight_.size(0) > 0 && weight_.size(1) > 0,
              "MLP projection requires a nonempty matrix: ",
              name_);
  TORCH_CHECK(!bias_.defined() || (bias_.dim() == 1 && bias_.size(0) == weight_.size(0) &&
                                   bias_.device() == weight_.device() &&
                                   bias_.scalar_type() == weight_.scalar_type()),
              "Invalid MLP projection bias: ",
              name_);
  for (const auto& bounds : {recipe_.input_clip, recipe_.output_clip})
    TORCH_CHECK(!bounds || (*bounds)[0] <= (*bounds)[1], "Invalid MLP clipping bounds: ", name_);
  dtype_ = weight_.scalar_type();
  input_width_ = weight_.size(1);
  output_width_ = weight_.size(0);
}

void MlpProjection::prepare_fp8() {
  if (quantized_)
    return;
  auto quantized = std::make_shared<Fp8Linear>(weight_);
  quantized_ = std::move(quantized);
  weight_ = at::Tensor();
}

at::Tensor MlpProjection::execute(const at::Tensor& input,
                                  const MlpContext& context,
                                  const at::Tensor& scales,
                                  MlpEpilogue epilogue,
                                  bool omit_bias) const {
  TORCH_CHECK(input_width_ && input.dim() > 0 && input.size(-1) == input_width_,
              "Invalid MLP projection input: ",
              name_);
  TORCH_CHECK(!scales.defined() || (fp8() && !recipe_.input_clip && !recipe_.cast_input),
              "Prequantized MLP input bypasses a projection transform: ",
              name_);
  auto x = input;
  if (recipe_.input_clip)
    x = x.clamp((*recipe_.input_clip)[0], (*recipe_.input_clip)[1]);
  if (recipe_.cast_input)
    x = x.to(dtype_);
  at::Tensor output;
  if (quantized_) {
    auto rows = x.dim() == 2 ? x : x.reshape({x.numel() / input_width_, input_width_});
    auto bias = omit_bias || recipe_.separate_bias ? at::Tensor() : bias_;
    output = scales.defined() ? quantized_->forward_quantized(rows, scales, bias)
                              : quantized_->forward(rows, bias);
    if (x.dim() != 2) {
      auto shape = x.sizes().vec();
      shape.back() = output_width_;
      output = output.reshape(shape);
    }
  } else {
    TORCH_CHECK(!omit_bias, "Unquantized MLP bias cannot be omitted");
    if (epilogue != MlpEpilogue::None) {
      if (context.activated_gemm)
        output = context.activated_gemm(*this, x, epilogue);
    } else if (context.gemm)
      output = context.gemm(*this, x);
    if (!output.defined()) {
      output =
          epilogue == MlpEpilogue::None
              ? at::linear(x, weight_, gemm_bias())
              : at::_addmm_activation(bias_, x, weight_.t(), 1, 1, epilogue == MlpEpilogue::Gelu);
    }
  }
  if (recipe_.separate_bias && bias_.defined() && !omit_bias)
    output = output + bias_;
  if (recipe_.output_clip)
    output = output.clamp((*recipe_.output_clip)[0], (*recipe_.output_clip)[1]);
  return output;
}

at::Tensor MlpProjection::forward(const at::Tensor& x, const MlpContext& context) const {
  return execute(x, context, {});
}

MlpActivationRecipe MlpActivationRecipe::gated(MlpActivation kind, bool gate_first) {
  MlpActivationRecipe recipe(kind);
  recipe.packed = true;
  recipe.gate_first = gate_first;
  return recipe;
}

MlpActivationRecipe
MlpActivationRecipe::numerical(std::function<at::Tensor(const at::Tensor&)> apply, bool packed) {
  TORCH_CHECK(bool(apply), "MLP numerical recipe cannot be empty");
  MlpActivationRecipe recipe;
  recipe.custom = std::move(apply);
  recipe.packed = packed;
  return recipe;
}

at::Tensor MlpActivationRecipe::apply(const at::Tensor& x, bool fp8) const {
  if (fp8 && quantized_custom)
    return quantized_custom(x);
  if (custom)
    return custom(x);
  if (!packed)
    return mlp_activation(x, kind);
  auto width = x.size(-1) / 2;
  auto first = x.narrow(-1, 0, width), second = x.narrow(-1, width, width);
  return gate_first ? mlp_activation(first, kind) * second : first * mlp_activation(second, kind);
}

DenseMlp::DenseMlp(MlpProjection input, MlpProjection output, MlpActivationRecipe activation)
    : input_(std::move(input)), output_(std::move(output)), activation_(std::move(activation)) {
  TORCH_CHECK(input_.output_width() == output_.input_width() * (activation_.packed ? 2 : 1),
              "MLP projection widths do not match activation layout");
  input_.role_ = MlpProjectionRole::Input;
  output_.role_ = MlpProjectionRole::Output;
  TORCH_CHECK(!activation_.packed || activation_.epilogue == MlpEpilogue::None,
              "Packed gated MLP cannot use a unary GEMM epilogue");
}

DenseMlp::DenseMlp(MlpProjection gate,
                   MlpProjection up,
                   MlpProjection output,
                   MlpActivationRecipe activation)
    : DenseMlp(std::move(gate), std::move(output), std::move(activation)) {
  TORCH_CHECK(!activation_.packed && up.input_width() == input_.input_width() &&
                  up.output_width() == input_.output_width(),
              "Invalid separate gate/up MLP");
  up.role_ = MlpProjectionRole::Up;
  up_ = std::move(up);
}

void DenseMlp::prepare(MlpPrecision precision) {
  TORCH_CHECK(precision == MlpPrecision::Fp8 || !fp8(),
              "Cannot restore released model-dtype MLP weights");
  if (precision == MlpPrecision::Model)
    return;
  auto input = input_, output = output_;
  auto up = up_;
  input.prepare_fp8();
  output.prepare_fp8();
  if (up)
    up->prepare_fp8();
  input_ = std::move(input);
  output_ = std::move(output);
  up_ = std::move(up);
}

at::Tensor DenseMlp::forward(const at::Tensor& x,
                             const MlpContext& context,
                             const at::Tensor& input_scales) const {
  TORCH_CHECK(!input_scales.defined() || fp8(), "Quantized MLP input requires FP8 preparation");
  auto epilogue = !fp8() && x.is_cuda() && x.dim() == 2 && input_.bias_.defined() &&
                          !input_.recipe_.separate_bias && !input_.recipe_.output_clip
                      ? activation_.epilogue
                      : MlpEpilogue::None;
#ifdef TEI_TORCH_CUDA_KERNELS
  if (fp8() && activation_.fusion == MlpFusion::RoundedBiasGelu && x.is_cuda() &&
      input_.bias_.defined() && !input_.recipe_.output_clip && !activation_.packed && !up_) {
    auto middle = input_.execute(x, context, input_scales, MlpEpilogue::None, true);
    return output_.forward(encoder_bias_gelu(middle, input_.bias_), context);
  }
#endif
  auto middle = input_.execute(x, context, input_scales, epilogue);
#ifdef TEI_TORCH_CUDA_KERNELS
  if (fp8() && activation_.fusion == MlpFusion::RoundedSwiGlu && activation_.packed && !up_ &&
      !output_.recipe_.input_clip && !output_.recipe_.cast_input && middle.is_cuda() &&
      middle.dim() == 2 && middle.is_contiguous() && middle.size(1) / 2 <= 32768) {
    auto quantized = fp8_swiglu_quantize(middle);
    return output_.execute(quantized.first, context, quantized.second);
  }
#endif
  auto activated = epilogue == MlpEpilogue::None ? activation_.apply(middle, fp8()) : middle;
  if (up_)
    activated = activated * up_->execute(x, context, input_scales);
  return output_.forward(activated, context);
}
}
