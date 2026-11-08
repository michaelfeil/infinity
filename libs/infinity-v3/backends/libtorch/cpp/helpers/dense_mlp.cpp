// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/dense_mlp.h"
#include "helpers/mxfp8_linear.h"
#include "helpers/nvfp4_linear.h"
#include <ATen/ops/_addmm_activation.h>
#include "kernels/fp8_fusions.h"
#ifdef INFINITY_TORCH_CUDA_KERNELS
#include "kernels/encoder_activation.h"
#endif

namespace infinity {
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
#ifdef INFINITY_TORCH_CUDA_KERNELS
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

namespace {
class ModelLinearMethod final : public MlpLinearMethod {
  at::Tensor weight_;

public:
  explicit ModelLinearMethod(at::Tensor weight) : weight_(std::move(weight)) {}

  MlpPrecision precision() const override {
    return MlpPrecision::Model;
  }

  const at::Tensor& weight() const override {
    return weight_;
  }

  bool supports_epilogue() const override {
    return true;
  }

  at::Tensor apply(const MlpProjection& projection,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const MlpContext& context,
                   MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(!scales.defined(), "Model-dtype method cannot consume FP8 rows");
    at::Tensor output;
    if (epilogue != MlpEpilogue::None) {
      if (context.activated_gemm)
        output = context.activated_gemm(projection, x, epilogue);
    } else if (context.gemm)
      output = context.gemm(projection, x);
    if (output.defined())
      return output;
    return epilogue == MlpEpilogue::None
               ? at::linear(x, weight_, bias)
               : at::_addmm_activation(bias, x, weight_.t(), 1, 1, epilogue == MlpEpilogue::Gelu);
  }
};

class Fp8LinearMethod final : public MlpLinearMethod {
  Fp8Linear linear_;
  at::Tensor ordinary_weight_; // No ordinary matrix retained after preparation.

public:
  explicit Fp8LinearMethod(const at::Tensor& weight) : linear_(weight) {}

  MlpPrecision precision() const override {
    return MlpPrecision::Fp8;
  }

  const at::Tensor& weight() const override {
    return ordinary_weight_;
  }

  bool accepts_fp8_rows() const override {
    return true;
  }

  std::pair<at::Tensor, at::Tensor> apply_swiglu_quantized(const MlpProjection& projection,
                                                           const at::Tensor& x,
                                                           const at::Tensor& scales,
                                                           const MlpContext&) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused SwiGLU requires 2D projection rows");
    return linear_.forward_swiglu_quantized(x, scales);
  }

  std::pair<at::Tensor, at::Tensor> apply_gated_quantized(const MlpProjection& projection,
                                                          const at::Tensor& x,
                                                          const at::Tensor& scales,
                                                          const MlpContext&,
                                                          Fp8GatedActivation kind,
                                                          bool gate_first) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused gated activation requires 2D projection rows");
    return linear_.forward_gated_quantized(x, scales, kind, gate_first);
  }

  std::pair<at::Tensor, at::Tensor> apply_activation_quantized(const MlpProjection& projection,
                                                               const at::Tensor& x,
                                                               const at::Tensor& scales,
                                                               const MlpContext&,
                                                               Fp8Activation kind) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused activation requires 2D projection rows");
    return linear_.forward_activation_quantized(x, scales, projection.bias(), kind);
  }

  at::Tensor apply(const MlpProjection& projection,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const MlpContext&,
                   MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(epilogue == MlpEpilogue::None, "FP8 method has no GEMM activation epilogue");
    auto rows = x.dim() == 2
                    ? x
                    : x.reshape({x.numel() / projection.input_width(), projection.input_width()});
    auto output = scales.defined() ? linear_.forward_quantized(rows, scales, bias)
                                   : linear_.forward(rows, bias);
    if (x.dim() != 2) {
      auto shape = x.sizes().vec();
      shape.back() = projection.output_width();
      output = output.reshape(shape);
    }
    return output;
  }
};

class Mxfp8LinearMethod final : public MlpLinearMethod {
  Mxfp8Linear linear_;
  at::Tensor ordinary_weight_;

public:
  explicit Mxfp8LinearMethod(const at::Tensor& weight) : linear_(weight) {}

  MlpPrecision precision() const override {
    return MlpPrecision::Fp8;
  }

  Quantization format() const override {
    return Quantization::Mxfp8;
  }

  const at::Tensor& weight() const override {
    return ordinary_weight_;
  }

  bool accepts_fp8_rows() const override {
    return true;
  }

  bool supports_epilogue() const override {
    return false;
  }

  bool supports_rowwise_rms_input() const override {
    return false;
  }

  std::pair<at::Tensor, at::Tensor> apply_swiglu_quantized(const MlpProjection& projection,
                                                           const at::Tensor& x,
                                                           const at::Tensor& scales,
                                                           const MlpContext&) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused MXFP8 SwiGLU requires 2D projection rows");
    return linear_.forward_gated_quantized(x, scales, Fp8GatedActivation::RoundedSilu, true);
  }

  std::pair<at::Tensor, at::Tensor> apply_gated_quantized(const MlpProjection& projection,
                                                          const at::Tensor& x,
                                                          const at::Tensor& scales,
                                                          const MlpContext&,
                                                          Fp8GatedActivation kind,
                                                          bool gate_first) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused MXFP8 gated activation requires 2D projection rows");
    if (kind != Fp8GatedActivation::RoundedSilu || !gate_first)
      return {};
    return linear_.forward_gated_quantized(x, scales, kind, gate_first);
  }

  at::Tensor apply(const MlpProjection& projection,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const MlpContext&,
                   MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(epilogue == MlpEpilogue::None, "MXFP8 MLP has no GEMM activation epilogue");
    auto rows = x.dim() == 2
                    ? x
                    : x.reshape({x.numel() / projection.input_width(), projection.input_width()});
    auto output = scales.defined() ? linear_.forward_quantized(rows, scales, bias)
                                   : linear_.forward(rows, bias);
    if (x.dim() != 2) {
      auto shape = x.sizes().vec();
      shape.back() = projection.output_width();
      output = output.reshape(shape);
    }
    return output;
  }
};

// W4A4 packed E2M1 with E4M3 block scales. Native quantized rows carry both
// block and tensor scales; they use a separate contract from FP8 row scales.
class Nvfp4LinearMethod final : public MlpLinearMethod {
  Nvfp4Linear linear_;
  at::Tensor ordinary_weight_;

public:
  explicit Nvfp4LinearMethod(const at::Tensor& weight) : linear_(weight) {}

  MlpPrecision precision() const override {
    return MlpPrecision::Fp8;
  }

  Quantization format() const override {
    return Quantization::Nvfp4;
  }

  const at::Tensor& weight() const override {
    return ordinary_weight_;
  }

  bool accepts_fp8_rows() const override {
    return false;
  }

  bool supports_epilogue() const override {
    return false;
  }

  bool supports_rowwise_rms_input() const override {
    return false;
  }

  at::Tensor apply(const MlpProjection& projection,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const MlpContext&,
                   MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(!scales.defined(), "NVFP4 MLP cannot consume FP8 rows");
    TORCH_CHECK(epilogue == MlpEpilogue::None, "NVFP4 MLP has no GEMM activation epilogue");
    auto rows = x.dim() == 2
                    ? x
                    : x.reshape({x.numel() / projection.input_width(), projection.input_width()});
    auto output = linear_.forward(rows, bias);
    if (x.dim() != 2) {
      auto shape = x.sizes().vec();
      shape.back() = projection.output_width();
      output = output.reshape(shape);
    }
    return output;
  }

  bool supports_native_swiglu_quantized() const override {
    return true;
  }

  MlpNativeQuantized apply_native_swiglu_quantized(const MlpProjection& projection,
                                                   const at::Tensor& x,
                                                   const MlpContext&) const override {
    TORCH_CHECK(x.dim() == 2 && x.size(1) == projection.input_width(),
                "Fused native SwiGLU requires 2D projection rows");
    const auto quantized = linear_.forward_swiglu_quantized(x);
    return {MlpNativeQuantized::Format::Nvfp4,
            quantized.values,
            quantized.block_scales,
            quantized.global_scale,
            quantized.rows,
            quantized.cols,
            quantized.row_scales};
  }

  bool accepts_native_quantized_rows() const override {
    return true;
  }

  at::Tensor apply_native_quantized(const MlpProjection& projection,
                                    const MlpNativeQuantized& rows,
                                    const at::Tensor& bias,
                                    const MlpContext&) const override {
    TORCH_CHECK(rows.format == MlpNativeQuantized::Format::Nvfp4,
                "NVFP4 method received unsupported native rows");
    TORCH_CHECK(rows.cols == projection.input_width(),
                "NVFP4 native rows width does not match projection");
    const Nvfp4Quantized quantized{
        rows.values, rows.block_scales, rows.global_scale, rows.rows, rows.cols, rows.row_scales};
    return linear_.forward_quantized(quantized, bias);
  }
};
}

MlpProjection::MlpProjection(at::Tensor weight,
                             at::Tensor bias,
                             ProjectionRecipe recipe,
                             std::string name)
    : bias_(std::move(bias)), recipe_(recipe), name_(std::move(name)) {
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) > 0 && weight.size(1) > 0,
              "MLP projection requires a nonempty matrix: ",
              name_);
  TORCH_CHECK(!bias_.defined() || (bias_.dim() == 1 && bias_.size(0) == weight.size(0) &&
                                   bias_.device() == weight.device() &&
                                   bias_.scalar_type() == weight.scalar_type()),
              "Invalid MLP projection bias: ",
              name_);
  for (const auto& bounds : {recipe_.input_clip, recipe_.output_clip})
    TORCH_CHECK(!bounds || (*bounds)[0] <= (*bounds)[1], "Invalid MLP clipping bounds: ", name_);
  dtype_ = weight.scalar_type();
  input_width_ = weight.size(1);
  output_width_ = weight.size(0);
  method_ = std::make_shared<ModelLinearMethod>(std::move(weight));
}

void MlpProjection::prepare(const MlpMethodFactory& factory) {
  auto method = factory(*this);
  TORCH_CHECK(method, "MLP method factory returned no method: ", name_);
  method_ = std::move(method);
}

at::Tensor MlpProjection::execute(const at::Tensor& input,
                                  const MlpContext& context,
                                  const at::Tensor& scales,
                                  MlpEpilogue epilogue,
                                  bool omit_bias) const {
  TORCH_CHECK(input_width_ && input.dim() > 0 && input.size(-1) == input_width_,
              "Invalid MLP projection input: ",
              name_);
  TORCH_CHECK(!scales.defined() ||
                  (method_->accepts_fp8_rows() && !recipe_.input_clip && !recipe_.cast_input),
              "Prequantized MLP input bypasses a projection transform: ",
              name_);
  auto x = input;
  if (recipe_.input_clip)
    x = x.clamp((*recipe_.input_clip)[0], (*recipe_.input_clip)[1]);
  if (recipe_.cast_input)
    x = x.to(dtype_);
  auto bias = omit_bias || recipe_.separate_bias ? at::Tensor() : bias_;
  auto output = method_->apply(*this, x, bias, context, epilogue, scales);
  if (recipe_.separate_bias && bias_.defined() && !omit_bias)
    output = output + bias_;
  if (recipe_.output_clip)
    output = output.clamp((*recipe_.output_clip)[0], (*recipe_.output_clip)[1]);
  return output;
}

at::Tensor MlpProjection::forward(const at::Tensor& x, const MlpContext& context) const {
  return execute(x, context, {});
}

at::Tensor MlpProjection::execute_native(const MlpNativeQuantized& rows,
                                         const MlpContext& context) const {
  TORCH_CHECK(input_width_ && rows.cols == input_width_,
              "Native quantized MLP rows width does not match projection: ",
              name_);
  TORCH_CHECK(!recipe_.input_clip && !recipe_.cast_input,
              "Native quantized MLP rows bypass a projection transform: ",
              name_);
  auto bias = recipe_.separate_bias ? at::Tensor() : bias_;
  auto output = method_->apply_native_quantized(*this, rows, bias, context);
  if (recipe_.separate_bias && bias_.defined())
    output = output + bias_;
  if (recipe_.output_clip)
    output = output.clamp((*recipe_.output_clip)[0], (*recipe_.output_clip)[1]);
  return output;
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

void DenseMlp::prepare(Quantization format) {
  const bool same = input_.method().format() == format && output_.method().format() == format &&
                    (!up_ || up_->method().format() == format);
  // Prepared methods own their representation; repeated preparation is a no-op.
  if (same)
    return;
  TORCH_CHECK(input_.method().format() == Quantization::None &&
                  output_.method().format() == Quantization::None &&
                  (!up_ || up_->method().format() == Quantization::None),
              "Cannot change an already prepared MLP quantization format from ",
              quantization_name(input_.method().format()),
              " to ",
              quantization_name(format));
  switch (format) {
  case Quantization::None:
    return;
  case Quantization::Fp8:
    prepare([](const MlpProjection& projection) -> std::shared_ptr<const MlpLinearMethod> {
      return std::make_shared<Fp8LinearMethod>(projection.weight());
    });
    return;
  case Quantization::Mxfp8:
    TORCH_CHECK(supports_mxfp8(),
                "MXFP8 MLP requested but the projection geometry or provider is unsupported");
    prepare([](const MlpProjection& projection) -> std::shared_ptr<const MlpLinearMethod> {
      return std::make_shared<Mxfp8LinearMethod>(projection.weight());
    });
    return;
  case Quantization::Nvfp4:
    TORCH_CHECK(supports_nvfp4(),
                "NVFP4 MLP requested but the projection geometry or provider is unsupported");
    prepare([](const MlpProjection& projection) -> std::shared_ptr<const MlpLinearMethod> {
      return std::make_shared<Nvfp4LinearMethod>(projection.weight());
    });
    return;
  }
  TORCH_CHECK(false, "Unsupported MLP quantization format");
}

void DenseMlp::prepare(const MlpMethodFactory& factory) {
  TORCH_CHECK(bool(factory) && input_.input_width() && output_.input_width(),
              "MLP preparation requires initialized projections and a method factory");
  auto input = input_, output = output_;
  auto up = up_;
  input.prepare(factory);
  output.prepare(factory);
  if (up)
    up->prepare(factory);
  TORCH_CHECK(input.method().precision() == output.method().precision() &&
                  (!up || input.method().precision() == up->method().precision()),
              "MLP projections require the same numerical precision recipe");
  input_ = std::move(input);
  output_ = std::move(output);
  up_ = std::move(up);
}

bool DenseMlp::supports_mxfp8() const {
  if (up_ || !activation_.packed || !activation_.gate_first ||
      activation_.fusion != MlpFusion::RoundedSwiGlu || input_.recipe_.input_clip ||
      input_.recipe_.cast_input || input_.recipe_.output_clip || output_.recipe_.input_clip ||
      output_.recipe_.cast_input || output_.recipe_.output_clip || input_.bias_.defined() ||
      !input_.weight().defined() || !output_.weight().defined())
    return false;
  if (input_.output_width() % 2 != 0 || input_.output_width() / 2 > fp8_scaled_swiglu_max_width)
    return false;
  return mxfp8_mlp_supported(input_.weight(), output_.weight());
}

bool DenseMlp::supports_nvfp4() const {
  if (up_ || !activation_.packed || !activation_.gate_first ||
      activation_.fusion != MlpFusion::RoundedSwiGlu || activation_.quantized_custom ||
      input_.recipe_.input_clip || input_.recipe_.cast_input || input_.recipe_.output_clip ||
      output_.recipe_.input_clip || output_.recipe_.cast_input || output_.recipe_.output_clip ||
      input_.bias_.defined() || !input_.weight().defined() || !output_.weight().defined())
    return false;
  return nvfp4_mlp_supported(input_.weight(), output_.weight());
}

at::Tensor DenseMlp::forward(const at::Tensor& x,
                             const MlpContext& context,
                             const at::Tensor& input_scales) const {
  TORCH_CHECK(!input_scales.defined() || input_.method().accepts_fp8_rows(),
              "Quantized MLP input requires FP8 preparation");
  auto epilogue = input_.method().supports_epilogue() && x.is_cuda() && x.dim() == 2 &&
                          input_.bias_.defined() && !input_.recipe_.separate_bias &&
                          !input_.recipe_.output_clip
                      ? activation_.epilogue
                      : MlpEpilogue::None;
#ifdef INFINITY_TORCH_CUDA_KERNELS
  auto fp8_activation = activation_.fp8_activation;
  if (!fp8_activation && activation_.fusion == MlpFusion::RoundedBiasGelu)
    fp8_activation = Fp8Activation::RoundedGelu;
  if (!fp8_activation && !activation_.custom && !activation_.quantized_custom &&
      activation_.kind == MlpActivation::Relu)
    fp8_activation = Fp8Activation::Relu;
  if (fp8_activation && input_.fp8() && input_.output_width() <= 65536 &&
      output_.method().accepts_fp8_rows() && !activation_.packed && !up_ &&
      !input_.recipe_.input_clip && !input_.recipe_.cast_input && !input_.recipe_.output_clip &&
      !output_.recipe_.input_clip && !output_.recipe_.cast_input && x.is_cuda() && x.dim() == 2 &&
      x.size(1) == input_.input_width()) {
    auto fused = input_.method().apply_activation_quantized(
        input_, x, input_scales, context, *fp8_activation);
    if (fused.first.defined() && fused.second.defined())
      return output_.execute(fused.first, context, fused.second);
  }
  if ((fp8() || input_.recipe_.separate_bias) && activation_.fusion == MlpFusion::RoundedBiasGelu &&
      x.is_cuda() && x.dim() == 2 && input_.bias_.defined() && !input_.recipe_.output_clip &&
      !activation_.packed && !up_) {
    auto middle = input_.execute(x, context, input_scales, MlpEpilogue::None, true);
    return output_.forward(encoder_bias_gelu(middle.contiguous(), input_.bias_.contiguous()),
                           context);
  }
#endif
#ifdef INFINITY_TORCH_CUDA_KERNELS
  // One eligibility rule for both fused forms below: the input projection must be able to produce
  // the packed activation rows and the output projection must consume FP8 rows without transforms.
  const bool rounded_swiglu =
      input_.method().accepts_fp8_rows() && output_.method().accepts_fp8_rows() &&
      activation_.fusion == MlpFusion::RoundedSwiGlu && activation_.packed && !up_ &&
      !output_.recipe_.input_clip && !output_.recipe_.cast_input;
  // Explicit gated recipes cover the native rounded GELU form and reversed gate/up layouts while
  // the legacy RoundedSwiGlu tag above keeps using its original hook.
  const bool explicit_gated = activation_.fp8_gated_activation.has_value() && input_.fp8() &&
                              input_.method().accepts_fp8_rows() &&
                              output_.method().accepts_fp8_rows() && activation_.packed && !up_ &&
                              !input_.bias().defined() && !input_.recipe_.input_clip &&
                              !input_.recipe_.cast_input && !input_.recipe_.output_clip &&
                              !output_.recipe_.input_clip && !output_.recipe_.cast_input &&
                              x.is_cuda() && x.dim() == 2 && x.size(1) == input_.input_width();
  bool explicit_gated_hook_declined = false;
  // The fused hooks keep the input projection's exact GEMM and scales. They are used only when the
  // input projection needs no bias, cast, or clipping that the fused boundary cannot express.
  if (rounded_swiglu && input_.fp8() && !input_.bias().defined() && !input_.recipe_.input_clip &&
      !input_.recipe_.cast_input && !input_.recipe_.output_clip && x.is_cuda() && x.dim() == 2 &&
      x.size(1) == input_.input_width() &&
      input_.output_width() / 2 <= fp8_scaled_swiglu_max_width) {
    auto fused = input_.method().apply_swiglu_quantized(input_, x, input_scales, context);
    if (fused.first.defined() && fused.second.defined())
      return output_.execute(fused.first, context, fused.second);
  }
  if (explicit_gated && input_.output_width() / 2 <= fp8_scaled_swiglu_max_width) {
    auto fused = input_.method().apply_gated_quantized(input_,
                                                       x,
                                                       input_scales,
                                                       context,
                                                       *activation_.fp8_gated_activation,
                                                       activation_.gate_first);
    if (fused.first.defined() && fused.second.defined())
      return output_.execute(fused.first, context, fused.second);
    explicit_gated_hook_declined = true;
  }
#endif
#ifdef INFINITY_TORCH_CUDA_KERNELS
  // Native block-scaled hook: the format method owns the whole input
  // projection, activation arithmetic and scale metadata, so this does not
  // reuse the FP8-row contract. The fusion is limited to the shared
  // rounded-SwiGLU geometry and projections whose numerical transforms the
  // fused boundary can express exactly.
  const bool native_swiglu =
      activation_.fusion == MlpFusion::RoundedSwiGlu && activation_.packed &&
      activation_.gate_first && !up_ && !activation_.quantized_custom &&
      input_.method().supports_native_swiglu_quantized() &&
      output_.method().accepts_native_quantized_rows() && !input_.bias_.defined() &&
      !input_scales.defined() && !input_.recipe_.input_clip && !input_.recipe_.cast_input &&
      !input_.recipe_.output_clip && !output_.recipe_.input_clip && !output_.recipe_.cast_input &&
      x.is_cuda() && x.dim() == 2 && x.size(1) == input_.input_width() &&
      input_.output_width() % 2 == 0 && (input_.output_width() / 2) % 16 == 0;
  if (native_swiglu) {
    auto quantized = input_.method().apply_native_swiglu_quantized(input_, x, context);
    if (quantized.defined())
      return output_.execute_native(quantized, context);
  }
#endif
  auto middle = input_.execute(x, context, input_scales, epilogue);
#ifdef INFINITY_TORCH_CUDA_KERNELS
  if (rounded_swiglu && middle.is_cuda() && middle.dim() == 2 && middle.is_contiguous() &&
      middle.size(1) / 2 <= 32768) {
    auto quantized = fp8_swiglu_quantize(middle);
    return output_.execute(quantized.first, context, quantized.second);
  }
  if (explicit_gated && !explicit_gated_hook_declined && middle.is_cuda() && middle.dim() == 2 &&
      middle.is_contiguous() && middle.size(1) / 2 <= 32768) {
    auto quantized =
        fp8_gated_quantize(middle, *activation_.fp8_gated_activation, activation_.gate_first);
    return output_.execute(quantized.first, context, quantized.second);
  }
#endif
  auto activated = epilogue == MlpEpilogue::None ? activation_.apply(middle, fp8()) : middle;
  if (up_)
    activated = activated * up_->execute(x, context, input_scales);
  return output_.forward(activated, context);
}

bool mxfp8_mlp_supported(const at::Tensor& gate_up, const at::Tensor& down) {
  if (!gate_up.is_cuda() || !down.is_cuda() || gate_up.device() != down.device() ||
      gate_up.dim() != 2 || down.dim() != 2 || gate_up.scalar_type() != at::kBFloat16 ||
      down.scalar_type() != at::kBFloat16 || gate_up.size(0) <= 0 || down.size(0) <= 0 ||
      gate_up.size(1) % 32 != 0 || down.size(1) % 32 != 0 || gate_up.size(0) % 64 != 0 ||
      down.size(1) != gate_up.size(0) / 2)
    return false;
  const auto device = gate_up.device();
  if (!mxfp8_provider_supported(device))
    return false;
  return mxfp8_geometry_supported(device, 1, gate_up.size(0), gate_up.size(1)) &&
         mxfp8_geometry_supported(device, 128, gate_up.size(0), gate_up.size(1)) &&
         mxfp8_geometry_supported(device, 1, down.size(0), down.size(1)) &&
         mxfp8_geometry_supported(device, 128, down.size(0), down.size(1));
}

bool nvfp4_mlp_supported(const at::Tensor& gate_up, const at::Tensor& down) {
  if (!gate_up.is_cuda() || !down.is_cuda() || gate_up.device() != down.device() ||
      gate_up.dim() != 2 || down.dim() != 2 || gate_up.scalar_type() != at::kBFloat16 ||
      down.scalar_type() != at::kBFloat16 || gate_up.size(0) <= 0 || down.size(0) <= 0 ||
      gate_up.size(1) % 32 != 0 || down.size(1) % 32 != 0 || down.size(1) * 2 != gate_up.size(0))
    return false;
  const auto device = gate_up.device();
  if (!nvfp4_provider_supported(device))
    return false;
  return nvfp4_geometry_supported(device, 1, gate_up.size(0), gate_up.size(1)) &&
         nvfp4_geometry_supported(device, 128, gate_up.size(0), gate_up.size(1)) &&
         nvfp4_geometry_supported(device, 1, down.size(0), down.size(1)) &&
         nvfp4_geometry_supported(device, 128, down.size(0), down.size(1));
}

}
