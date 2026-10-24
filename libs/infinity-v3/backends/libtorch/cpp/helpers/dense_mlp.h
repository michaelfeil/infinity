// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include "helpers/fp8_linear.h"
#include <array>
#include <functional>
#include <optional>
#include <string>

namespace tei {
enum class MlpPrecision { Model, Fp8 };
enum class MlpActivation { Identity, Relu, Silu, Gelu, GeluTanh, Tanh, RoundedSilu, RoundedGelu };
enum class MlpEpilogue { None, Relu, Gelu };
enum class MlpProjectionRole { Input, Output, Up };
enum class MlpFusion { None, RoundedSwiGlu, RoundedBiasGelu };
at::Tensor mlp_activation(const at::Tensor&, MlpActivation);

struct ProjectionRecipe {
  // Separate bias preserves a model-dtype GEMM rounding before addition.
  bool separate_bias = false;
  bool cast_input = false;
  std::optional<std::array<double, 2>> input_clip, output_clip;
};

class MlpProjection;

struct MlpContext {
  // Kernel selection only: implementations consume the supplied weight/bias
  // and requested epilogue. Undefined output requests the ordinary ATen kernel.
  std::function<at::Tensor(const MlpProjection&, const at::Tensor&)> gemm;
  std::function<at::Tensor(const MlpProjection&, const at::Tensor&, MlpEpilogue)> activated_gemm;
};

class MlpProjection {
  at::Tensor weight_, bias_;
  std::shared_ptr<Fp8Linear> quantized_;
  ProjectionRecipe recipe_;
  std::string name_;
  at::ScalarType dtype_ = at::kFloat;
  int64_t input_width_ = 0, output_width_ = 0;
  MlpProjectionRole role_ = MlpProjectionRole::Input;
  friend class DenseMlp;
  at::Tensor execute(const at::Tensor&,
                     const MlpContext&,
                     const at::Tensor& scales,
                     MlpEpilogue = MlpEpilogue::None,
                     bool omit_bias = false) const;
  void prepare_fp8();

public:
  MlpProjection() = default;
  MlpProjection(at::Tensor weight,
                at::Tensor bias = {},
                ProjectionRecipe recipe = {},
                std::string name = {});

  const at::Tensor& weight() const {
    return weight_;
  }

  const at::Tensor& bias() const {
    return bias_;
  }

  at::Tensor gemm_bias() const {
    return recipe_.separate_bias ? at::Tensor() : bias_;
  }

  const std::string& name() const {
    return name_;
  }

  MlpProjectionRole role() const {
    return role_;
  }

  bool fp8() const {
    return bool(quantized_);
  }

  int64_t input_width() const {
    return input_width_;
  }

  int64_t output_width() const {
    return output_width_;
  }

  at::Tensor forward(const at::Tensor& x, const MlpContext& context = {}) const;
};

// The mathematical recipe is prepared once and applied in either precision.
// Custom recipes describe numerical contracts, never projection dispatch.
struct MlpActivationRecipe {
  MlpActivation kind = MlpActivation::Identity;
  bool packed = false, gate_first = true;
  std::function<at::Tensor(const at::Tensor&)> custom, quantized_custom;
  MlpEpilogue epilogue = MlpEpilogue::None;
  MlpFusion fusion = MlpFusion::None;
  MlpActivationRecipe() = default;

  MlpActivationRecipe(MlpActivation activation) : kind(activation) {}

  static MlpActivationRecipe gated(MlpActivation kind, bool gate_first = true);
  static MlpActivationRecipe numerical(std::function<at::Tensor(const at::Tensor&)> apply,
                                       bool packed = false);
  at::Tensor apply(const at::Tensor&, bool fp8) const;
};

class DenseMlp {
  MlpProjection input_, output_;
  std::optional<MlpProjection> up_;
  MlpActivationRecipe activation_;

public:
  DenseMlp() = default;
  DenseMlp(MlpProjection input, MlpProjection output, MlpActivationRecipe activation);
  DenseMlp(MlpProjection gate,
           MlpProjection up,
           MlpProjection output,
           MlpActivationRecipe activation);
  void prepare(MlpPrecision precision);

  bool fp8() const {
    return input_.fp8();
  }

  const MlpProjection& input() const {
    return input_;
  }

  const MlpProjection& output() const {
    return output_;
  }

  const std::optional<MlpProjection>& up() const {
    return up_;
  }

  at::Tensor forward(const at::Tensor& input,
                     const MlpContext& context = {},
                     const at::Tensor& input_scales = {}) const;
};
}
