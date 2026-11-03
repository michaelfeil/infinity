// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "helpers/dense_mlp.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#endif
#include <c10/core/InferenceMode.h>
#include <iostream>

namespace {
void check_equal(const at::Tensor& actual, const at::Tensor& expected) {
  TORCH_CHECK(actual.sizes() == expected.sizes() &&
                  actual.scalar_type() == expected.scalar_type() && at::equal(actual, expected),
              "MLP changed shape, dtype or arithmetic");
}

// A loader can install another execution method without editing the MLP.
class TestLinearMethod final : public infinity::MlpLinearMethod {
  at::Tensor weight_;
  std::shared_ptr<int> calls_;

public:
  TestLinearMethod(at::Tensor weight, std::shared_ptr<int> calls)
      : weight_(std::move(weight)), calls_(std::move(calls)) {}

  infinity::MlpPrecision precision() const override {
    return infinity::MlpPrecision::Model;
  }

  const at::Tensor& weight() const override {
    return weight_;
  }

  at::Tensor apply(const infinity::MlpProjection&,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const infinity::MlpContext&,
                   infinity::MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(epilogue == infinity::MlpEpilogue::None && !scales.defined(),
                "MLP did not respect method capabilities");
    ++*calls_;
    return at::linear(x, weight_, bias);
  }
};

void check(const c10::Device& device, at::ScalarType dtype) {
  const auto opts = at::TensorOptions().device(device).dtype(dtype);
  auto wi = at::randn({64, 32}, opts) * .03, wo = at::randn({16, 64}, opts) * .03;
  auto bi = at::randn({64}, opts) * .1, bo = at::randn({16}, opts) * .1;
  for (auto kind : {infinity::MlpActivation::Identity,
                    infinity::MlpActivation::Relu,
                    infinity::MlpActivation::Silu,
                    infinity::MlpActivation::Gelu,
                    infinity::MlpActivation::GeluTanh,
                    infinity::MlpActivation::Tanh}) {
    infinity::DenseMlp mlp({wi, bi}, {wo, bo}, {kind});
    for (auto precision : {infinity::MlpPrecision::Model, infinity::MlpPrecision::Fp8}) {
      if (precision == infinity::MlpPrecision::Fp8 && !device.is_cuda())
        continue;
      mlp.prepare(precision);
      mlp.prepare(precision); // Preparation is idempotent, including released FP8 weights.
      for (const auto& shape : {std::vector<int64_t>{7, 32}, {2, 7, 32}, {0, 32}, {32}}) {
        auto doubled = shape;
        doubled.back() *= 2;
        auto x = at::randn(doubled, opts).slice(-1, 0, doubled.back(), 2);
        auto rows = x.reshape({x.numel() / 32, 32});
        auto projected = at::linear(x, wi, bi);
        if (mlp.fp8())
          projected = infinity::Fp8Linear(wi).forward(rows, bi);
        at::Tensor activated;
        switch (kind) {
        case infinity::MlpActivation::Identity:
          activated = projected;
          break;
        case infinity::MlpActivation::Relu:
          activated = at::relu(projected);
          break;
        case infinity::MlpActivation::Silu:
          activated = at::silu(projected);
          break;
        case infinity::MlpActivation::Gelu:
          activated = at::gelu(projected, "none");
          break;
        case infinity::MlpActivation::GeluTanh:
          activated = at::gelu(projected, "tanh");
          break;
        default:
          activated = at::tanh(projected);
          break;
        }
        auto expected = mlp.fp8() ? infinity::Fp8Linear(wo).forward(activated, bo)
                                  : at::linear(activated, wo, bo);
        auto output_shape = shape;
        output_shape.back() = 16;
        check_equal(mlp.forward(x), expected.reshape(output_shape));
      }
      TORCH_CHECK(mlp.input().bias().defined() && mlp.output().bias().defined(),
                  "Precision preparation discarded biases");
      if (mlp.fp8()) {
        TORCH_CHECK(!mlp.input().weight().defined() && !mlp.output().weight().defined(),
                    "FP8 retained original matrix storage");
        auto x = at::randn({7, 32}, opts);
        auto q = infinity::fp8_quantize_rows(x);
        check_equal(mlp.forward(q.first, {}, q.second), mlp.forward(x));
      }
    }
  }

  auto up = at::randn_like(wi) * .03, bu = at::randn_like(bi) * .1;
  infinity::DenseMlp separate({wi, bi}, {up, bu}, {wo, bo}, {infinity::MlpActivation::Silu});
  auto x = at::randn({7, 32}, opts);
  infinity::ProjectionRecipe separate_bias;
  separate_bias.separate_bias = true;
  infinity::MlpActivationRecipe rounded(infinity::MlpActivation::RoundedGelu);
  rounded.fusion = infinity::MlpFusion::RoundedBiasGelu;
  auto strided_bias = at::stack({bi, bi}, -1).select(-1, 0);
  infinity::DenseMlp rounded_mlp(
      {wi, strided_bias, separate_bias}, {wo, bo, separate_bias}, rounded);
  for (const auto& input : {x, x.unsqueeze(0)})
    check_equal(rounded_mlp.forward(input),
                at::linear(infinity::mlp_activation(at::linear(input, wi) + strided_bias,
                                                    infinity::MlpActivation::RoundedGelu),
                           wo) +
                    bo);
  check_equal(separate.forward(x),
              at::linear(at::silu(at::linear(x, wi, bi)) * at::linear(x, up, bu), wo, bo));
  if (device.is_cuda()) {
    separate.prepare(infinity::MlpPrecision::Fp8);
    auto q = infinity::fp8_quantize_rows(x);
    check_equal(separate.forward(q.first, {}, q.second), separate.forward(x));
    check_equal(separate.forward(x),
                infinity::Fp8Linear(wo).forward(at::silu(infinity::Fp8Linear(wi).forward(x, bi)) *
                                                    infinity::Fp8Linear(up).forward(x, bu),
                                                bo));
  }

  infinity::ProjectionRecipe clipped;
  clipped.separate_bias = true;
  clipped.cast_input = true;
  clipped.input_clip = std::array<double, 2>{-.4, .5};
  clipped.output_clip = std::array<double, 2>{-.1, .2};
  infinity::DenseMlp bounded({wi, bi, clipped}, {wo, bo, clipped}, {infinity::MlpActivation::Tanh});
  for (auto precision : {infinity::MlpPrecision::Model, infinity::MlpPrecision::Fp8}) {
    if (precision == infinity::MlpPrecision::Fp8 && !device.is_cuda())
      continue;
    bounded.prepare(precision);
    auto project = [&](const at::Tensor& value, const at::Tensor& weight, const at::Tensor& bias) {
      auto input = value.clamp(-.4, .5).to(dtype);
      auto output =
          bounded.fp8() ? infinity::Fp8Linear(weight).forward(input) : at::linear(input, weight);
      return (output + bias).clamp(-.1, .2);
    };
    auto input = x.to(at::kFloat);
    const auto expected = project(at::tanh(project(input, wi, bi)), wo, bo);
    int gemms = 0;
    infinity::MlpContext context;
    context.gemm = [&](const infinity::MlpProjection& projection, const at::Tensor& rows) {
      ++gemms;
      TORCH_CHECK(!projection.gemm_bias().defined(), "Separate bias escaped into GEMM");
      TORCH_CHECK(rows.scalar_type() == dtype && rows.abs().max().item<float>() <= .501,
                  "Projection kernel observed input before cast/clipping");
      return at::linear(rows, projection.weight());
    };
    check_equal(bounded.forward(input, context), expected);
    TORCH_CHECK(gemms == (bounded.fp8() ? 0 : 2), "Unexpected precision backend selection");
  }
  infinity::DenseMlp owned({wi, bi, clipped}, {wo, bo, clipped}, {infinity::MlpActivation::Tanh});
  auto calls = std::make_shared<int>(0);
  const auto expected_owned = owned.forward(x.to(at::kFloat));
  owned.prepare([&](const infinity::MlpProjection& projection) {
    return std::make_shared<TestLinearMethod>(projection.weight(), calls);
  });
  check_equal(owned.forward(x.to(at::kFloat)), expected_owned);
  TORCH_CHECK(*calls == 2, "MLP bypassed the selected projection method");
  auto copied = owned;
  owned = {}; // The copied MLP retains immutable method state and weights.
  check_equal(copied.forward(x.to(at::kFloat)), expected_owned);
  int prepared = 0;
  bool failed = false;
  try {
    copied.prepare([&](const infinity::MlpProjection& projection)
                       -> std::shared_ptr<const infinity::MlpLinearMethod> {
      TORCH_CHECK(++prepared < 2, "Synthetic output preparation failure");
      return std::make_shared<TestLinearMethod>(projection.weight(), std::make_shared<int>(0));
    });
  } catch (const c10::Error&) {
    failed = true;
  }
  auto before = *calls;
  check_equal(copied.forward(x.to(at::kFloat)), expected_owned);
  TORCH_CHECK(failed && *calls == before + 2, "Failed preparation replaced a live method");
#ifdef TEI_TORCH_CUDA_KERNELS
  if (device.is_cuda()) {
    // Explicit composition retains the projection's model-dtype rounding, bias position,
    // activation recipe and output transforms for comparison with the fused candidate.
    auto project = [&](const at::Tensor& value,
                       const at::Tensor& weight,
                       const at::Tensor& bias,
                       const infinity::ProjectionRecipe& projection) {
      auto input = value;
      if (projection.input_clip)
        input = input.clamp((*projection.input_clip)[0], (*projection.input_clip)[1]);
      if (projection.cast_input)
        input = input.to(dtype);
      auto output = projection.separate_bias ? infinity::Fp8Linear(weight).forward(input)
                                             : infinity::Fp8Linear(weight).forward(input, bias);
      if (projection.separate_bias && bias.defined())
        output = output + bias;
      if (projection.output_clip)
        output = output.clamp((*projection.output_clip)[0], (*projection.output_clip)[1]);
      return output;
    };
    auto check_activation_recipe = [&](const infinity::MlpActivationRecipe& activation,
                                       infinity::MlpActivation expected_activation) {
      for (bool use_bias : {false, true})
        for (bool separate_bias : {false, true}) {
          infinity::ProjectionRecipe input_recipe, output_recipe;
          input_recipe.separate_bias = separate_bias;
          output_recipe.separate_bias = separate_bias;
          auto input_bias = use_bias ? bi : at::Tensor();
          auto output_bias = use_bias ? bo : at::Tensor();
          infinity::DenseMlp mlp(
              {wi, input_bias, input_recipe}, {wo, output_bias, output_recipe}, activation);
          mlp.prepare(infinity::MlpPrecision::Fp8);
          auto projected = project(x, wi, input_bias, input_recipe);
          auto activated = infinity::mlp_activation(projected, expected_activation);
          auto expected = project(activated, wo, output_bias, output_recipe);
          check_equal(mlp.forward(x), expected);
          auto quantized = infinity::fp8_quantize_rows(x);
          check_equal(mlp.forward(quantized.first, {}, quantized.second), expected);
        }
    };
    check_activation_recipe(infinity::MlpActivationRecipe(infinity::MlpActivation::Relu),
                            infinity::MlpActivation::Relu);
    {
      infinity::MlpActivationRecipe activation(infinity::MlpActivation::RoundedSilu);
      activation.fp8_activation = infinity::Fp8Activation::RoundedSilu;
      check_activation_recipe(activation, infinity::MlpActivation::RoundedSilu);
    }
    {
      infinity::MlpActivationRecipe activation(infinity::MlpActivation::RoundedGelu);
      activation.fp8_activation = infinity::Fp8Activation::RoundedGelu;
      check_activation_recipe(activation, infinity::MlpActivation::RoundedGelu);
    }
    {
      infinity::MlpActivationRecipe activation(infinity::MlpActivation::RoundedGelu);
      activation.fusion = infinity::MlpFusion::RoundedBiasGelu;
      check_activation_recipe(activation, infinity::MlpActivation::RoundedGelu);
    }
    // A custom activation remains authoritative unless a rounded kernel is explicitly selected.
    for (bool quantized_callback : {false, true}) {
      int calls = 0;
      auto callback = [&](const at::Tensor& rows) {
        ++calls;
        return at::relu(rows) + at::ones({}, rows.options());
      };
      infinity::MlpActivationRecipe activation(infinity::MlpActivation::Relu);
      if (quantized_callback)
        activation.quantized_custom = callback;
      else
        activation.custom = callback;
      infinity::DenseMlp mlp({wi, bi}, {wo, bo}, activation);
      mlp.prepare(infinity::MlpPrecision::Fp8);
      auto expected = infinity::Fp8Linear(wo).forward(
          at::relu(infinity::Fp8Linear(wi).forward(x, bi)) + at::ones({}, opts), bo);
      check_equal(mlp.forward(x), expected);
      TORCH_CHECK(calls == 1, "Custom FP8 activation callback was bypassed");
    }
    auto check_guarded = [&](const at::Tensor& values,
                             infinity::ProjectionRecipe input_recipe,
                             infinity::ProjectionRecipe output_recipe) {
      infinity::MlpActivationRecipe activation(infinity::MlpActivation::RoundedSilu);
      activation.fp8_activation = infinity::Fp8Activation::RoundedSilu;
      infinity::DenseMlp mlp({wi, bi, input_recipe}, {wo, bo, output_recipe}, activation);
      mlp.prepare(infinity::MlpPrecision::Fp8);
      auto projected = project(values, wi, bi, input_recipe);
      auto activated = infinity::mlp_activation(projected, infinity::MlpActivation::RoundedSilu);
      check_equal(mlp.forward(values), project(activated, wo, bo, output_recipe));
    };
    {
      infinity::ProjectionRecipe input_recipe;
      input_recipe.separate_bias = true;
      input_recipe.input_clip = std::array<double, 2>{-.4, .5};
      check_guarded(x, input_recipe, {});
      input_recipe = {};
      input_recipe.separate_bias = true;
      input_recipe.cast_input = true;
      check_guarded(x.to(at::kFloat), input_recipe, {});
      input_recipe = {};
      input_recipe.separate_bias = true;
      input_recipe.output_clip = std::array<double, 2>{-.1, .2};
      check_guarded(x, input_recipe, {});
      infinity::ProjectionRecipe output_recipe;
      output_recipe.separate_bias = true;
      output_recipe.input_clip = std::array<double, 2>{-.2, .3};
      check_guarded(x, {}, output_recipe);
      output_recipe = {};
      output_recipe.separate_bias = true;
      output_recipe.cast_input = true;
      check_guarded(x, {}, output_recipe);
      output_recipe = {};
      output_recipe.separate_bias = true;
      output_recipe.output_clip = std::array<double, 2>{-.05, .1};
      check_guarded(x, {}, output_recipe);
    }
    auto packed = at::cat({wi, up}, 0);
    auto recipe = infinity::MlpActivationRecipe::numerical(
        [](const at::Tensor& rows) { return infinity::fused_gated_activation(rows, 5); }, true);
    recipe.fusion = infinity::MlpFusion::RoundedSwiGlu;
    infinity::DenseMlp mlp({packed}, {wo, bo}, recipe);
    mlp.prepare(infinity::MlpPrecision::Fp8);
    check_equal(
        mlp.forward(x),
        infinity::Fp8Linear(wo).forward(
            infinity::fused_gated_activation(infinity::Fp8Linear(packed).forward(x), 5), bo));
    for (int64_t rows : {0, 1, 17, 65}) {
      auto values = at::randn({rows, 32}, opts);
      auto quantized = infinity::fp8_quantize_rows(values);
      auto expected = infinity::Fp8Linear(wo).forward(
          infinity::fused_gated_activation(infinity::Fp8Linear(packed).forward(values), 5), bo);
      check_equal(mlp.forward(values), expected);
      check_equal(mlp.forward(quantized.first, {}, quantized.second), expected);
    }
    // Input bias and clipping must retain their positions around model-dtype rounding.
    // These recipes cannot use the fused projection boundary.
    for (int transform : {0, 1, 2}) {
      infinity::ProjectionRecipe projection;
      auto bias = transform == 0 ? at::cat({bi, bu}) : at::Tensor();
      if (transform == 1)
        projection.input_clip = std::array<double, 2>{-.1, .2};
      if (transform == 2)
        projection.output_clip = std::array<double, 2>{-.01, .02};
      infinity::DenseMlp guarded({packed, bias, projection}, {wo, bo}, recipe);
      guarded.prepare(infinity::MlpPrecision::Fp8);
      auto values = transform == 1 ? x.clamp(-.1, .2) : x;
      auto middle = infinity::Fp8Linear(packed).forward(values, bias);
      if (transform == 2)
        middle = middle.clamp(-.01, .02);
      check_equal(guarded.forward(x),
                  infinity::Fp8Linear(wo).forward(infinity::fused_gated_activation(middle, 5), bo));
    }
    auto packed_bias = at::cat({bi, bu});
    auto gated_code = [](infinity::Fp8GatedActivation kind) {
      return kind == infinity::Fp8GatedActivation::RoundedSilu ? 5 : 4;
    };
    auto make_gated_recipe = [&](infinity::Fp8GatedActivation kind, bool gate_first) {
      const int code = gated_code(kind);
      auto recipe = infinity::MlpActivationRecipe::numerical(
          [code, gate_first](const at::Tensor& rows) {
            return infinity::fused_gated_activation(rows, code, gate_first);
          },
          true);
      recipe.gate_first = gate_first;
      recipe.fp8_gated_activation = kind;
      return recipe;
    };
    // Explicit gated markers must use the same fused projection boundary as the composed
    // model-dtype projection + native gated activation + FP8 output.
    for (auto kind :
         {infinity::Fp8GatedActivation::RoundedSilu, infinity::Fp8GatedActivation::RoundedGelu})
      for (bool gate_first : {false, true}) {
        const int code = gated_code(kind);
        infinity::DenseMlp gated_mlp({packed}, {wo, bo}, make_gated_recipe(kind, gate_first));
        gated_mlp.prepare(infinity::MlpPrecision::Fp8);
        auto composed = [&](const at::Tensor& values) {
          auto projected = infinity::Fp8Linear(packed).forward(values);
          auto activated = infinity::fused_gated_activation(projected, code, gate_first);
          return infinity::Fp8Linear(wo).forward(activated, bo);
        };
        auto quantized = infinity::fp8_quantize_rows(x);
        check_equal(gated_mlp.forward(x), composed(x));
        check_equal(gated_mlp.forward(quantized.first, {}, quantized.second), composed(x));
        auto empty_x = at::empty({0, 32}, opts);
        check_equal(gated_mlp.forward(empty_x), composed(empty_x));
        // Output bias and clipping remain outside the fused boundary but must be preserved.
        infinity::ProjectionRecipe output_recipe;
        output_recipe.output_clip = std::array<double, 2>{-.1, .2};
        infinity::DenseMlp clipped_mlp(
            {packed}, {wo, bo, output_recipe}, make_gated_recipe(kind, gate_first));
        clipped_mlp.prepare(infinity::MlpPrecision::Fp8);
        auto composed_clipped = [&](const at::Tensor& values) {
          auto projected = infinity::Fp8Linear(packed).forward(values);
          auto activated = infinity::fused_gated_activation(projected, code, gate_first);
          return infinity::Fp8Linear(wo).forward(activated, bo).clamp(-.1, .2);
        };
        check_equal(clipped_mlp.forward(x), composed_clipped(x));
        check_equal(clipped_mlp.forward(quantized.first, {}, quantized.second),
                    composed_clipped(x));
        if (kind == infinity::Fp8GatedActivation::RoundedSilu && gate_first)
          check_equal(gated_mlp.forward(x), mlp.forward(x));
      }
    // Projection transforms and separate up projections decline fusion but keep their arithmetic.
    auto check_gated_guarded = [&](infinity::Fp8GatedActivation kind,
                                   bool gate_first,
                                   const at::Tensor& values,
                                   infinity::ProjectionRecipe input_recipe,
                                   infinity::ProjectionRecipe output_recipe,
                                   bool with_input_bias) {
      const int code = gated_code(kind);
      auto bias = with_input_bias ? packed_bias : at::Tensor();
      infinity::DenseMlp gated_mlp({packed, bias, input_recipe},
                                   {wo, bo, output_recipe},
                                   make_gated_recipe(kind, gate_first));
      gated_mlp.prepare(infinity::MlpPrecision::Fp8);
      auto projected = project(values, packed, bias, input_recipe);
      auto activated = infinity::fused_gated_activation(projected, code, gate_first);
      check_equal(gated_mlp.forward(values), project(activated, wo, bo, output_recipe));
    };
    {
      check_gated_guarded(infinity::Fp8GatedActivation::RoundedGelu, false, x, {}, {}, true);
      infinity::ProjectionRecipe input_recipe;
      input_recipe.input_clip = std::array<double, 2>{-.1, .2};
      check_gated_guarded(
          infinity::Fp8GatedActivation::RoundedGelu, false, x, input_recipe, {}, false);
      input_recipe = {};
      input_recipe.cast_input = true;
      check_gated_guarded(infinity::Fp8GatedActivation::RoundedGelu,
                          false,
                          x.to(at::kFloat),
                          input_recipe,
                          {},
                          false);
      input_recipe = {};
      input_recipe.output_clip = std::array<double, 2>{-.01, .02};
      check_gated_guarded(
          infinity::Fp8GatedActivation::RoundedGelu, false, x, input_recipe, {}, false);
      infinity::ProjectionRecipe output_recipe;
      output_recipe.input_clip = std::array<double, 2>{-.2, .3};
      check_gated_guarded(
          infinity::Fp8GatedActivation::RoundedGelu, false, x, {}, output_recipe, false);
      output_recipe = {};
      output_recipe.cast_input = true;
      check_gated_guarded(
          infinity::Fp8GatedActivation::RoundedGelu, false, x, {}, output_recipe, false);
    }
    {
      auto activation = infinity::MlpActivationRecipe::numerical([](const at::Tensor& rows) {
        return infinity::mlp_activation(rows, infinity::MlpActivation::RoundedSilu);
      });
      activation.fp8_gated_activation = infinity::Fp8GatedActivation::RoundedSilu;
      infinity::DenseMlp separate_mlp({wi, bi}, {up, bu}, {wo, bo}, activation);
      separate_mlp.prepare(infinity::MlpPrecision::Fp8);
      auto expected = infinity::Fp8Linear(wo).forward(
          infinity::mlp_activation(infinity::Fp8Linear(wi).forward(x, bi),
                                   infinity::MlpActivation::RoundedSilu) *
              infinity::Fp8Linear(up).forward(x, bu),
          bo);
      check_equal(separate_mlp.forward(x), expected);
    }
    {
      int calls = 0;
      auto activation = infinity::MlpActivationRecipe::numerical(
          [&calls](const at::Tensor& rows) {
            ++calls;
            return infinity::fused_gated_activation(rows, 4, false);
          },
          true);
      activation.gate_first = false;
      infinity::DenseMlp custom_mlp({packed}, {wo, bo}, activation);
      custom_mlp.prepare(infinity::MlpPrecision::Fp8);
      auto expected = infinity::Fp8Linear(wo).forward(
          infinity::fused_gated_activation(infinity::Fp8Linear(packed).forward(x), 4, false), bo);
      check_equal(custom_mlp.forward(x), expected);
      TORCH_CHECK(calls == 1, "Unknown custom gated activation was bypassed");
    }
    {
      int calls = 0;
      auto activation = make_gated_recipe(infinity::Fp8GatedActivation::RoundedGelu, true);
      activation.custom = [&calls](const at::Tensor& rows) {
        ++calls;
        return at::relu(rows);
      };
      infinity::DenseMlp marked_mlp({packed}, {wo, bo}, activation);
      marked_mlp.prepare(infinity::MlpPrecision::Fp8);
      auto expected = infinity::Fp8Linear(wo).forward(
          infinity::fused_gated_activation(infinity::Fp8Linear(packed).forward(x), 4, true), bo);
      check_equal(marked_mlp.forward(x), expected);
      TORCH_CHECK(calls == 0, "Explicit gated marker did not bypass the fallback callback");
    }
    infinity::DenseMlp invalid({wi}, {at::zeros({17, 64}, opts)}, {infinity::MlpActivation::Relu});
    bool rejected = false;
    try {
      invalid.prepare(infinity::MlpPrecision::Fp8);
    } catch (const c10::Error&) {
      rejected = true;
    }
    TORCH_CHECK(rejected && !invalid.fp8() && invalid.input().weight().defined(),
                "Failed FP8 preparation left a partially quantized MLP");
  }
#endif
}
}

int main(int argc, char**) {
  try {
    c10::InferenceMode inference;
    at::manual_seed(846);
    const auto device = c10::Device(argc > 1 ? "cuda:0" : "cpu");
    for (auto dtype : {at::kHalf, at::kBFloat16})
      check(device, dtype);
    if (!device.is_cuda())
      check(device, at::kFloat);
    std::cout << "Persistent MLP bias, clipping, layout, preparation and precision checks passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
