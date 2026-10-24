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
class TestLinearMethod final : public tei::MlpLinearMethod {
  at::Tensor weight_;
  std::shared_ptr<int> calls_;

public:
  TestLinearMethod(at::Tensor weight, std::shared_ptr<int> calls)
      : weight_(std::move(weight)), calls_(std::move(calls)) {}

  tei::MlpPrecision precision() const override {
    return tei::MlpPrecision::Model;
  }

  const at::Tensor& weight() const override {
    return weight_;
  }

  at::Tensor apply(const tei::MlpProjection&,
                   const at::Tensor& x,
                   const at::Tensor& bias,
                   const tei::MlpContext&,
                   tei::MlpEpilogue epilogue,
                   const at::Tensor& scales) const override {
    TORCH_CHECK(epilogue == tei::MlpEpilogue::None && !scales.defined(),
                "MLP did not respect method capabilities");
    ++*calls_;
    return at::linear(x, weight_, bias);
  }
};

void check(const c10::Device& device, at::ScalarType dtype) {
  const auto opts = at::TensorOptions().device(device).dtype(dtype);
  auto wi = at::randn({64, 32}, opts) * .03, wo = at::randn({16, 64}, opts) * .03;
  auto bi = at::randn({64}, opts) * .1, bo = at::randn({16}, opts) * .1;
  for (auto kind : {tei::MlpActivation::Identity,
                    tei::MlpActivation::Relu,
                    tei::MlpActivation::Silu,
                    tei::MlpActivation::Gelu,
                    tei::MlpActivation::GeluTanh,
                    tei::MlpActivation::Tanh}) {
    tei::DenseMlp mlp({wi, bi}, {wo, bo}, {kind});
    for (auto precision : {tei::MlpPrecision::Model, tei::MlpPrecision::Fp8}) {
      if (precision == tei::MlpPrecision::Fp8 && !device.is_cuda())
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
          projected = tei::Fp8Linear(wi).forward(rows, bi);
        at::Tensor activated;
        switch (kind) {
        case tei::MlpActivation::Identity:
          activated = projected;
          break;
        case tei::MlpActivation::Relu:
          activated = at::relu(projected);
          break;
        case tei::MlpActivation::Silu:
          activated = at::silu(projected);
          break;
        case tei::MlpActivation::Gelu:
          activated = at::gelu(projected, "none");
          break;
        case tei::MlpActivation::GeluTanh:
          activated = at::gelu(projected, "tanh");
          break;
        default:
          activated = at::tanh(projected);
          break;
        }
        auto expected =
            mlp.fp8() ? tei::Fp8Linear(wo).forward(activated, bo) : at::linear(activated, wo, bo);
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
        auto q = tei::fp8_quantize_rows(x);
        check_equal(mlp.forward(q.first, {}, q.second), mlp.forward(x));
      }
    }
  }

  auto up = at::randn_like(wi) * .03, bu = at::randn_like(bi) * .1;
  tei::DenseMlp separate({wi, bi}, {up, bu}, {wo, bo}, {tei::MlpActivation::Silu});
  auto x = at::randn({7, 32}, opts);
  check_equal(separate.forward(x),
              at::linear(at::silu(at::linear(x, wi, bi)) * at::linear(x, up, bu), wo, bo));
  if (device.is_cuda()) {
    separate.prepare(tei::MlpPrecision::Fp8);
    auto q = tei::fp8_quantize_rows(x);
    check_equal(separate.forward(q.first, {}, q.second), separate.forward(x));
    check_equal(separate.forward(x),
                tei::Fp8Linear(wo).forward(at::silu(tei::Fp8Linear(wi).forward(x, bi)) *
                                               tei::Fp8Linear(up).forward(x, bu),
                                           bo));
  }

  tei::ProjectionRecipe clipped;
  clipped.separate_bias = true;
  clipped.cast_input = true;
  clipped.input_clip = std::array<double, 2>{-.4, .5};
  clipped.output_clip = std::array<double, 2>{-.1, .2};
  tei::DenseMlp bounded({wi, bi, clipped}, {wo, bo, clipped}, {tei::MlpActivation::Tanh});
  for (auto precision : {tei::MlpPrecision::Model, tei::MlpPrecision::Fp8}) {
    if (precision == tei::MlpPrecision::Fp8 && !device.is_cuda())
      continue;
    bounded.prepare(precision);
    auto project = [&](const at::Tensor& value, const at::Tensor& weight, const at::Tensor& bias) {
      auto input = value.clamp(-.4, .5).to(dtype);
      auto output =
          bounded.fp8() ? tei::Fp8Linear(weight).forward(input) : at::linear(input, weight);
      return (output + bias).clamp(-.1, .2);
    };
    auto input = x.to(at::kFloat);
    const auto expected = project(at::tanh(project(input, wi, bi)), wo, bo);
    int gemms = 0;
    tei::MlpContext context;
    context.gemm = [&](const tei::MlpProjection& projection, const at::Tensor& rows) {
      ++gemms;
      TORCH_CHECK(!projection.gemm_bias().defined(), "Separate bias escaped into GEMM");
      TORCH_CHECK(rows.scalar_type() == dtype && rows.abs().max().item<float>() <= .501,
                  "Projection kernel observed input before cast/clipping");
      return at::linear(rows, projection.weight());
    };
    check_equal(bounded.forward(input, context), expected);
    TORCH_CHECK(gemms == (bounded.fp8() ? 0 : 2), "Unexpected precision backend selection");
  }
  tei::DenseMlp owned({wi, bi, clipped}, {wo, bo, clipped}, {tei::MlpActivation::Tanh});
  auto calls = std::make_shared<int>(0);
  const auto expected_owned = owned.forward(x.to(at::kFloat));
  owned.prepare([&](const tei::MlpProjection& projection) {
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
    copied.prepare(
        [&](const tei::MlpProjection& projection) -> std::shared_ptr<const tei::MlpLinearMethod> {
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
    auto packed = at::cat({wi, up}, 0);
    auto recipe = tei::MlpActivationRecipe::numerical(
        [](const at::Tensor& rows) { return tei::fused_gated_activation(rows, 5); }, true);
    recipe.fusion = tei::MlpFusion::RoundedSwiGlu;
    tei::DenseMlp mlp({packed}, {wo, bo}, recipe);
    mlp.prepare(tei::MlpPrecision::Fp8);
    check_equal(mlp.forward(x),
                tei::Fp8Linear(wo).forward(
                    tei::fused_gated_activation(tei::Fp8Linear(packed).forward(x), 5), bo));
    tei::DenseMlp invalid({wi}, {at::zeros({17, 64}, opts)}, {tei::MlpActivation::Relu});
    bool rejected = false;
    try {
      invalid.prepare(tei::MlpPrecision::Fp8);
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
