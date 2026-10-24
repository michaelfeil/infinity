// SPDX-License-Identifier: Apache-2.0
/// \file
/// Loading, validation, and execution of pooled Sentence Transformers Dense modules.

#include "models/dense_models.h"

namespace tei {
/// Load ordered Sentence Transformers Dense modules, checking width continuity and supported
/// Identity/Tanh activations. Tensor handles share loaded parameter storage; inference
/// transforms pooled rows only.
DenseChain::DenseChain(const Options& options, const Weights& weights) {
  const auto count = options.integer("_dense_count", 0);
  TORCH_CHECK(count >= 0, "Dense module count must be nonnegative");
  for (int64_t i = 0; i < count; ++i) {
    const auto prefix = "_dense." + std::to_string(i) + ".";
    const auto in = options.integer(prefix + "in_features", 0);
    const auto out = options.integer(prefix + "out_features", 0);
    TORCH_CHECK(in > 0 && out > 0, "Dense module dimensions must be positive");
    auto w = weight(weights, prefix + "linear.weight");
    TORCH_CHECK(w.dim() == 2 && w.size(0) == out && w.size(1) == in,
                "Dense weight shape does not match config");
    at::Tensor b;
    if (options.boolean(prefix + "bias", false)) {
      b = weight(weights, prefix + "linear.bias");
      TORCH_CHECK(b.dim() == 1 && b.size(0) == out, "Dense bias shape does not match config");
      TORCH_CHECK(b.device() == w.device() && b.scalar_type() == w.scalar_type(),
                  "Dense weight and bias must share device and dtype");
    }
    auto activation =
        options.string(prefix + "activation_function", "torch.nn.modules.linear.Identity");
    TORCH_CHECK(activation == "torch.nn.modules.linear.Identity" ||
                    activation == "torch.nn.modules.activation.Tanh",
                "Unsupported Sentence Transformers Dense activation: ",
                activation);
    if (!layers.empty())
      TORCH_CHECK(layers.back().out_features == in, "Dense module chain dimensions disagree");
    layers.push_back({w, b, in, out, activation == "torch.nn.modules.activation.Tanh"});
  }
}

int64_t DenseChain::output_width(int64_t input_width) const {
  if (layers.empty())
    return input_width;
  TORCH_CHECK(layers.front().in_features == input_width,
              "Dense input width does not match pooled embeddings");
  return layers.back().out_features;
}

at::Tensor DenseChain::forward(const at::Tensor& value) const {
  auto result = value;
  for (const auto& layer : layers) {
    TORCH_CHECK(result.dim() == 2 && result.size(1) == layer.in_features,
                "Dense module input shape mismatch");
    // Mean pooling accumulates in FP32, then rounds back to the checkpoint
    // precision before projection, matching Candle's pooled-output contract.
    // Pooling may produce FP32 rows even when weights use Half/BF16. Round to this layer's
    // checkpoint dtype before projection, then apply its optional tanh stage.
    result = at::linear(result.to(layer.weight.scalar_type()), layer.weight, layer.bias);
    if (layer.tanh)
      result = at::tanh(result);
  }
  return result;
}
}
