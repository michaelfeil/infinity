// SPDX-License-Identifier: Apache-2.0
#include "models/dense_models.h"
#include <c10/core/InferenceMode.h>
#include <iostream>

int main() {
  c10::InferenceMode inference;
  tei::Options options;
  options.values = {{"_dense_count", "2"},
                    {"_dense.0.in_features", "4"},
                    {"_dense.0.out_features", "3"},
                    {"_dense.0.bias", "true"},
                    {"_dense.0.activation_function", "torch.nn.modules.activation.Tanh"},
                    {"_dense.1.in_features", "3"},
                    {"_dense.1.out_features", "2"}};
  tei::Weights weights;
  weights["_dense.0.linear.weight"] = at::arange(12, at::kFloat).view({3, 4}) / 20;
  weights["_dense.0.linear.bias"] = at::tensor({.1f, -.2f, .3f});
  weights["_dense.1.linear.weight"] = at::tensor({1.f, 0.f, -1.f, .2f, .3f, .4f}).view({2, 3});
  tei::DenseChain chain(options, weights);
  // Tensor references remain valid when decoder construction releases the loader map.
  weights.clear();
  auto input = at::tensor({1.f, 2.f, 3.f, 4.f, -1.f, 0.f, 1.f, 2.f}).view({2, 4});
  auto first = at::tanh(at::mm(input, at::arange(12, at::kFloat).view({3, 4}).t() / 20) +
                        at::tensor({.1f, -.2f, .3f}));
  auto expected = at::mm(first, at::tensor({1.f, 0.f, -1.f, .2f, .3f, .4f}).view({2, 3}).t());
  TORCH_CHECK(chain.output_width(4) == 2, "Dense width must reflect entire chain");
  TORCH_CHECK(at::allclose(chain.forward(input), expected), "Dense chain arithmetic differs");
  bool rejected = false;
  try {
    chain.output_width(3);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Invalid Dense input width must fail at startup");
  tei::DenseChain identity(tei::Options{}, tei::Weights{});
  TORCH_CHECK(identity.output_width(4) == 4 && at::equal(identity.forward(input), input),
              "Empty chain must preserve embeddings");
  std::cout << "Dense projection chain tests passed\n";
}
