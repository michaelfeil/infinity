// SPDX-License-Identifier: Apache-2.0
/// \file
/// Pooled embedding projection-chain interface for Sentence Transformers modules.

#pragma once
#include "models/model.h"
#include "helpers/options.h"

namespace tei {
// Sentence Transformers projections transform pooled rows only. Raw token
// embeddings retain the backbone width, including when both outputs are requested.
class DenseChain {
public:
  DenseChain(const Options&, const Weights&);

  bool empty() const {
    return layers.empty();
  }

  int64_t output_width(int64_t input_width) const;
  at::Tensor forward(const at::Tensor&) const;

private:
  /// One pooled linear projection with optional bias and tanh activation.
  struct Layer {
    at::Tensor weight, bias;
    int64_t in_features, out_features;
    bool tanh;
  };

  std::vector<Layer> layers;
};
}
