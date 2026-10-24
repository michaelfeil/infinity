// SPDX-License-Identifier: Apache-2.0
/// \file
/// Pooled embedding projection-chain interface for Sentence Transformers modules.

#pragma once
#include "models/model.h"
#include "helpers/dense_mlp.h"
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

  /// Resolve the final pooled width and check the backbone width against the first projection.
  int64_t output_width(int64_t input_width) const;
  /// Apply ordered linear/tanh modules to pooled rows; the backbone's raw token rows stay separate.
  at::Tensor forward(const at::Tensor&) const;

private:
  /// One pooled linear projection with optional bias and tanh activation.
  struct Layer {
    /// weight is [out_features,in_features]; bias is optional [out_features]. Handles retain loaded
    /// parameter storage while activations pass through the chain.
    MlpProjection projection;
    int64_t in_features, out_features;
    bool tanh;
  };

  std::vector<Layer> layers;
};
}
