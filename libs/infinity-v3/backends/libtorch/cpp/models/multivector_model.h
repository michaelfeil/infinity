/// \file
/// Native adapter for returning token matrices from a backbone model.
/// Applies an optional learned projection and per-token L2 normalization.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/options.h"

namespace infinity {
// Token projections are part of the model, not sentence pooling. Keep their
// output width visible to the packed bridge before it reserves host buffers.
class MultiVectorModel final : public Model {
  std::unique_ptr<Model> backbone;
  /// Optional token projection [output_width,backbone_width] and matching bias. An absent
  /// projection preserves backbone width while still applying per-token normalization.
  at::Tensor projection, bias;
  double epsilon;

public:
  /// Take ownership of the backbone and retain optional projection/bias tensor handles. epsilon
  /// floors the per-token L2 denominator; it does not add a pooling or
  /// sequence-level normalization step.
  MultiVectorModel(std::unique_ptr<Model> model,
                   at::Tensor weight,
                   at::Tensor offset,
                   double eps = 1e-12)
      : backbone(std::move(model)), projection(std::move(weight)), bias(std::move(offset)),
        epsilon(eps) {}

  /// Prepare the backbone and reject projection/bias shapes that disagree with its token width.
  void ready() override {
    backbone->ready();
    TORCH_CHECK(!projection.defined() ||
                    (projection.dim() == 2 && projection.size(1) == backbone->output_width()),
                "Invalid multi-vector projection");
    TORCH_CHECK(!bias.defined() ||
                    (projection.defined() && bias.dim() == 1 && bias.size(0) == projection.size(0)),
                "Invalid multi-vector projection bias");
  }

  int64_t output_width() const override {
    return projection.defined() ? projection.size(0) : backbone->output_width();
  }

  bool supports_images() const override {
    return backbone->supports_images();
  }

  /// Return projected token rows normalized independently along their final dimension. The norm
  /// floor protects zero rows; sequence pooling is outside this adapter.
  at::Tensor forward(const PackedInput& input) const override {
    auto states = backbone->forward(input);
    if (projection.defined())
      states = at::linear(states, projection, bias);
    return states / at::linalg_vector_norm(states, 2, {-1}, true).clamp_min(epsilon);
  }
};
}
