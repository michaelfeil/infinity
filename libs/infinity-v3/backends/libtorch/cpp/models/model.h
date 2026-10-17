#pragma once
#include <ATen/ATen.h>
#include "batching/packed_input.h"

namespace tei {
class Model {
public:
  virtual ~Model() = default;
  virtual bool supports_images() const { return false; }
  virtual bool supports_audio() const { return false; }
  virtual void ready() = 0;
  virtual int64_t output_width() const { return 0; }
  virtual int64_t classification_width() const { return 0; }
  // Resolve data-dependent media indices before CUDA graph capture.
  virtual void prepare_input(PackedInput&) const {}
  virtual at::Tensor forward(const PackedInput&) const = 0;
  // Per-sequence embedding mean; preserve the bridge's F32 default unless a
  // model has independently qualified model-dtype reduction semantics.
  virtual at::Tensor mean_pool(const at::Tensor& value) const {
    return value.mean(0, true, at::kFloat);
  }
  virtual bool model_dtype_mean_pool() const { return false; }
  // Optional pooled-output transform; token-level projections belong in forward().
  virtual at::Tensor project_pooled(const at::Tensor& value) const { return value; }
  virtual at::Tensor token_scores(const at::Tensor&) const {
    TORCH_CHECK(false, "This model does not implement SPLADE");
  }
  virtual at::Tensor predict(const PackedInput&, bool) const {
    TORCH_CHECK(false, "This model does not implement classification");
  }
  virtual bool supports_radix() const { return false; }
  // Exposed only after ready(); unknown implementations cannot bypass admission.
};
}
