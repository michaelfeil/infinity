#pragma once
#include <ATen/ATen.h>
#include <atomic>
#include <limits>
#include "batching/packed_input.h"

namespace tei {
// Saturating estimates reject impossible admissions instead of overflowing a
// configured byte budget. Adapters report owned tensor payloads, not scratch.
inline uint64_t prefix_storage_bytes(uint64_t rows, uint64_t width, uint64_t element_size) {
  const auto maximum = std::numeric_limits<uint64_t>::max();
  if (width && rows > maximum / width)
    return maximum;
  const auto elements = rows * width;
  return element_size && elements > maximum / element_size ? maximum : elements * element_size;
}

inline uint64_t prefix_storage_sum(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max()
                                                      : a + b;
}

class Model {
public:
  Model() : prefix_epoch(next_prefix_epoch.fetch_add(1, std::memory_order_relaxed)) {}

  Model(const Model&) : Model() {}

  uint64_t prefix_namespace() const {
    return prefix_epoch;
  }

  virtual ~Model() = default;

  virtual bool supports_images() const {
    return false;
  }

  virtual bool supports_audio() const {
    return false;
  }

  virtual void ready() = 0;

  virtual int64_t output_width() const {
    return 0;
  }

  virtual int64_t classification_width() const {
    return 0;
  }

  // Resolve data-dependent media indices before CUDA graph capture.
  virtual void prepare_input(PackedInput&) const {}

  virtual at::Tensor forward(const PackedInput&) const = 0;

  // Per-sequence embedding mean; preserve the bridge's F32 default unless a
  // model has independently qualified model-dtype reduction semantics.
  virtual at::Tensor mean_pool(const at::Tensor& value) const {
    return value.mean(0, true, at::kFloat);
  }

  virtual bool model_dtype_mean_pool() const {
    return false;
  }

  // Optional pooled-output transform; token-level projections belong in forward().
  virtual at::Tensor project_pooled(const at::Tensor& value) const {
    return value;
  }

  virtual at::Tensor token_scores(const at::Tensor&) const {
    TORCH_CHECK(false, "This model does not implement SPLADE");
  }

  virtual at::Tensor predict(const PackedInput&, bool) const {
    TORCH_CHECK(false, "This model does not implement classification");
  }

  // Optional sequence readout from complete, already restored logical rows.
  // Defined results must own writable storage independent of the source rows.
  // Undefined means that this model has not qualified backbone-free readout.
  virtual at::Tensor classify_hidden(const PackedInput&, const at::Tensor&) const {
    return {};
  }

  virtual bool supports_radix() const {
    return false;
  }

  virtual bool supports_prefix_cache(int) const {
    return false;
  }

  virtual int64_t prefix_layers() const {
    return 0;
  }

  virtual uint64_t prefix_payload_bytes(int64_t) const {
    return 0;
  }

  virtual bool prefix_checkpoint_only() const {
    return false;
  }

private:
  inline static std::atomic<uint64_t> next_prefix_epoch{1};
  const uint64_t prefix_epoch;
};
}
