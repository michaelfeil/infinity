/// \file
/// Shared contract for the native LibTorch model families.
/// Declares packed-input inference, optional task and media capabilities, and
/// prefix-cache size estimates consumed by the bridge.

#pragma once
#include <ATen/ATen.h>
#include <atomic>
#include <limits>
#include "batching/packed_input.h"

namespace infinity {
// Saturating estimates reject impossible admissions instead of overflowing a
// configured byte budget. Adapters report owned tensor payloads, not scratch.
inline uint64_t prefix_storage_bytes(uint64_t rows, uint64_t width, uint64_t element_size) {
  const auto maximum = std::numeric_limits<uint64_t>::max();
  if (width && rows > maximum / width)
    return maximum;
  const auto elements = rows * width;
  return element_size && elements > maximum / element_size ? maximum : elements * element_size;
}

/// Add retained-byte estimates, saturating rather than wrapping on overflow.
inline uint64_t prefix_storage_sum(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max()
                                                      : a + b;
}

/// Base interface for packed model execution and optional inference capabilities.
/// Implementations prepare parameters in ready() and evaluate inputs in forward().
/// The bridge uses capability methods to select supported execution paths.
class Model {
public:
  Model() : prefix_epoch(next_prefix_epoch.fetch_add(1, std::memory_order_relaxed)) {}

  Model(const Model&) : Model() {}

  /// Identifies this model instance for resident-cache reuse; copied models get a new namespace.
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

  /// Validate and prepare loaded parameters before accepting inference requests.
  virtual void ready() = 0;

  /// Width of each token row returned by forward(), before any pooled-only projection.
  virtual int64_t output_width() const {
    return 0;
  }

  /// Number of classifier outputs per selected row; zero means no classification head.
  virtual int64_t classification_width() const {
    return 0;
  }

  // Resolve data-dependent media indices before CUDA graph capture.
  virtual void prepare_input(PackedInput&) const {}

  /// Evaluate packed sequences and return token rows in the adapter's output space. Sequence
  /// boundaries come from PackedInput, rather than padding between sequences.
  virtual at::Tensor forward(const PackedInput&) const = 0;

  /// Whether pooled-only last-token requests can skip unused final-layer rows.
  virtual bool supports_last_token(const PackedInput&) const {
    return false;
  }

  virtual at::Tensor forward_last(const PackedInput& input) const {
    auto hidden = forward(input);
    auto rows = input.cumulative.narrow(0, 1, input.batch).to(at::kLong) - 1;
    return hidden.index_select(0, rows);
  }

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

  /// Convert hidden rows to sparse vocabulary scores; unsupported adapters throw.
  virtual at::Tensor token_scores(const at::Tensor&) const {
    TORCH_CHECK(false, "This model does not implement SPLADE");
  }

  /// Run classification; the boolean selects tokenwise rather than sequence readout. Unsupported
  /// adapters throw instead of returning embedding rows as logits.
  virtual at::Tensor predict(const PackedInput&, bool) const {
    TORCH_CHECK(false, "This model does not implement classification");
  }

  // Optional sequence readout from complete, already restored logical rows.
  // Defined results must own writable storage independent of the source rows.
  // Undefined means that this model has not qualified backbone-free readout.
  virtual at::Tensor classify_hidden(const PackedInput&, const at::Tensor&) const {
    return {};
  }

  /// Whether identical causal prefixes may share compact projection rows in one request.
  virtual bool supports_radix() const {
    return false;
  }

  /// Whether the specified execution operation can reuse resident prefix snapshots.
  virtual bool supports_prefix_cache(int) const {
    return false;
  }

  virtual int64_t prefix_layers() const {
    return 0;
  }

  /// Estimate retained state for this many prefix tokens, excluding transient workspace.
  virtual uint64_t prefix_payload_bytes(int64_t) const {
    return 0;
  }

  /// Whether reuse requires a complete saved checkpoint rather than a shorter partial match.
  virtual bool prefix_checkpoint_only() const {
    return false;
  }

private:
  inline static std::atomic<uint64_t> next_prefix_epoch{1};
  const uint64_t prefix_epoch;
};
}
