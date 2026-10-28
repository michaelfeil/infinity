/// \file
/// Interface and counters for the bounded native CUDA graph cache.
/// Model execution and supported readouts are captured separately from host transfers.

#pragma once
#include <memory>
#include "models/model.h"
#include "helpers/math_policy.h"

namespace infinity {
// The caller serializes model inference and orders CUDA streams. Graphs cache exact packed shapes only;
// Caller output ordering and all transfers remain outside capture. Outputs are overwritten by
// the next replay of the same entry, so consume them before another inference.
class CudaGraphCache {
  struct Impl;
  std::unique_ptr<Impl> impl;
  friend class PrefixCache;
  at::Tensor forward(
      const Model&, const PackedInput&, int, const MathPolicy*, PrefixContext* publish = nullptr);

public:
  /// Graph-cache reuse, capture, eviction, and retained-memory counters.
  struct Stats {
    uint64_t hits = 0, misses = 0, captures = 0, attempts = 0, evictions = 0, live_bytes = 0,
             admitted_bytes = 0, prefix_hits = 0, prefix_captures = 0;
  };

  explicit CudaGraphCache(size_t entries = 4,
                          int64_t max_tokens = 4096,
                          size_t reuse = 3,
                          uint64_t max_bytes = 512ULL << 20,
                          size_t max_captures = 8);
  ~CudaGraphCache();
  CudaGraphCache(const CudaGraphCache&) = delete;
  CudaGraphCache& operator=(const CudaGraphCache&) = delete;
  // operation: 0 hidden states, 1 sequence classification, 2 token classification,
  // 3 last-token embeddings.
  /// Execute or replay an exact-shape entry; returned graph outputs
  /// may be overwritten by its next replay.
  at::Tensor forward(const Model&, const PackedInput&, int operation = 0);
  size_t size() const;
  Stats stats() const;
  /// Enable capture of full layer/output views needed to publish
  /// independent resident prefix snapshots.
  void retain_prefix_states(bool enabled = true);
};
}
