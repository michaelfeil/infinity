#pragma once
#include <memory>
#include "models/model.h"
namespace tei {
// The caller serializes model inference. Graphs cache exact packed shapes only;
// pooling and all transfers remain outside capture. Outputs are overwritten by
// the next replay of the same entry, so consume them before another inference.
class CudaGraphCache {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  struct Stats { uint64_t hits=0, misses=0, captures=0, attempts=0, evictions=0, live_bytes=0, admitted_bytes=0; };
  explicit CudaGraphCache(size_t entries = 4, int64_t max_tokens = 4096,
    size_t reuse = 3, uint64_t max_bytes = 512ULL << 20, size_t max_captures = 8);
  ~CudaGraphCache();
  CudaGraphCache(const CudaGraphCache&) = delete;
  CudaGraphCache& operator=(const CudaGraphCache&) = delete;
  // operation: 0 hidden states, 1 sequence classification, 2 token classification.
  at::Tensor forward(const Model&, const PackedInput&, int operation = 0);
  size_t size() const;
  Stats stats() const;
};
}
