// Native adaptation of backends/candle/src/models/gemma4_vision.rs.
// Upstream Apache-2.0 notices remain in LICENSE and NOTICE.
/// \file
/// Model-owned cache for immutable image feature tensors.
/// Tracks bounded retention and CUDA readiness before publishing cached results.

#pragma once
#include <ATen/ATen.h>
#include <array>
#include <list>
#include <memory>
#include <utility>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
namespace tei {
// Cached tensors are immutable to callers. The model owns one device/dtype.
class ImageFeatureCache {
public:
  using Key = std::array<uint8_t, 32>;

private:
  /// Cached feature tensor, content key, and optional CUDA readiness event.
  struct Entry {
    Key key;
    at::Tensor value;
#ifdef TEI_TORCH_CUDA_KERNELS
    std::unique_ptr<at::cuda::CUDAEvent> ready;
#endif
  };

  mutable std::mutex mutex;
  mutable std::list<Entry> entries;
  mutable size_t bytes = 0;
  size_t byte_limit, entry_limit;

  /// Look up and refresh LRU order under the mutex; CUDA hits wait on the producer event on the
  /// current stream and register allocator use before returning immutable features.
  at::Tensor find(const Key& key) const {
    for (auto found = entries.begin(); found != entries.end(); ++found)
      if (found->key == key) {
#ifdef TEI_TORCH_CUDA_KERNELS
        if (found->ready) {
          auto stream = at::cuda::getCurrentCUDAStream(found->value.get_device());
          found->ready->block(stream);
          c10::cuda::CUDACachingAllocator::recordStream(found->value.storage().data_ptr(), stream);
        }
#endif
        auto result = found->value;
        entries.splice(entries.end(), entries, found);
        return result;
      }
    return {};
  }

  /// Retain an immutable result under the mutex, evicting oldest entries to honor both limits.
  /// Views are cloned when necessary so accounting covers the actual retained allocation.
  at::Tensor publish(const Key& key, at::Tensor value) const {
    TORCH_CHECK(value.defined(), "Undefined image feature cache value");
    const size_t logical_bytes = value.numel() * value.element_size();
    if (!entry_limit || logical_bytes > byte_limit)
      return value;
    // A contiguous slice may still retain a larger allocation. Clone only when
    // needed so the budget measures actual owned storage, not its visible view.
    if (!value.is_contiguous() || value.storage_offset() != 0 ||
        value.storage().nbytes() != logical_bytes)
      value = value.clone(at::MemoryFormat::Contiguous);
    const size_t owned_bytes = value.storage().nbytes();
    if (owned_bytes > byte_limit)
      return value;
    while (!entries.empty() &&
           (bytes > byte_limit - owned_bytes || entries.size() >= entry_limit)) {
      bytes -= entries.front().value.storage().nbytes();
      entries.pop_front();
    }
    Entry entry;
    entry.key = key;
    entry.value = value;
#ifdef TEI_TORCH_CUDA_KERNELS
    if (value.is_cuda()) {
      entry.ready = std::make_unique<at::cuda::CUDAEvent>();
      entry.ready->record(at::cuda::getCurrentCUDAStream(value.get_device()));
    }
#endif
    bytes += owned_bytes;
    entries.push_back(std::move(entry));
    return value;
  }

public:
  explicit ImageFeatureCache(size_t byte_limit = 64 * 1024 * 1024, size_t entry_limit = 32)
      : byte_limit(byte_limit), entry_limit(entry_limit) {}

  // Serialize a cache miss and its publication. GPU work remains asynchronous;
  // ready events make hits safe on a different caller stream.
  template <class Compute> at::Tensor get_or_compute(const Key& key, Compute compute) const {
    std::lock_guard<std::mutex> lock(mutex);
    auto value = find(key);
    return value.defined() ? value : publish(key, compute());
  }

  /// Compute distinct keyed misses in one callback and restore the original request order. Missing
  /// keys always compute independently; the callback receives indices into keys. The mutex covers
  /// lookup, computation submission, and publication; do not reenter this cache.
  template <class Compute>
  std::vector<at::Tensor> get_batch(const std::vector<std::optional<Key>>& keys,
                                    Compute compute) const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<at::Tensor> result(keys.size());
    std::vector<size_t> missing, slots(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i])
        result[i] = find(*keys[i]);
      if (result[i].defined())
        continue;
      size_t slot = 0;
      while (slot < missing.size() && (!keys[i] || keys[i] != keys[missing[slot]]))
        ++slot;
      slots[i] = slot;
      if (slot == missing.size())
        missing.push_back(i);
    }
    auto values = missing.empty() ? std::vector<at::Tensor>{} : compute(missing);
    TORCH_CHECK(values.size() == missing.size(), "Image feature cache batch size mismatch");
    for (size_t slot = 0; slot < missing.size(); ++slot) {
      const auto& key = keys[missing[slot]];
      if (key)
        values[slot] = publish(*key, values[slot]);
    }
    for (size_t i = 0; i < keys.size(); ++i)
      if (!result[i].defined())
        result[i] = values[slots[i]];
    return result;
  }

  /// Return the retained entry count and owned storage bytes under the cache mutex.
  std::pair<size_t, size_t> usage() const {
    std::lock_guard<std::mutex> lock(mutex);
    return {entries.size(), bytes};
  }
};
}
