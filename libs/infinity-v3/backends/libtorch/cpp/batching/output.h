// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "helpers/admission_error.h"
#include "helpers/device_admission.h"
#include <ATen/ATen.h>
#include "helpers/pinned_admission.h"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <vector>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace tei {
// Two reusable slots bound pinned memory and pending GPU outputs per replica.
// A lease stays alive through CPU result construction, including cancellation.
struct OutputSlot {
  bool busy = false;
  at::Tensor host, device_output;
  std::vector<at::Tensor> gpu;
  std::vector<at::Tensor> inputs;
};
struct OutputPool {
  std::mutex mutex;
  std::condition_variable available;
  std::array<OutputSlot, 2> slots;
  size_t next_slot = 0;
  size_t max_bytes = 512ULL << 20;
  size_t retained_bytes() const {
    size_t bytes = 0;
    for (const auto& slot : slots) {
      if (slot.host.defined()) bytes += slot.host.storage().nbytes();
      if (slot.device_output.defined()) bytes += slot.device_output.storage().nbytes();
      for (const auto& input : slot.inputs)
        if (input.defined()) bytes += input.storage().nbytes();
    }
    return bytes;
  }
};
class OutputLease {
  std::shared_ptr<OutputPool> pool;
  OutputSlot* slot = nullptr;
  at::Device device;
  size_t input_cursor = 0;
  bool output_started = false;
  std::shared_ptr<DeviceMemoryPermit> memory_permit;
  // Called under the pool lock, before allocating replacement storage.
  void make_room(size_t bytes, size_t replaced, bool shrink_output = false) {
    require_memory_admission(bytes <= pool->max_bytes, "Native output staging exceeds its memory budget");
    auto fits = [&] { return pool->retained_bytes() - replaced <= pool->max_bytes - bytes; };
    // Reacquiring a slot makes it busy, but its trailing pins have not been
    // used by this lease. Active input pins (including the replacement) stay live.
    if (!fits() && input_cursor < slot->inputs.size()) slot->inputs.resize(input_cursor);
    if (!fits()) {
      for (auto& idle : pool->slots) {
        if (&idle == slot || idle.busy) continue;
        idle.host = at::Tensor(); idle.device_output = at::Tensor(); idle.inputs.clear();
        if (fits()) break;
      }
    }
    if (!fits() && shrink_output && !output_started && slot->host.defined()
        && slot->device_output.defined() && size <= std::numeric_limits<size_t>::max() / (2 * sizeof(float))) {
      const auto capacity = std::max(size, size_t{1});
      const auto needed = capacity * sizeof(float);
      const auto previous = slot->host.storage().nbytes() + slot->device_output.storage().nbytes();
      if (previous > 2 * needed && 2 * needed <= pool->max_bytes - bytes
          && pool->retained_bytes() - replaced - previous <= pool->max_bytes - bytes - 2 * needed) {
        // No output copy references these oversized buffers yet. Keep only
        // this request's output capacity so its real inputs can be admitted.
        slot->host = at::Tensor(); slot->device_output = at::Tensor();
        slot->host = allocate_pinned_staging(static_cast<int64_t>(capacity),
          at::TensorOptions().dtype(at::kFloat));
        slot->device_output = at::empty({static_cast<int64_t>(capacity)},
          at::TensorOptions().dtype(at::kFloat).device(device));
      }
    }
    require_memory_admission(fits(), "Native output staging memory budget is occupied by pending results");
  }

#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAStream stream;
  at::cuda::CUDAEvent ready;
  bool recorded = false;
  bool finished = false;
#endif
public:
  const size_t size;
  OutputLease(std::shared_ptr<OutputPool> owner, at::Device dev, size_t count)
      : pool(std::move(owner)), device(dev),
#ifdef TEI_TORCH_CUDA_KERNELS
        stream(c10::cuda::getCurrentCUDAStream(dev.index())),
#endif
        size(count) {
    std::unique_lock lock(pool->mutex);
    // One model producer and FIFO completion alternate A/B, waiting through
    // result packing before reusing the next slot.
    pool->available.wait(lock, [&] { return !pool->slots[pool->next_slot].busy; });
    slot = &pool->slots[pool->next_slot];
    slot->busy = true;
    pool->next_slot = (pool->next_slot + 1) % pool->slots.size();
  }
  ~OutputLease() {
#ifdef TEI_TORCH_CUDA_KERNELS
    // Also drain partially submitted work on errors, before releasing pins.
    try { if (!finished) { if (recorded) ready.synchronize(); else stream.synchronize(); } }
    catch (...) { /* CUDA failures are reported by submit/wait. */ }
#endif
    slot->gpu.clear();
    memory_permit.reset();
    { std::lock_guard lock(pool->mutex);
#ifdef TEI_TORCH_CUDA_KERNELS
      if (!recorded) {
        // A failed submission has no result to preserve. Do not cache its
        // partially allocated buffers after draining any submitted transfers.
        slot->host = at::Tensor(); slot->device_output = at::Tensor(); slot->inputs.clear();
      }
#endif
      slot->busy = false;
    }
    pool->available.notify_one();
  }
  void allocate() {
    // Keep wait() non-null even when no output rows were requested.
    TORCH_CHECK(size <= static_cast<size_t>(std::numeric_limits<int64_t>::max())
      && size <= std::numeric_limits<size_t>::max() / (2 * sizeof(float)),
                "Native output capacity is too large");
    const auto capacity = static_cast<int64_t>(std::max(size, size_t{1}));
    std::lock_guard lock(pool->mutex);
    const bool grow_host = !slot->host.defined() || static_cast<size_t>(slot->host.numel()) < size;
    const bool grow_device = !slot->device_output.defined() || static_cast<size_t>(slot->device_output.numel()) < size;
    const auto bytes = static_cast<size_t>(capacity) * sizeof(float);
    const auto replaced = (grow_host && slot->host.defined() ? slot->host.storage().nbytes() : 0)
      + (grow_device && slot->device_output.defined() ? slot->device_output.storage().nbytes() : 0);
    make_room(bytes * (static_cast<size_t>(grow_host) + static_cast<size_t>(grow_device)), replaced);
    // Old storage has no remaining consumer after its previous lease completed.
    // Release it before allocating the larger replacement to avoid doubling peaks.
    if (grow_host) slot->host = at::Tensor();
    if (grow_device) slot->device_output = at::Tensor();
    if (grow_host)
      slot->host = allocate_pinned_staging(capacity,
          at::TensorOptions().dtype(at::kFloat));
    if (grow_device)
      slot->device_output = at::empty({capacity},
          at::TensorOptions().dtype(at::kFloat).device(device));
  }
  void admit_memory(std::shared_ptr<DeviceMemoryPermit> permit) { memory_permit=std::move(permit); }
  at::Tensor upload(const at::Tensor& host) {
    TORCH_CHECK(host.device().is_cpu() && host.is_contiguous(), "Invalid staged input");
    std::unique_lock lock(pool->mutex);
    if (input_cursor == slot->inputs.size()) slot->inputs.emplace_back();
    auto& pin = slot->inputs[input_cursor++];
    if (!pin.defined() || pin.scalar_type() != host.scalar_type() || pin.numel() < host.numel()) {
      make_room(host.nbytes(), pin.defined() ? pin.storage().nbytes() : 0, true);
      pin = at::Tensor();
      pin = allocate_pinned_staging(host.numel(), host.options());
    }
    lock.unlock();
    auto view = pin.narrow(0, 0, host.numel()).view(host.sizes());
    std::memcpy(view.data_ptr(), host.data_ptr(), host.nbytes());
    auto result = view.to(device, /*non_blocking=*/true);
    slot->gpu.push_back(result);
    return result;
  }
  void copy(const at::Tensor& tensor, size_t offset) {
    output_started = true;
    auto source = tensor.contiguous().view({-1});
    TORCH_CHECK(offset <= size && static_cast<size_t>(source.numel()) <= size - offset,
                "Native output exceeds provided capacity");
    auto converted = slot->device_output.narrow(0, offset, source.numel());
    converted.copy_(source, true);
    slot->host.narrow(0, offset, source.numel()).copy_(converted, true);
    slot->gpu.push_back(std::move(source));
  }
  void record() {
#ifdef TEI_TORCH_CUDA_KERNELS
    // H2D, graph replay and D2H use this same stream. The next replay cannot
    // overwrite graph inputs/outputs before this submission has consumed them.
    ready.record(stream);
    recorded = true;
#endif
  }
  const float* wait() {
#ifdef TEI_TORCH_CUDA_KERNELS
    c10::cuda::CUDAGuard guard(stream.device());
    TORCH_CHECK(recorded, "Output event was not recorded");
    ready.synchronize();
    finished = true;
#endif
    memory_permit.reset();
    return slot->host.data_ptr<float>();
  }
};
}
