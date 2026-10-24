#pragma once
#include "batching/packed_input.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#endif
namespace tei {
// Bound layer intermediates rather than just uploaded pixels. A single large
// image remains executable; batching uses a conservative share of headroom.
inline std::vector<std::pair<size_t, size_t>> vision_batches(const std::vector<ImageInput>& images,
                                                             const c10::Device& device,
                                                             int64_t hidden,
                                                             int64_t intermediate) {
  size_t budget = 2ULL << 30;
#ifdef TEI_TORCH_CUDA_KERNELS
  if (device.is_cuda()) {
    c10::cuda::CUDAGuard guard(device);
    size_t available = 0, total = 0;
    C10_CUDA_CHECK(cudaMemGetInfo(&available, &total));
    const auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(guard.current_device().index());
    available += stats.reserved_bytes[0].current - stats.allocated_bytes[0].current;
    budget = std::min(budget, available / 4);
  }
#endif
  // Include FP32 rotary/normalization temporaries and gated MLP buffers.
  const auto limit = std::max<size_t>(1, budget / (4 * (8 * hidden + 2 * intermediate)));
  std::vector<std::pair<size_t, size_t>> batches;
  size_t start = 0, patches = 0;
  for (size_t i = 0; i < images.size(); ++i) {
    const auto& grid = images[i].grid_thw;
    TORCH_CHECK(grid[0] > 0 && grid[1] > 0 && grid[2] > 0, "Invalid vision batch grid");
    const size_t count = images[i].pixels.size(0);
    if (i > start && (patches >= limit || count > limit - patches)) {
      batches.emplace_back(start, i);
      start = i;
      patches = 0;
    }
    patches += count;
  }
  if (start < images.size())
    batches.emplace_back(start, images.size());
  return batches;
}
}
