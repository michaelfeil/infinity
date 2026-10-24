// SPDX-License-Identifier: Apache-2.0
/// \file
/// CUDA batching of bit-preserving copies from strided state to contiguous storage.

#include "kernels/prefix_state_copy.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cstdint>

namespace tei {
namespace {
/// Byte-addressed row-strided source and contiguous destination copy descriptor.
struct Copy {
  const uint8_t* source;
  uint8_t* target;
  int64_t rows, width, stride;
};

/// Fixed-capacity batch of state-copy descriptors passed to the CUDA kernel.
struct Copies {
  Copy items[64];
};

static_assert(sizeof(Copies) <= 4096);

template <class Word> __device__ void copy_rows(const Copy& copy) {
  const auto width = copy.width / int64_t(sizeof(Word)),
             stride = copy.stride / int64_t(sizeof(Word));
  const auto* source = reinterpret_cast<const Word*>(copy.source);
  auto* target = reinterpret_cast<Word*>(copy.target);
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; index < copy.rows * width;
       index += int64_t(gridDim.x) * blockDim.x)
    target[index] = source[(index / width) * stride + index % width];
}

__global__ void copy_states(Copies copies) {
  const auto& copy = copies.items[blockIdx.y];
  const auto alignment = reinterpret_cast<uintptr_t>(copy.source) |
                         reinterpret_cast<uintptr_t>(copy.target) | uint64_t(copy.width) |
                         uint64_t(copy.stride);
  if (!(alignment & 15))
    copy_rows<uint4>(copy);
  else if (!(alignment & 3))
    copy_rows<uint32_t>(copy);
  else if (!(alignment & 1))
    copy_rows<uint16_t>(copy);
  else
    copy_rows<uint8_t>(copy);
}
}

/// Validate every copy before launching any, then submit bounded
/// descriptor batches on the current stream.
bool prefix_state_copies(const std::vector<at::Tensor>& targets,
                         const std::vector<at::Tensor>& sources,
                         bool leading_rows) {
  if (targets.empty())
    return sources.empty();
  if (targets.size() != sources.size() || !targets[0].defined() || !targets[0].is_cuda())
    return false;
  const auto device = targets[0].device();
  std::vector<Copy> copies;
  copies.reserve(targets.size());
  int64_t largest = 0;
  for (size_t i = 0; i < targets.size(); ++i) {
    const auto& target = targets[i];
    const auto& source = sources[i];
    if (!source.defined() || !target.defined() || source.device() != device ||
        target.device() != device || source.scalar_type() != target.scalar_type() ||
        !target.is_contiguous() || source.is_conj() || target.is_conj() || source.is_neg() ||
        target.is_neg() || source.dim() == 0 || source.size(0) == 0 || source.numel() == 0)
      return false;
    if (leading_rows) {
      if (target.dim() != source.dim() || target.size(0) <= 0 || target.size(0) > source.size(0) ||
          target.sizes().slice(1) != source.sizes().slice(1))
        return false;
    } else if (source.sizes() != target.sizes())
      return false;
    int64_t inner = 1;
    for (int64_t dim = source.dim() - 1; dim > 0; --dim) {
      if (source.size(dim) != 1 && source.stride(dim) != inner)
        return false;
      inner *= source.size(dim);
    }
    const auto width = inner * int64_t(source.element_size()),
               stride = source.stride(0) * int64_t(source.element_size());
    if (stride < 0)
      return false;
    copies.push_back({static_cast<const uint8_t*>(source.const_data_ptr()),
                      static_cast<uint8_t*>(target.mutable_data_ptr()),
                      target.size(0),
                      width,
                      stride});
    largest = std::max(largest, int64_t(target.nbytes()));
  }
  c10::cuda::CUDAGuard guard(device);
  const auto blocks = std::min<int64_t>(128, std::max<int64_t>(1, (largest + 4095) / 4096));
  for (size_t first = 0; first < copies.size(); first += 64) {
    Copies batch{};
    const auto count = std::min<size_t>(64, copies.size() - first);
    std::copy_n(copies.begin() + first, count, batch.items);
    copy_states<<<dim3(blocks, count), 256, 0, at::cuda::getCurrentCUDAStream()>>>(batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}
}
