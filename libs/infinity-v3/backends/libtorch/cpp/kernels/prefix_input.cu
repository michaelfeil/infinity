// SPDX-License-Identifier: Apache-2.0
/// \file
/// CUDA copies that pack uncached token suffixes into independently owned inputs.

#include "kernels/prefix_input.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

namespace tei {
namespace {
constexpr int batch_limit = 64, threads = 256;

/// Source and destination row span for one sequence suffix copy.
struct Slice {
  int64_t input = 0, output = 0, count = 0;
};

/// Batched ID/position/type pointers and validated sequence-copy spans.
struct InputCopies {
  const int64_t* source[3]{};
  int64_t* target[3]{};
  Slice rows[batch_limit]{};
};

static_assert(sizeof(InputCopies) <= 4096);

__global__ void gather_prefix_inputs(InputCopies copies) {
  const auto row = copies.rows[blockIdx.x];
  for (int64_t cell = int64_t(blockIdx.y) * blockDim.x + threadIdx.x; cell < 3 * row.count;
       cell += int64_t(gridDim.y) * blockDim.x) {
    const auto field = cell / row.count, index = cell % row.count;
    copies.target[field][row.output + index] = copies.source[field][row.input + index];
  }
}
}

bool prefix_inputs_supported(const at::Tensor& ids,
                             const at::Tensor& positions,
                             const at::Tensor& types) {
  if (!ids.defined() || !ids.is_cuda())
    return false;
  for (const auto& tensor : {ids, positions, types})
    if (!tensor.defined() || tensor.device() != ids.device() || tensor.scalar_type() != at::kLong ||
        tensor.dim() != 1 || !tensor.is_contiguous() || tensor.numel() != ids.numel() ||
        tensor.is_neg() || tensor.is_conj())
      return false;
  return true;
}

/// Validate host full/suffix boundaries, then gather three token vectors into one owned allocation.
/// Returned vectors are disjoint views of that allocation and do not retain full-input storage.
PrefixInputs pack_prefix_inputs(const at::Tensor& ids,
                                const at::Tensor& positions,
                                const at::Tensor& types,
                                std::span<const int32_t> full,
                                std::span<const int32_t> suffix,
                                std::span<const int64_t> hits) {
  TORCH_CHECK(prefix_inputs_supported(ids, positions, types), "Unsupported prefix input layout");
  TORCH_CHECK(!hits.empty() && full.size() == hits.size() + 1 && suffix.size() == full.size() &&
                  full.front() == 0 && suffix.front() == 0 && full.back() == ids.numel() &&
                  suffix.back() > 0,
              "Invalid prefix input boundaries");
  for (size_t row = 0; row < hits.size(); ++row) {
    const auto count = int64_t(full[row + 1]) - full[row];
    TORCH_CHECK(count > 0 && hits[row] >= 0 && hits[row] < count &&
                    int64_t(suffix[row + 1]) - suffix[row] == count - hits[row],
                "Invalid prefix input suffix");
  }
  c10::cuda::CUDAGuard guard(ids.device());
  auto owned = at::empty({3, suffix.back()}, ids.options());
  PrefixInputs result{owned.select(0, 0), owned.select(0, 1), owned.select(0, 2)};
  InputCopies copies;
  copies.source[0] = ids.const_data_ptr<int64_t>();
  copies.source[1] = positions.const_data_ptr<int64_t>();
  copies.source[2] = types.const_data_ptr<int64_t>();
  copies.target[0] = result.ids.mutable_data_ptr<int64_t>();
  copies.target[1] = result.positions.mutable_data_ptr<int64_t>();
  copies.target[2] = result.types.mutable_data_ptr<int64_t>();
  const auto stream = at::cuda::getCurrentCUDAStream(ids.get_device());
  for (size_t first = 0; first < hits.size(); first += batch_limit) {
    const auto count = std::min<size_t>(batch_limit, hits.size() - first);
    int64_t maximum = 0;
    for (size_t row = 0; row < count; ++row) {
      const auto input_row = first + row;
      copies.rows[row] = {int64_t(full[input_row]) + hits[input_row],
                          suffix[input_row],
                          int64_t(suffix[input_row + 1]) - suffix[input_row]};
      maximum = std::max(maximum, copies.rows[row].count);
    }
    const auto blocks = std::min<int64_t>((3 * maximum + threads - 1) / threads, 65535);
    gather_prefix_inputs<<<dim3(count, blocks), threads, 0, stream>>>(copies);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}
}
