// SPDX-License-Identifier: Apache-2.0
/// \file
/// CUDA construction of sequence-local disentangled-attention bias tiles.
/// Combines content/position scores using packed relative-position indices.

#include "encoder_bias.h"
#include <limits>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace infinity {
template <class T>
__global__ void packed_bias_fill_kernel(T* out,
                                        const T* ck,
                                        const T* pc,
                                        const int64_t* indices,
                                        const int64_t* index_offsets,
                                        const int32_t* cumulative,
                                        int64_t heads,
                                        int64_t total,
                                        int64_t bins,
                                        int64_t max_sequence,
                                        int64_t stride,
                                        int64_t index_count) {
  const int64_t sequence = blockIdx.y, start = cumulative[sequence],
                n = cumulative[sequence + 1] - start;
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  CUDA_KERNEL_ASSERT(start >= 0 && n > 0 && n <= max_sequence && start + n <= total);
  if (idx >= heads * n * n)
    return;
  const int64_t head = idx / (n * n), row = (idx % (n * n)) / n, col = idx % n;
  CUDA_KERNEL_ASSERT(!index_offsets || (index_offsets[sequence] >= 0 &&
                                        index_offsets[sequence] <= index_count - n * n));
  const int64_t bucket = (ck || pc)
                             ? indices[index_offsets ? index_offsets[sequence] + row * n + col
                                                     : row * max_sequence + col]
                             : 0;
  CUDA_KERNEL_ASSERT(!(ck || pc) || (bucket >= 0 && bucket < bins));
  T value = T(0.f);
  // Preserve the original model-dtype zero+c2p, then +p2c rounding.
  if (ck)
    value = __hadd(value, ck[(head * total + start + row) * bins + bucket]);
  if (pc)
    value = __hadd(value, pc[(head * total + start + col) * bins + bucket]);
  out[(sequence * heads + head) * max_sequence * stride + row * stride + col] = value;
}

// Load position-content along contiguous bucket coordinates, then use a shared
// tile to retain coalesced content-key reads and output stores. Swizzle the two
// 32x32 tiles rather than adding bank-padding columns or token rows.
__device__ __forceinline__ int bucket_slot(int row, int col) {
  return row * 32 + (col ^ row);
}

__device__ __forceinline__ int position_slot(int row, int col) {
  return row * 32 + (col ^ ((row & 15) * 2));
}

template <class T>
__global__ void packed_bias_fill_tiled_kernel(T* out,
                                              const T* ck,
                                              const T* pc,
                                              const int64_t* indices,
                                              const int64_t* index_offsets,
                                              const int32_t* cumulative,
                                              int64_t heads,
                                              int64_t total,
                                              int64_t bins,
                                              int64_t max_sequence,
                                              int64_t stride,
                                              int64_t index_count) {
  __shared__ int32_t bucket_tile[32 * 32];
  __shared__ T position_tile[32 * 32];
  const int64_t sequence = blockIdx.y, head = blockIdx.z, start = cumulative[sequence];
  const int64_t n = cumulative[sequence + 1] - start;
  const int64_t tiles = (max_sequence + 31) / 32;
  const int64_t row_start = (blockIdx.x / tiles) * 32, col_start = (blockIdx.x % tiles) * 32;
  const int x = threadIdx.x, y = threadIdx.y;
  CUDA_KERNEL_ASSERT(start >= 0 && n > 0 && n <= max_sequence && start + n <= total);
  if (row_start >= n || col_start >= n)
    return; // Uniform CTA exit before either barrier.
  CUDA_KERNEL_ASSERT(!index_offsets || (index_offsets[sequence] >= 0 &&
                                        index_offsets[sequence] <= index_count - n * n));
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = y + 8 * i, col = x;
    if (row_start + row < n && col_start + col < n) {
      const int64_t bucket =
          indices[index_offsets ? index_offsets[sequence] + (row_start + row) * n + col_start + col
                                : (row_start + row) * max_sequence + col_start + col];
      CUDA_KERNEL_ASSERT(bucket >= 0 && bucket < bins);
      bucket_tile[bucket_slot(row, col)] = int32_t(bucket);
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = x, col = y + 8 * i;
    if (row_start + row < n && col_start + col < n) {
      const int32_t bucket = bucket_tile[bucket_slot(row, col)];
      position_tile[position_slot(row, col)] =
          pc[(head * total + start + col_start + col) * bins + bucket];
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = y + 8 * i, col = x;
    if (row_start + row < n && col_start + col < n) {
      const int32_t bucket = bucket_tile[bucket_slot(row, col)];
      T value = T(0.f);
      // Preserve zero+c2p and then+p2c, with the original model-dtype rounding.
      if (ck)
        value = __hadd(value, ck[(head * total + start + row_start + row) * bins + bucket]);
      value = __hadd(value, position_tile[position_slot(row, col)]);
      out[(sequence * heads + head) * max_sequence * stride + (row_start + row) * stride +
          col_start + col] = value;
    }
  }
}

/// Write sequence-local disentangled bias tiles into caller-owned strided storage. The host
/// validates geometry; device indices select relative terms without transferring token rows.
void packed_bias_fill(const at::Tensor& out,
                      const at::Tensor& ck,
                      const at::Tensor& pc,
                      const at::Tensor& indices,
                      const at::Tensor& index_offsets,
                      const at::Tensor& cumulative,
                      int64_t max_sequence) {
  TORCH_CHECK(out.is_cuda() &&
                  (out.scalar_type() == at::kHalf || out.scalar_type() == at::kBFloat16),
              "Packed relative bias requires CUDA Half/BFloat16");
  TORCH_CHECK(out.dim() == 4 && out.size(0) == 1 && out.size(1) > 0 && out.size(2) > 0 &&
                  out.size(2) == out.size(3) && out.storage_offset() == 0,
              "Invalid packed relative bias shape");
  const int64_t heads = out.size(1), total = out.size(2);
  TORCH_CHECK(max_sequence > 0 && max_sequence <= total &&
                  max_sequence <= std::numeric_limits<int32_t>::max() - 7,
              "Invalid packed relative bias maximum length");
  const int64_t stride = ((max_sequence + 7) / 8) * 8;
  auto multiply = [](int64_t a, int64_t b) {
    TORCH_CHECK(a >= 0 && b >= 0 && (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
                "Packed relative bias size overflow");
    return a * b;
  };
  auto add = [](int64_t a, int64_t b) {
    TORCH_CHECK(a <= std::numeric_limits<int64_t>::max() - b, "Packed relative bias size overflow");
    return a + b;
  };
  const int64_t tile = multiply(max_sequence, stride), batch_stride = multiply(heads, tile);
  TORCH_CHECK(cumulative.dim() == 1 && cumulative.numel() > 1 && cumulative.numel() <= 65536,
              "Invalid packed relative bias batch");
  const int64_t batch = cumulative.numel() - 1;
  TORCH_CHECK(out.stride(0) == batch_stride && out.stride(1) == tile && out.stride(2) == stride &&
                  out.stride(3) == 1,
              "Invalid packed relative bias strides");
  const int64_t actual = multiply(batch, batch_stride),
                logical = add(add(multiply(heads - 1, tile), multiply(total - 1, stride)), total);
  TORCH_CHECK(uint64_t(std::max(actual, logical)) <= out.storage().nbytes() / out.element_size(),
              "Packed relative bias storage is too small");
  const int64_t work = multiply(heads, multiply(max_sequence, max_sequence));
  TORCH_CHECK(work <= int64_t(std::numeric_limits<int32_t>::max()) * 256 - 255,
              "Packed relative bias launch is too large");
  c10::cuda::CUDAGuard guard(out.device());
  const auto& source = ck.defined() ? ck : pc;
  const int64_t bins = source.defined() && source.dim() == 3 ? source.size(2) : 0;
  for (const auto* t : {&ck, &pc})
    if (t->defined())
      TORCH_CHECK(t->dim() == 3 && t->size(0) == heads && t->size(1) == total &&
                      t->size(2) == bins && bins > 0 && t->device() == out.device() &&
                      t->scalar_type() == out.scalar_type() && t->is_contiguous(),
                  "Packed relative bias source mismatch");
  TORCH_CHECK(indices.device() == out.device() && indices.scalar_type() == at::kLong &&
                  indices.is_contiguous() && cumulative.device() == out.device() &&
                  cumulative.scalar_type() == at::kInt && cumulative.is_contiguous(),
              "Packed relative bias metadata mismatch");
  if (indices.dim() == 2) {
    TORCH_CHECK(indices.size(0) == max_sequence && indices.size(1) == max_sequence &&
                    !index_offsets.defined(),
                "Shared relative bias table must match the real maximum sequence");
  } else {
    TORCH_CHECK(indices.dim() == 1 && index_offsets.defined() && index_offsets.dim() == 1 &&
                    index_offsets.device() == out.device() &&
                    index_offsets.scalar_type() == at::kLong && index_offsets.is_contiguous() &&
                    index_offsets.numel() == batch + 1,
                "Packed relative bias offsets mismatch");
    TORCH_CHECK(indices.numel() >= total, "Packed relative bias indices are too short");
  }

  auto stream = at::cuda::getCurrentCUDAStream(out.device().index());
  dim3 grid((heads * max_sequence * max_sequence + 255) / 256, batch);
#define RUN(T)                                                                                     \
  packed_bias_fill_kernel<T><<<grid, 256, 0, stream>>>(                                            \
      reinterpret_cast<T*>(out.mutable_data_ptr()),                                                \
      ck.defined() ? reinterpret_cast<const T*>(ck.const_data_ptr()) : nullptr,                    \
      pc.defined() ? reinterpret_cast<const T*>(pc.const_data_ptr()) : nullptr,                    \
      indices.const_data_ptr<int64_t>(),                                                           \
      index_offsets.defined() ? index_offsets.const_data_ptr<int64_t>() : nullptr,                 \
      cumulative.const_data_ptr<int32_t>(),                                                        \
      heads,                                                                                       \
      total,                                                                                       \
      bins,                                                                                        \
      max_sequence,                                                                                \
      stride,                                                                                      \
      indices.numel())
  const bool tiled = pc.defined() && max_sequence >= 128 &&
                     bins <= std::numeric_limits<int32_t>::max() && heads <= 65535;
  const int64_t tile_count = (max_sequence + 31) / 32;
  dim3 tiled_grid(tile_count * tile_count, batch, heads);
#define RUN_TILED(T)                                                                               \
  packed_bias_fill_tiled_kernel<T><<<tiled_grid, dim3(32, 8), 0, stream>>>(                        \
      reinterpret_cast<T*>(out.mutable_data_ptr()),                                                \
      ck.defined() ? reinterpret_cast<const T*>(ck.const_data_ptr()) : nullptr,                    \
      reinterpret_cast<const T*>(pc.const_data_ptr()),                                             \
      indices.const_data_ptr<int64_t>(),                                                           \
      index_offsets.defined() ? index_offsets.const_data_ptr<int64_t>() : nullptr,                 \
      cumulative.const_data_ptr<int32_t>(),                                                        \
      heads,                                                                                       \
      total,                                                                                       \
      bins,                                                                                        \
      max_sequence,                                                                                \
      stride,                                                                                      \
      indices.numel())
  if (out.scalar_type() == at::kHalf) {
    if (tiled) {
      RUN_TILED(__half);
    } else {
      RUN(__half);
    }
  } else {
    if (tiled) {
      RUN_TILED(__nv_bfloat16);
    } else {
      RUN(__nv_bfloat16);
    }
  }
#undef RUN_TILED
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
