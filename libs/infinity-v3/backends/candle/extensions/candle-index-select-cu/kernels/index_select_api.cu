/* SPDX-License-Identifier: Apache-2.0 OR MIT
 * Copyright (c) 2025 Michael Feil
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
 * http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.

 * Authors explaination: Provide a copy of the first two lines in each
 redistributed version.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

// =======================================
// 1. Scalar row-wise kernel (fallback)
// =======================================
template <typename T>
__global__ void index_select_rows_scalar(const T *__restrict__ x,
                                         const uint32_t *__restrict__ idx,
                                         T *__restrict__ out, uint32_t rows,
                                         uint32_t cols, uint64_t source_row_stride, uint32_t index_count) {
  uint32_t j = blockIdx.x;
  const uint32_t j_stride = gridDim.x;

  for (; j < index_count; j += j_stride) {
    const uint32_t r = idx[j];
    const T *__restrict__ src_row = x + (size_t)r * source_row_stride;
    T *__restrict__ dst_row = out + (size_t)j * cols;

    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
      dst_row[c] = src_row[c];
    }
  }
}

// =======================================
// 2. Vectorized row-wise kernel (float4 / half2)
// =======================================
template <typename T, int Vec> struct VecType;

template <> struct VecType<float, 4> { using Type = float4; };

template <> struct VecType<__half, 2> { using Type = __half2; };

template <typename T, int Vec>
__global__ void index_select_rows_vec(const T *__restrict__ x,
                                      const uint32_t *__restrict__ idx,
                                      T *__restrict__ out, uint32_t rows,
                                      uint32_t cols, uint64_t source_row_stride, uint32_t index_count) {
  using VecT = typename VecType<T, Vec>::Type;

  const uint32_t cols_vec = cols / Vec;
  uint32_t j = blockIdx.x;
  const uint32_t j_stride = gridDim.x;

  const VecT *__restrict__ x_vec = reinterpret_cast<const VecT *>(x);
  VecT *__restrict__ out_vec = reinterpret_cast<VecT *>(out);

  for (; j < index_count; j += j_stride) {
    const uint32_t r = idx[j];

    const VecT *__restrict__ src_row = x_vec + (size_t)r * (source_row_stride / Vec);
    VecT *__restrict__ dst_row = out_vec + (size_t)j * cols_vec;

    for (uint32_t c_vec = threadIdx.x; c_vec < cols_vec; c_vec += blockDim.x) {
      dst_row[c_vec] = src_row[c_vec];
    }
  }
}

// =======================================
// 3. Warp-specialized half2 kernel
// =======================================
//
// Each warp handles one output row j:
//   - warp_id in block -> which row this warp owns
//   - lane_id -> which inner index within the row
//
// We operate on half2, so we treat cols as cols_vec = cols / 2.
//
template <int WARPS_PER_BLOCK>
__global__ void index_select_rows_half2_warp(const __half *__restrict__ x,
                                             const uint32_t *__restrict__ idx,
                                             __half *__restrict__ out,
                                             uint32_t rows, uint32_t cols,
                                             uint64_t source_row_stride, uint32_t index_count) {
  const uint32_t cols_vec = cols / 2; // number of half2 per row
  const int threads_per_warp = 32;
  const int warps_per_block = WARPS_PER_BLOCK;
  const int lanes_per_block = warps_per_block * threads_per_warp;

  const int lane_id = threadIdx.x % threads_per_warp;
  const int warp_id = threadIdx.x / threads_per_warp; // 0 .. WARPS_PER_BLOCK-1

  if (threadIdx.x >= lanes_per_block)
    return; // safety if blockDim.x > lanes_per_block

  const uint32_t rows_per_block = warps_per_block;

  // Work in half2 units.
  const __half2 *__restrict__ x_vec = reinterpret_cast<const __half2 *>(x);
  __half2 *__restrict__ out_vec = reinterpret_cast<__half2 *>(out);

  // Global row index this warp starts with
  uint32_t j = blockIdx.x * rows_per_block + warp_id;

  while (j < index_count) {
    const uint32_t r = idx[j];
    // Optional debug: if (r >= rows) return;

    const __half2 *__restrict__ src_row = x_vec + (size_t)r * (source_row_stride / 2);
    __half2 *__restrict__ dst_row = out_vec + (size_t)j * cols_vec;

    // Each lane walks across the row in steps of warp size (32).
    // This gives fully coalesced loads/stores: lane 0 loads [0, 32, 64, ...],
    // lane 1 [1, 33, 65, ...], etc.
    for (uint32_t c_vec = lane_id; c_vec < cols_vec;
         c_vec += threads_per_warp) {
#pragma unroll 4
      for (uint32_t u = 0; u < 1; ++u) {
        uint32_t idx_vec = c_vec + u * threads_per_warp;
        if (idx_vec < cols_vec) {
          dst_row[idx_vec] = src_row[idx_vec];
        }
      }
    }

    // Next row this warp will handle (grid-stride in units of rows_per_block)
    j += gridDim.x * rows_per_block;
  }
}

// =======================================
// 4. Launch config helpers
// =======================================
static inline void compute_launch_config_rows_scalar(uint32_t index_count,
                                                     uint32_t /*cols*/,
                                                     int multi_processor_count,
                                                     dim3 &grid, dim3 &block) {
  const int threads = 256;
  int max_blocks =
      multi_processor_count > 0 ? multi_processor_count * 16 : 1024;
  int blocks = (int)index_count;
  if (blocks > max_blocks)
    blocks = max_blocks;
  if (blocks < 1)
    blocks = 1;
  grid = dim3(blocks, 1, 1);
  block = dim3(threads, 1, 1);
}

// For the warp-specialized half2 kernel (with oversubscription)
template <int WARPS_PER_BLOCK>
static inline void compute_launch_config_rows_half2(uint32_t index_count,
                                                    int multi_processor_count,
                                                    dim3 &grid, dim3 &block) {
  const int threads = WARPS_PER_BLOCK * 32;
  int sm_count = multi_processor_count > 0 ? multi_processor_count : 80;

  int max_blocks = sm_count * 8;
  if (max_blocks < 1)
    max_blocks = 1;

  // Warps needed to cover all rows once
  uint64_t warps_needed = (uint64_t)index_count;
  int blocks_needed =
      (int)((warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  // For small index_count, oversubscribe to max_blocks;
  // warps with j >= index_count just exit immediately.
  int blocks = blocks_needed;
  if (blocks < max_blocks)
    blocks = max_blocks;
  if (blocks > max_blocks)
    blocks = max_blocks;

  grid = dim3(blocks, 1, 1);
  block = dim3(threads, 1, 1);
}

// =======================================
// 5. Type-specific launcher
// =======================================
template <typename T>
static void launch_index_select(const void *x, const uint32_t *idx, void *dst,
                                uint32_t rows, uint32_t cols,
                                uint64_t source_row_stride, uint32_t index_count,
                                int multi_processor_count, cudaStream_t stream) {
  if (index_count == 0 || rows == 0 || cols == 0) {
    return;
  }

  const T *x_t = static_cast<const T *>(x);
  T *dst_t = static_cast<T *>(dst);

  // ---------- float path: float4 vectorization ----------
  if constexpr (std::is_same<T, float>::value) {
    dim3 grid, block;
    compute_launch_config_rows_scalar(index_count, cols, multi_processor_count,
                                      grid, block);

    const uintptr_t x_addr = reinterpret_cast<uintptr_t>(x_t);
    const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_t);
    const bool aligned = ((x_addr % alignof(float4)) == 0) &&
                         ((dst_addr % alignof(float4)) == 0);

    if (aligned && (cols % 4 == 0) && (source_row_stride % 4 == 0)) {
      index_select_rows_vec<float, 4>
          <<<grid, block, 0, stream>>>(x_t, idx, dst_t, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
      {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("index_select_rows_vec<float,4> launch failed: %s\n",
                 cudaGetErrorString(err));
        }
      }
#endif
      return;
    }

    // Scalar fallback
    index_select_rows_scalar<T>
        <<<grid, block, 0, stream>>>(x_t, idx, dst_t, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
    {
      const cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("index_select_rows_scalar<float> launch failed: %s\n",
               cudaGetErrorString(err));
      }
    }
#endif
    return;
  }

  // ---------- __half path: warp-specialized half2 + half2-row + scalar
  // fallback ----------
  if constexpr (std::is_same<T, __half>::value) {
    const bool even_cols = (cols % 2 == 0) && (source_row_stride % 2 == 0)
        && (reinterpret_cast<uintptr_t>(x_t) % alignof(__half2) == 0);
    const uint64_t cols_vec = cols / 2;
    const uint64_t work_items = (uint64_t)index_count * cols_vec;

    // Heuristic: only use warp-specialized kernel for "big" workloads.
    // (Tune this threshold based on your benchmarks.)
    const bool big_enough = even_cols && (work_items >= (uint64_t)1 << 22);

    if (big_enough) {
      constexpr int WARPS_PER_BLOCK = 4; // 4 warps -> 128 threads
      dim3 grid, block;
      compute_launch_config_rows_half2<WARPS_PER_BLOCK>(
          index_count, multi_processor_count, grid, block);

      const __half *x_h = static_cast<const __half *>(x);
      __half *dst_h = static_cast<__half *>(dst);

      index_select_rows_half2_warp<WARPS_PER_BLOCK>
          <<<grid, block, 0, stream>>>(x_h, idx, dst_h, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
      {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          printf("index_select_rows_half2_warp launch failed: %s\n",
                 cudaGetErrorString(err));
        }
      }
#endif
      return;
    }

    // Medium/small, but even cols: half2 row kernel (like float4 path)
    if (even_cols) {
      dim3 grid, block;
      compute_launch_config_rows_scalar(index_count, cols,
                                        multi_processor_count, grid, block);

      const __half *x_h = static_cast<const __half *>(x);
      __half *dst_h = static_cast<__half *>(dst);

      using VecT = typename VecType<__half, 2>::Type;
      const uintptr_t x_addr = reinterpret_cast<uintptr_t>(x_h);
      const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst_h);
      const bool aligned =
          ((x_addr % alignof(VecT)) == 0) && ((dst_addr % alignof(VecT)) == 0);

      if (aligned) {
        index_select_rows_vec<__half, 2>
            <<<grid, block, 0, stream>>>(x_h, idx, dst_h, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
        {
          const cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
            printf("index_select_rows_vec<__half,2> launch failed: %s\n",
                   cudaGetErrorString(err));
          }
        }
#endif
        return;
      }

      // misaligned -> fall through to scalar
    }

    // Odd cols or misaligned -> scalar fallback
    dim3 grid, block;
    compute_launch_config_rows_scalar(index_count, cols, multi_processor_count,
                                      grid, block);

    index_select_rows_scalar<T>
        <<<grid, block, 0, stream>>>(x_t, idx, dst_t, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
    {
      const cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("index_select_rows_scalar<__half> launch failed: %s\n",
               cudaGetErrorString(err));
      }
    }
#endif
    return;
  }

  // ---------- generic T (shouldn't really be used) ----------
  dim3 grid, block;
  compute_launch_config_rows_scalar(index_count, cols, multi_processor_count,
                                    grid, block);
  index_select_rows_scalar<T>
      <<<grid, block, 0, stream>>>(x_t, idx, dst_t, rows, cols, source_row_stride, index_count);
#ifdef DEBUG
  {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("index_select_rows_scalar<T> launch failed: %s\n",
             cudaGetErrorString(err));
    }
  }
#endif
}

// =======================================
// 6. C entry point
// =======================================
extern "C" void run_index_select(const void *x, const uint32_t *indices,
                                 void *dst, uint32_t rows, uint32_t cols,
                                 uint64_t source_row_stride, uint32_t index_count,
                                 int multi_processor_count,
                                 uint32_t dtype_code, cudaStream_t stream) {
  // dtype_code: 0 = f16, 1 = f32
  switch (dtype_code) {
  case 0:
    launch_index_select<__half>(x, indices, dst, rows, cols, source_row_stride, index_count,
                                multi_processor_count, stream);
    break;
  case 1:
    launch_index_select<float>(x, indices, dst, rows, cols, source_row_stride, index_count,
                               multi_processor_count, stream);
    break;
  default:
    // Unsupported dtype; Rust side should have rejected already.
    break;
  }
}
