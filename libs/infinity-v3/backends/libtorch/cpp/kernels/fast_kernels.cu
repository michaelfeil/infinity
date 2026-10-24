// SPDX-License-Identifier: Apache-2.0
// The included Q/K normalization kernel retains its original MIT notice.
/// \file
/// Shared CUDA implementations for normalization, activation, rotary, and routing.
/// Model adapters select the operation and its rounding contract through these wrappers.

#include "kernels/fast_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "../../../candle/extensions/candle-layer-norm/kernels/ln_fwd_kernels.cuh"
// Candle emits PTX directly and guards its BF16 kernel by __CUDA_ARCH__.
// LibTorch builds a host CUDA translation unit, which also needs the declaration.
extern "C" __global__ void qk_norm_rope_bf16(const __nv_bfloat16*,
                                             const __nv_bfloat16*,
                                             const __nv_bfloat16*,
                                             const __nv_bfloat16*,
                                             const __nv_bfloat16*,
                                             const __nv_bfloat16*,
                                             __nv_bfloat16*,
                                             int,
                                             int,
                                             int,
                                             int,
                                             int,
                                             float);
#include "../../../candle/src/kernels/qk_norm_rope.cu"

namespace infinity {
namespace {
#include "kernels/prefix_qkv_norm_rope.cuh"

// Eight-way descending bitonic network derived from Candle's sort.cu:
// https://github.com/michaelfeil/candle/blob/036f39e44015e78ba18d716d73dfdc2e78730c44/candle-kernels/src/sort.cu
// Original source: Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
// This derived portion is licensed under MIT; retained permission notice:
/*
 * Permission is hereby granted, free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the
 * Software without restriction, including without
 * limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software
 * is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice
 * shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
 * ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
 * SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
__global__ void nomic_topk8_kernel(
    const float* probabilities, float* scores, int64_t* indices, int64_t rows, int k) {
  for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
       row += int64_t(gridDim.x) * blockDim.x) {
    int order[8];
#pragma unroll
    for (int col = 0; col < 8; ++col)
      order[col] = col;
#pragma unroll
    for (int width = 2; width <= 8; width *= 2) {
#pragma unroll
      for (int stride = width / 2; stride > 0; stride /= 2) {
#pragma unroll
        for (int col = 0; col < 8; ++col) {
          const int partner = col ^ stride;
          if (partner > col) {
            const float left = probabilities[row * 8 + order[col]];
            const float right = probabilities[row * 8 + order[partner]];
            if (((col & width) == 0 && left < right) || ((col & width) != 0 && left > right)) {
              const int tmp = order[col];
              order[col] = order[partner];
              order[partner] = tmp;
            }
          }
        }
      }
    }
    for (int col = 0; col < k; ++col) {
      indices[row * k + col] = order[col];
      scores[row * k + col] = probabilities[row * 8 + order[col]];
    }
  }
}

__global__ void qk_bf16_bridge(const __nv_bfloat16* q,
                               const __nv_bfloat16* k,
                               const __nv_bfloat16* qw,
                               const __nv_bfloat16* kw,
                               const __nv_bfloat16* cosine,
                               const __nv_bfloat16* sine,
                               __nv_bfloat16* out,
                               int tokens,
                               int qheads,
                               int kheads,
                               int qstride,
                               int kstride,
                               float eps) {
  ::fused_qk_norm_rope(
      q, k, qw, kw, cosine, sine, out, tokens, qheads, kheads, qstride, kstride, eps);
}

__device__ float block_sum(float value) {
  __shared__ float warps[8];
  const auto lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int d = 16; d; d /= 2)
    value += __shfl_down_sync(0xffffffff, value, d);
  if (!lane)
    warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warps[lane] : 0.f;
  if (warp == 0)
    for (int d = 16; d; d /= 2)
      value += __shfl_down_sync(0xffffffff, value, d);
  if (!threadIdx.x)
    warps[0] = value;
  __syncthreads();
  float total = warps[0];
  __syncthreads();
  return total;
}

template <class T>
__global__ void add_norm_kernel(const T* x,
                                const T* residual,
                                const T* second,
                                const T* weight,
                                const T* bias,
                                T* out,
                                int64_t width,
                                float epsilon) {
  const int64_t row = blockIdx.x;
  float sum = 0;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    // Match eager tensor additions: round each intermediate to the model dtype.
    T value = T(float(x[row * width + col]) + float(residual[row * width + col]));
    if (second)
      value = T(float(value) + float(second[row * width + col]));
    sum += float(value);
  }
  const float mean = block_sum(sum) / width;
  float variance = 0;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    T value = T(float(x[row * width + col]) + float(residual[row * width + col]));
    if (second)
      value = T(float(value) + float(second[row * width + col]));
    float delta = float(value) - mean;
    variance += delta * delta;
  }
  const float inverse = rsqrtf(block_sum(variance) / width + epsilon);
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    T value = T(float(x[row * width + col]) + float(residual[row * width + col]));
    if (second)
      value = T(float(value) + float(second[row * width + col]));
    float normalized = (float(value) - mean) * inverse;
    out[row * width + col] =
        T(normalized * (weight ? float(weight[col]) : 1.f) + (bias ? float(bias[col]) : 0.f));
  }
}

template <class T, int Columns>
__global__ __launch_bounds__(256) void add_norm_register_kernel(const T* x,
                                                                const T* residual,
                                                                const T* second,
                                                                const T* weight,
                                                                const T* bias,
                                                                T* out,
                                                                int64_t width,
                                                                float epsilon) {
  const int64_t row = blockIdx.x;
  float values[Columns], sum = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    values[j] = 0;
    if (col < width) {
      T value = T(float(x[row * width + col]) + float(residual[row * width + col]));
      if (second)
        value = T(float(value) + float(second[row * width + col]));
      values[j] = float(value);
      sum += values[j];
    }
  }
  const float mean = block_sum(sum) / width;
  float variance = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j)
    if (threadIdx.x + j * 256 < width) {
      float delta = values[j] - mean;
      variance += delta * delta;
    }
  const float inverse = rsqrtf(block_sum(variance) / width + epsilon);
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    if (col < width)
      out[row * width + col] =
          T(((values[j] - mean) * inverse) * (weight ? float(weight[col]) : 1.f) +
            (bias ? float(bias[col]) : 0.f));
  }
}

template <class T, int Columns>
__global__ __launch_bounds__(256) void gather_norm_kernel(const int64_t* ids,
                                                          const int64_t* types,
                                                          const int64_t* positions,
                                                          const T* words,
                                                          const T* type_table,
                                                          const T* position_table,
                                                          const T* weight,
                                                          const T* bias,
                                                          T* out,
                                                          int64_t width,
                                                          float epsilon) {
  const int64_t row = blockIdx.x;
  float values[Columns], sum = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    values[j] = 0;
    if (col < width) {
      T value =
          T(float(words[ids[row] * width + col]) + float(type_table[types[row] * width + col]));
      value = T(float(value) + float(position_table[positions[row] * width + col]));
      values[j] = float(value);
      sum += values[j];
    }
  }
  const float mean = block_sum(sum) / width;
  float variance = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j)
    if (threadIdx.x + j * 256 < width) {
      float delta = values[j] - mean;
      variance += delta * delta;
    }
  const float inverse = rsqrtf(block_sum(variance) / width + epsilon);
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    if (col < width)
      out[row * width + col] =
          T(((values[j] - mean) * inverse) * (weight ? float(weight[col]) : 1.f) +
            (bias ? float(bias[col]) : 0.f));
  }
}

template <class T, int Columns>
__global__ void add_rms_kernel(const T* x,
                               const T* residual,
                               const T* weight,
                               T* out,
                               T* summed,
                               int64_t width,
                               float epsilon) {
  const int64_t row = blockIdx.x;
  float variance = 0;
  float values[Columns];
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    float value = col < width ? float(x[row * width + col]) : 0.f;
    if (col < width && residual)
      value += float(residual[row * width + col]);
    // Candle normalizes the unrounded FP32 sum, while its residual output
    // separately stores that sum in model dtype (round_residual=false).
    values[j] = value;
    variance += value * value;
    if (col < width && residual)
      summed[row * width + col] = T(value);
  }
  float inverse = rsqrtf(block_sum(variance) / width + epsilon);
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    auto col = threadIdx.x + j * 256;
    if (col < width)
      out[row * width + col] = T((values[j] * inverse) * float(weight[col]));
  }
}

// Vector global transfers retain the scalar kernel's per-thread summation
// order. Shared memory changes the access layout without changing arithmetic.
struct alignas(16) Bf16NormVector {
  __nv_bfloat16 v[8];
};

__global__ void add_rms_bf16_4096_kernel(const __nv_bfloat16* x,
                                         const __nv_bfloat16* residual,
                                         const __nv_bfloat16* weight,
                                         __nv_bfloat16* out,
                                         __nv_bfloat16* summed,
                                         float epsilon) {
  __shared__ float values[4096];
  const int64_t row = blockIdx.x;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int group = threadIdx.x + j * 256;
    auto input = reinterpret_cast<const Bf16NormVector*>(x)[row * 512 + group];
    Bf16NormVector add{}, sum;
    if (residual)
      add = reinterpret_cast<const Bf16NormVector*>(residual)[row * 512 + group];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = float(input.v[i]);
      if (residual)
        value += float(add.v[i]);
      values[group * 8 + i] = value;
      sum.v[i] = __nv_bfloat16(value);
    }
    if (residual)
      reinterpret_cast<Bf16NormVector*>(summed)[row * 512 + group] = sum;
  }
  __syncthreads();
  float variance = 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    float value = values[threadIdx.x + j * 256];
    variance += value * value;
  }
  float inverse = rsqrtf(block_sum(variance) / 4096 + epsilon);
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int group = threadIdx.x + j * 256;
    auto gamma = reinterpret_cast<const Bf16NormVector*>(weight)[group];
    Bf16NormVector result;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      result.v[i] = __nv_bfloat16((values[group * 8 + i] * inverse) * float(gamma.v[i]));
    reinterpret_cast<Bf16NormVector*>(out)[row * 512 + group] = result;
  }
}

template <class T>
__global__ void add_rms_large_kernel(const T* x,
                                     const T* residual,
                                     const T* weight,
                                     T* out,
                                     T* summed,
                                     int64_t width,
                                     float epsilon) {
  const int64_t row = blockIdx.x;
  float variance = 0;
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    float value = float(x[row * width + col]);
    if (residual)
      value += float(residual[row * width + col]);
    if (residual)
      summed[row * width + col] = T(value);
    variance += value * value;
  }
  float inverse = rsqrtf(block_sum(variance) / width + epsilon);
  for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
    float value = float(x[row * width + col]);
    if (residual)
      value += float(residual[row * width + col]);
    out[row * width + col] = T(value * inverse * float(weight[col]));
  }
}

void run_candle_half4096_rms(const at::Tensor& x,
                             const at::Tensor& residual,
                             const at::Tensor& weight,
                             at::Tensor& output,
                             at::Tensor& sum,
                             float epsilon,
                             cudaStream_t stream) {
  // Candle registers Half/half-residual width4096 with one CTA per row and
  // warp layout1x4. Its Welford statistics avoid the generic sum-of-squares
  // reduction's one-Half-ULP divergence on independently captured operands.
  layer_norm::FwdParams p;
  p.rows = x.numel() / 4096;
  p.cols = 4096;
  p.x0 = const_cast<void*>(x.const_data_ptr());
  p.x1 = nullptr;
  p.residual = residual.defined() ? const_cast<void*>(residual.const_data_ptr()) : nullptr;
  p.x = residual.defined() ? sum.mutable_data_ptr() : nullptr;
  p.z = output.mutable_data_ptr();
  p.gamma = const_cast<void*>(weight.const_data_ptr());
  p.beta = nullptr;
  p.dmask = nullptr;
  p.dmask1 = nullptr;
  p.x0_subset = nullptr;
  p.z_subset = nullptr;
  p.epsilon = epsilon;
  p.inverse_cols = 1.f / 4096;
  p.dropout_keep_p = 1.f;
  p.dropout_scale = 1.f;
  p.rowscale_const = 1.f;
  p.round_residual = false;
  p.is_rms_norm = true;
  using Traits = layer_norm::Kernel_traits<half, half, half, half, float, int, 4096, 1, 1, 4, 16>;
  auto kernel = layer_norm::ln_fwd_kernel<Traits, false, false, false, true>;
  static thread_local std::vector<int> occupancy;
  const int device = x.get_device();
  if (occupancy.size() <= static_cast<size_t>(device))
    occupancy.resize(device + 1, -1);
  int& blocks = occupancy[device];
  if (blocks < 0)
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, Traits::THREADS_PER_CTA, Traits::SMEM_BYTES_FWD));
  p.ctas_per_col = at::cuda::getDeviceProperties(device)->multiProcessorCount * blocks;
  kernel<<<p.ctas_per_col, Traits::THREADS_PER_CTA, Traits::SMEM_BYTES_FWD, stream>>>(p);
}

template <class T>
void run_rms(const at::Tensor& x,
             const at::Tensor& residual,
             const at::Tensor& weight,
             at::Tensor& out,
             at::Tensor& sum,
             float epsilon,
             cudaStream_t stream) {
  auto xp = reinterpret_cast<const T*>(x.const_data_ptr());
  auto rp = residual.defined() ? reinterpret_cast<const T*>(residual.const_data_ptr()) : nullptr;
  auto wp = reinterpret_cast<const T*>(weight.const_data_ptr());
  auto op = reinterpret_cast<T*>(out.mutable_data_ptr());
  auto sp = residual.defined() ? reinterpret_cast<T*>(sum.mutable_data_ptr()) : nullptr;
  auto width = x.size(-1), rows = x.numel() / width;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (width == 4096 && rows >= 128 && reinterpret_cast<uintptr_t>(xp) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(wp) % 16 == 0 &&
        (!rp || reinterpret_cast<uintptr_t>(rp) % 16 == 0)) {
      add_rms_bf16_4096_kernel<<<rows, 256, 0, stream>>>(xp, rp, wp, op, sp, epsilon);
      return;
    }
  }

#define TEI_RMS_COLUMNS(C)                                                                         \
  add_rms_kernel<T, C><<<rows, 256, 0, stream>>>(xp, rp, wp, op, sp, width, epsilon)
  if (width <= 256) {
    TEI_RMS_COLUMNS(1);
  } else if (width <= 512) {
    TEI_RMS_COLUMNS(2);
  } else if (width <= 768) {
    TEI_RMS_COLUMNS(3);
  } else if (width <= 1024) {
    TEI_RMS_COLUMNS(4);
  } else if (width <= 2048) {
    TEI_RMS_COLUMNS(8);
  } else if (width <= 4096) {
    TEI_RMS_COLUMNS(16);
  } else if (width <= 8192) {
    TEI_RMS_COLUMNS(32);
  } else
    add_rms_large_kernel<T><<<rows, 256, 0, stream>>>(xp, rp, wp, op, sp, width, epsilon);
#undef TEI_RMS_COLUMNS
}

template <class T>
__global__ void
gelu_kernel(const T* x, const T* bias, T* out, int64_t count, int64_t width, bool approximate) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  T rounded = x[index];
  if (bias)
    rounded = T(float(rounded) + float(bias[index % width]));
  float value = float(rounded);
  float y =
      approximate
          ? 0.5f * value *
                (1.f + tanhf(0.7978845608028654f * (value + 0.044715f * value * value * value)))
          : 0.5f * value * (1.f + erff(value * 0.7071067811865475f));
  out[index] = T(y);
}

template <class T> __device__ __forceinline__ T candle_model_exp(T value) {
  return T(expf(float(value)));
}

template <> __device__ __forceinline__ __half candle_model_exp(__half value) {
  return hexp(value);
}

template <> __device__ __forceinline__ __nv_bfloat16 candle_model_exp(__nv_bfloat16 value) {
  return hexp(value);
}

template <class T> __device__ __forceinline__ T gated_value(float gate, float up, int activation) {
  float activated;
  if (activation == 0)
    activated = gate / (1.f + expf(-gate));
  else if (activation == 5) {
    T value = T(gate), exponential = candle_model_exp(T(-value));
    T denominator = T(T(1) + exponential);
    activated = float(T(value / denominator));
  } else if (activation == 1)
    activated = .5f * gate * (1.f + erff(gate * .7071067811865475f));
  else if (activation == 2)
    activated =
        .5f * gate * (1.f + tanhf(.7978845608028654f * (gate + .044715f * gate * gate * gate)));
  else if (activation == 4) {
    // Match MIT-licensed Candle kernels/gated_activation.cu: each GELU
    // intermediate rounds to model dtype, including the tanh input/value.
    T x = T(gate), square = T(x * x), cube = T(square * x);
    T alpha = T(x + T(.044715) * cube);
    T argument = T(T(.7978845608028654) * alpha);
    T tangent = T(tanhf(float(argument)));
    activated = float(T(T(T(.5) * x) * T(T(1) + tangent)));
  } else
    activated = fmaxf(gate, 0.f);
  return T(float(T(activated)) * up);
}

template <class T>
__global__ void
gated_kernel(const T* x, T* out, int64_t count, int64_t width, int activation, bool gate_first) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  auto row = i / width, col = i % width;
  out[i] = gated_value<T>(float(x[row * 2 * width + col + (gate_first ? 0 : width)]),
                          float(x[row * 2 * width + col + (gate_first ? width : 0)]),
                          activation);
}

// Eight model-dtype lanes use one aligned 128-bit load for FP16/BF16.
// The four-lane and scalar fallbacks preserve exact arithmetic for other views.
template <class T, int Lanes> struct alignas(sizeof(T) * Lanes) GatedVector {
  T value[Lanes];
};
template <class Pair> __device__ Pair model_pair(float value);

template <> __device__ __half2 model_pair(float value) {
  return __float2half2_rn(value);
}

template <> __device__ __nv_bfloat162 model_pair(float value) {
  return __float2bfloat162_rn(value);
}
template <class Pair> __device__ Pair model_pair_tanh(Pair value);

template <> __device__ __half2 model_pair_tanh(__half2 value) {
  return __floats2half2_rn(tanhf(__low2float(value)), tanhf(__high2float(value)));
}

template <> __device__ __nv_bfloat162 model_pair_tanh(__nv_bfloat162 value) {
  return __floats2bfloat162_rn(tanhf(__low2float(value)), tanhf(__high2float(value)));
}

template <class Pair, class Index>
__global__ void
gated_gelu_pairs_kernel(const Pair* x, Pair* out, Index count, Index width, bool gate_first) {
  const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count / 4)
    return;
  const auto vectors_per_row = width / 4, row = index / vectors_per_row,
             col = index % vectors_per_row;
  const auto* input = reinterpret_cast<const GatedVector<Pair, 4>*>(x);
  const auto gate = input[row * 2 * vectors_per_row + col + (gate_first ? 0 : vectors_per_row)];
  const auto up = input[row * 2 * vectors_per_row + col + (gate_first ? vectors_per_row : 0)];
  GatedVector<Pair, 4> result;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    const auto value = gate.value[lane];
    const auto cube = __hmul2(__hmul2(value, value), value);
    const auto alpha = __hfma2(model_pair<Pair>(.044715f), cube, value);
    const auto tangent = model_pair_tanh(__hmul2(model_pair<Pair>(.7978845608028654f), alpha));
    const auto activated =
        __hmul2(__hmul2(model_pair<Pair>(.5f), value), __hadd2(model_pair<Pair>(1.f), tangent));
    result.value[lane] = __hmul2(activated, up.value[lane]);
  }
  reinterpret_cast<GatedVector<Pair, 4>*>(out)[index] = result;
}

// BF16 pair instructions preserve each rounded SiLU intermediate.
template <class Index>
__global__ void gated_silu_bf16_pairs_kernel(const __nv_bfloat162* x,
                                             __nv_bfloat162* out,
                                             Index count,
                                             Index width) {
  const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count / 8)
    return;
  const Index vectors = width / 8, row = index / vectors, col = index % vectors;
  auto gate = reinterpret_cast<const GatedVector<__nv_bfloat162, 4>*>(x)[row * 2 * vectors + col];
  auto up =
      reinterpret_cast<const GatedVector<__nv_bfloat162, 4>*>(x)[row * 2 * vectors + vectors + col];
  GatedVector<__nv_bfloat162, 4> result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    auto exponential = h2exp(__hneg2(gate.value[i]));
    auto denominator = __hadd2(__float2bfloat162_rn(1), exponential);
    result.value[i] = __hmul2(__h2div(gate.value[i], denominator), up.value[i]);
  }
  reinterpret_cast<GatedVector<__nv_bfloat162, 4>*>(out)[index] = result;
}

template <class T, class Index, int Lanes>
__global__ void
gated_vector_kernel(const T* x, T* out, Index count, Index width, int activation, bool gate_first) {
  Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count / Lanes)
    return;
  auto vectors_per_row = width / Lanes, row = index / vectors_per_row,
       col = index % vectors_per_row;
  const auto* input = reinterpret_cast<const GatedVector<T, Lanes>*>(x);
  const auto gate = input[row * 2 * vectors_per_row + col + (gate_first ? 0 : vectors_per_row)];
  const auto up = input[row * 2 * vectors_per_row + col + (gate_first ? vectors_per_row : 0)];
  GatedVector<T, Lanes> result;
#pragma unroll
  for (int lane = 0; lane < Lanes; ++lane)
    result.value[lane] = gated_value<T>(float(gate.value[lane]), float(up.value[lane]), activation);
  reinterpret_cast<GatedVector<T, Lanes>*>(out)[index] = result;
}

template <class T> __device__ __forceinline__ T encoder_rotary_fma(T a, T b, T c) {
  return T(fmaf(float(a), float(b), float(c)));
}

template <> __device__ __forceinline__ __half encoder_rotary_fma(__half a, __half b, __half c) {
  return __hfma(a, b, c);
}

template <>
__device__ __forceinline__ __nv_bfloat16 encoder_rotary_fma(__nv_bfloat16 a,
                                                            __nv_bfloat16 b,
                                                            __nv_bfloat16 c) {
  return __hfma(a, b, c);
}

template <class T>
__global__ void rotary_kernel(const T* q,
                              const T* k,
                              const T* cosine,
                              const T* sine,
                              T* output,
                              int64_t qrows,
                              int64_t krows,
                              int64_t qheads,
                              int64_t kheads,
                              int64_t qstride,
                              int64_t kstride,
                              int64_t dim,
                              int64_t half,
                              bool contract_first_product) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= (qrows + krows) * dim)
    return;
  auto row = i / dim, col = i % dim;
  bool isq = row < qrows;
  auto r = isq ? row : row - qrows, heads = isq ? qheads : kheads;
  auto token = r / heads, head = r % heads;
  const T* x = isq ? q : k;
  auto offset = token * (isq ? qstride : kstride) + head * dim;
  if (col >= 2 * half) {
    output[i] = x[offset + col];
    return;
  }
  bool upper = col >= half;
  auto lane = upper ? col - half : col;
  float a = float(x[offset + lane]), b = float(x[offset + lane + half]),
        c = float(cosine[token * half + lane]), s = float(sine[token * half + lane]);
  // Candle encoder rotary contracts the first half product into HFMA.
  // Decoder callers preserve the separately rounded products by default.
  if (contract_first_product)
    output[i] = upper ? encoder_rotary_fma(T(b), T(c), T(a * s))
                      : encoder_rotary_fma(T(a), T(c), T(-float(T(b * s))));
  else
    output[i] = upper ? T(float(T(b * c)) + float(T(a * s))) : T(float(T(a * c)) - float(T(b * s)));
}

// Each thread owns both rotated lanes. This shares input/frequency loads and
// uses 32-bit division for ordinary tensor sizes without changing rounding.
template <class T>
__global__ void paired_rotary_kernel(const T* q,
                                     const T* k,
                                     const T* cosine,
                                     const T* sine,
                                     T* output,
                                     uint32_t count,
                                     uint32_t qrows,
                                     uint32_t qheads,
                                     uint32_t kheads,
                                     uint32_t qstride,
                                     uint32_t kstride,
                                     uint32_t half,
                                     bool contract_first_product) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const uint32_t row = index / half, lane = index % half;
  const bool isq = row < qrows;
  const uint32_t r = isq ? row : row - qrows, heads = isq ? qheads : kheads, token = r / heads,
                 head = r % heads;
  const uint32_t dim = half * 2, offset = token * (isq ? qstride : kstride) + head * dim + lane;
  const T* x = isq ? q : k;
  const float a = float(x[offset]), b = float(x[offset + half]),
              c = float(cosine[token * half + lane]), s = float(sine[token * half + lane]);
  if (contract_first_product) {
    output[row * dim + lane] = encoder_rotary_fma(T(a), T(c), T(-float(T(b * s))));
    output[row * dim + lane + half] = encoder_rotary_fma(T(b), T(c), T(a * s));
  } else {
    output[row * dim + lane] = T(float(T(a * c)) - float(T(b * s)));
    output[row * dim + lane + half] = T(float(T(b * c)) + float(T(a * s)));
  }
}

// Process two adjacent rotary pairs with packed half arithmetic. Each product
// still rounds to model dtype before the fused multiply-add, as in the scalar path.
template <class Pair, class Index, bool Indexed>
__global__ void encoder_rotary_pairs_kernel(Pair* q,
                                            Pair* k,
                                            const Pair* cosine,
                                            const Pair* sine,
                                            Index qheads,
                                            Index kheads,
                                            Index qstride,
                                            Index kstride,
                                            Index tokens,
                                            Index dim,
                                            Index half,
                                            const int64_t* positions,
                                            Index table_rows) {
  const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * (qheads + kheads) * half)
    return;
  const auto row = index / half, lane = index % half;
  const bool isq = row < tokens * qheads;
  const auto r = isq ? row : row - tokens * qheads, heads = isq ? qheads : kheads;
  const auto token = r / heads, head = r % heads;
  Pair* x = isq ? q : k;
  const auto offset = token * (isq ? qstride : kstride) + head * dim + lane;
  Index frequency = token;
  if constexpr (Indexed) {
    const int64_t position = positions[token];
    CUDA_KERNEL_ASSERT(position >= 0 && uint64_t(position) < uint64_t(table_rows));
    frequency = Index(position);
  }
  const auto a = x[offset], b = x[offset + half], c = cosine[frequency * half + lane],
             s = sine[frequency * half + lane];
  x[offset] = __hfma2(a, c, __hneg2(__hmul2(b, s)));
  x[offset + half] = __hfma2(b, c, __hmul2(a, s));
}

template <class T, class Index, bool Indexed>
__global__ void encoder_rotary_inplace_kernel(T* q,
                                              T* k,
                                              const T* cosine,
                                              const T* sine,
                                              Index qheads,
                                              Index kheads,
                                              Index qstride,
                                              Index kstride,
                                              Index tokens,
                                              Index dim,
                                              Index half,
                                              const int64_t* positions,
                                              Index table_rows) {
  const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= tokens * (qheads + kheads) * half)
    return;
  const auto row = index / half, lane = index % half;
  const bool isq = row < tokens * qheads;
  const auto r = isq ? row : row - tokens * qheads, heads = isq ? qheads : kheads;
  const auto token = r / heads, head = r % heads;
  T* x = isq ? q : k;
  const auto offset = token * (isq ? qstride : kstride) + head * dim + lane;
  Index frequency = token;
  if constexpr (Indexed) {
    const int64_t position = positions[token];
    CUDA_KERNEL_ASSERT(position >= 0 && uint64_t(position) < uint64_t(table_rows));
    frequency = Index(position);
  }
  const T a = x[offset], b = x[offset + half], c = cosine[frequency * half + lane],
          s = sine[frequency * half + lane];
  // Load both original lanes before either write; every pair has one owner.
  x[offset] = encoder_rotary_fma(a, c, T(-float(T(b * s))));
  x[offset + half] = encoder_rotary_fma(b, c, T(a * s));
}

void validate(const at::Tensor& x, const at::Tensor& other, const char* name, bool full) {
  if (!other.defined())
    return;
  TORCH_CHECK(other.device() == x.device() && other.scalar_type() == x.scalar_type() &&
                  other.is_contiguous(),
              name,
              " must be contiguous and match input device/dtype");
  TORCH_CHECK(full ? other.sizes() == x.sizes() : other.numel() == x.size(-1),
              "Invalid ",
              name,
              " dimensions");
}

template <class T>
void run_norm(const at::Tensor& x,
              const at::Tensor& residual,
              const at::Tensor& second,
              const at::Tensor& weight,
              const at::Tensor& bias,
              at::Tensor& output,
              double epsilon,
              cudaStream_t stream) {
  auto p = [](const at::Tensor& t) {
    return t.defined() ? reinterpret_cast<const T*>(t.const_data_ptr()) : nullptr;
  };
  auto width = x.size(-1), rows = x.numel() / width;
#define TEI_REGISTER_LN(N)                                                                         \
  add_norm_register_kernel<T, N>                                                                   \
      <<<rows, 256, 0, stream>>>(p(x),                                                             \
                                 p(residual),                                                      \
                                 p(second),                                                        \
                                 p(weight),                                                        \
                                 p(bias),                                                          \
                                 reinterpret_cast<T*>(output.mutable_data_ptr()),                  \
                                 width,                                                            \
                                 epsilon)
  if (width <= 256) {
    TEI_REGISTER_LN(1);
  } else if (width <= 512) {
    TEI_REGISTER_LN(2);
  } else if (width <= 768) {
    TEI_REGISTER_LN(3);
  } else if (width <= 1024) {
    TEI_REGISTER_LN(4);
  } else if (width <= 1536) {
    TEI_REGISTER_LN(6);
  } else if (width <= 2048) {
    TEI_REGISTER_LN(8);
  } else if (width <= 3072) {
    TEI_REGISTER_LN(12);
  } else if (width <= 4096) {
    TEI_REGISTER_LN(16);
  } else
    add_norm_kernel<T><<<rows, 256, 0, stream>>>(p(x),
                                                 p(residual),
                                                 p(second),
                                                 p(weight),
                                                 p(bias),
                                                 reinterpret_cast<T*>(output.mutable_data_ptr()),
                                                 width,
                                                 epsilon);
#undef TEI_REGISTER_LN
}

template <class T>
void run_gelu(const at::Tensor& x,
              const at::Tensor& bias,
              at::Tensor& output,
              bool approximate,
              cudaStream_t stream) {
  gelu_kernel<T><<<(x.numel() + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const T*>(x.const_data_ptr()),
      bias.defined() ? reinterpret_cast<const T*>(bias.const_data_ptr()) : nullptr,
      reinterpret_cast<T*>(output.mutable_data_ptr()),
      x.numel(),
      x.size(-1),
      approximate);
}
}

at::Tensor fused_add_layer_norm(const at::Tensor& x,
                                const at::Tensor& residual,
                                const at::Tensor& weight,
                                const at::Tensor& bias,
                                double epsilon,
                                const at::Tensor& second) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2 && x.size(-1) > 0,
              "Fused residual layernorm requires contiguous CUDA input");
  TORCH_CHECK(residual.defined(), "Residual must be defined");
  validate(x, residual, "residual", true);
  validate(x, second, "second residual", true);
  validate(x, weight, "weight", false);
  validate(x, bias, "bias", false);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x);
  if (x.numel() == 0)
    return output;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  if (x.scalar_type() == at::kHalf)
    run_norm<__half>(x, residual, second, weight, bias, output, epsilon, stream);
  else if (x.scalar_type() == at::kBFloat16)
    run_norm<__nv_bfloat16>(x, residual, second, weight, bias, output, epsilon, stream);
  else if (x.scalar_type() == at::kFloat)
    run_norm<float>(x, residual, second, weight, bias, output, epsilon, stream);
  else
    TORCH_CHECK(false, "Unsupported fused layernorm dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor fused_embedding_layer_norm(const at::Tensor& ids,
                                      const at::Tensor& types,
                                      const at::Tensor& positions,
                                      const at::Tensor& words,
                                      const at::Tensor& type_table,
                                      const at::Tensor& position_table,
                                      const at::Tensor& weight,
                                      const at::Tensor& bias,
                                      double epsilon) {
  TORCH_CHECK(words.is_cuda() && words.scalar_type() == at::kHalf && words.dim() == 2 &&
                  words.size(1) == 768 && words.is_contiguous(),
              "Embedding fusion requires contiguous CUDA Half hidden768 words");
  for (const auto* table : {&type_table, &position_table})
    TORCH_CHECK(table->device() == words.device() && table->scalar_type() == words.scalar_type() &&
                    table->dim() == 2 && table->size(1) == 768 && table->is_contiguous(),
                "Embedding tables must match CUDA Half hidden768 words");
  for (const auto* index : {&ids, &types, &positions})
    TORCH_CHECK(index->device() == words.device() && index->dim() == 1 &&
                    index->numel() == ids.numel() && index->is_contiguous() &&
                    index->scalar_type() == at::kLong,
                "Embedding indices require matching contiguous device INT64 vectors");
  for (const auto* affine : {&weight, &bias})
    TORCH_CHECK(affine->device() == words.device() &&
                    affine->scalar_type() == words.scalar_type() && affine->dim() == 1 &&
                    affine->numel() == 768 && affine->is_contiguous(),
                "Embedding affine requires matching hidden768 vectors");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0,
              "Embedding epsilon must be finite and nonnegative");
  // IDs are range-validated by the existing Engine host input contract, before capture.
  c10::cuda::CUDAGuard guard(words.device());
  auto out = at::empty({ids.numel(), 768}, words.options());
  if (ids.numel())
    gather_norm_kernel<__half, 3>
        <<<ids.numel(), 256, 0, at::cuda::getCurrentCUDAStream(words.device().index())>>>(
            ids.const_data_ptr<int64_t>(),
            types.const_data_ptr<int64_t>(),
            positions.const_data_ptr<int64_t>(),
            reinterpret_cast<const __half*>(words.const_data_ptr()),
            reinterpret_cast<const __half*>(type_table.const_data_ptr()),
            reinterpret_cast<const __half*>(position_table.const_data_ptr()),
            reinterpret_cast<const __half*>(weight.const_data_ptr()),
            reinterpret_cast<const __half*>(bias.const_data_ptr()),
            reinterpret_cast<__half*>(out.mutable_data_ptr()),
            768,
            epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<at::Tensor, at::Tensor> fused_add_rms_norm(const at::Tensor& x,
                                                     const at::Tensor& residual,
                                                     const at::Tensor& weight,
                                                     double epsilon) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() >= 2 && x.size(-1) > 0,
              "Fused residual RMSNorm requires contiguous CUDA input");
  validate(x, residual, "residual", true);
  validate(x, weight, "weight", false);
  TORCH_CHECK(weight.defined(), "RMSNorm weight must be defined");
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x), sum = residual.defined() ? at::empty_like(x) : x;
  if (x.numel() == 0)
    return {output, sum};
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  if (x.scalar_type() == at::kHalf && x.size(-1) == 4096) {
    run_candle_half4096_rms(x, residual, weight, output, sum, epsilon, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, sum};
  }
#define TEI_RMS_DISPATCH(T) run_rms<T>(x, residual, weight, output, sum, epsilon, stream)
  if (x.scalar_type() == at::kHalf) {
    TEI_RMS_DISPATCH(__half);
  } else if (x.scalar_type() == at::kBFloat16) {
    TEI_RMS_DISPATCH(__nv_bfloat16);
  } else if (x.scalar_type() == at::kFloat) {
    TEI_RMS_DISPATCH(float);
  } else
    TORCH_CHECK(false, "Unsupported fused RMSNorm dtype");
#undef TEI_RMS_DISPATCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, sum};
}

at::Tensor fused_bias_gelu(const at::Tensor& x, const at::Tensor& bias, bool approximate) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() > 0 && x.size(-1) > 0,
              "Fused GELU requires contiguous CUDA input");
  validate(x, bias, "bias", false);
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x);
  if (x.numel() == 0)
    return output;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  if (x.scalar_type() == at::kHalf)
    run_gelu<__half>(x, bias, output, approximate, stream);
  else if (x.scalar_type() == at::kBFloat16)
    run_gelu<__nv_bfloat16>(x, bias, output, approximate, stream);
  else if (x.scalar_type() == at::kFloat)
    run_gelu<float>(x, bias, output, approximate, stream);
  else
    TORCH_CHECK(false, "Unsupported fused GELU dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor fused_gated_activation(const at::Tensor& x, int32_t activation, bool gate_first) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() > 0 && x.size(-1) > 0 &&
                  x.size(-1) % 2 == 0,
              "Fused gated activation requires contiguous CUDA gate/up input");
  TORCH_CHECK(activation >= 0 && activation <= 5, "Unsupported fused gated activation");
  c10::cuda::CUDAGuard guard(x.device());
  auto shape = x.sizes().vec();
  shape.back() /= 2;
  auto output = at::empty(shape, x.options());
  if (output.numel() == 0)
    return output;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  if (activation == 5 && gate_first && x.scalar_type() == at::kBFloat16 && shape.back() % 8 == 0 &&
      reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0) {
#define TEI_SILU_PAIRS(I)                                                                          \
  gated_silu_bf16_pairs_kernel<I><<<(output.numel() / 8 + 255) / 256, 256, 0, stream>>>(           \
      reinterpret_cast<const __nv_bfloat162*>(x.const_data_ptr()),                                 \
      reinterpret_cast<__nv_bfloat162*>(output.mutable_data_ptr()),                                \
      I(output.numel()),                                                                           \
      I(shape.back()))
    if (x.numel() <= UINT32_MAX) {
      TEI_SILU_PAIRS(uint32_t);
    } else {
      TEI_SILU_PAIRS(uint64_t);
    }
#undef TEI_SILU_PAIRS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
  if (activation == 4 && (x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16) &&
      shape.back() % 8 == 0 && reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0) {
#define TEI_GATE_PAIRS(T, I)                                                                       \
  gated_gelu_pairs_kernel<T, I><<<(output.numel() / 8 + 255) / 256, 256, 0, stream>>>(             \
      reinterpret_cast<const T*>(x.const_data_ptr()),                                              \
      reinterpret_cast<T*>(output.mutable_data_ptr()),                                             \
      I(output.numel() / 2),                                                                       \
      I(shape.back() / 2),                                                                         \
      gate_first)
    if (x.scalar_type() == at::kHalf) {
      if (x.numel() <= UINT32_MAX) {
        TEI_GATE_PAIRS(__half2, uint32_t);
      } else {
        TEI_GATE_PAIRS(__half2, uint64_t);
      }
    } else {
      if (x.numel() <= UINT32_MAX) {
        TEI_GATE_PAIRS(__nv_bfloat162, uint32_t);
      } else {
        TEI_GATE_PAIRS(__nv_bfloat162, uint64_t);
      }
    }
#undef TEI_GATE_PAIRS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
  }
#define TEI_GATE(T)                                                                                \
  do {                                                                                             \
    if (x.element_size() == 2 && shape.back() % 8 == 0 &&                                          \
        reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0) {                               \
      if (x.numel() <= UINT32_MAX)                                                                 \
        gated_vector_kernel<T, uint32_t, 8><<<(output.numel() / 8 + 255) / 256, 256, 0, stream>>>( \
            reinterpret_cast<const T*>(x.const_data_ptr()),                                        \
            reinterpret_cast<T*>(output.mutable_data_ptr()),                                       \
            output.numel(),                                                                        \
            shape.back(),                                                                          \
            activation,                                                                            \
            gate_first);                                                                           \
      else                                                                                         \
        gated_vector_kernel<T, int64_t, 8><<<(output.numel() / 8 + 255) / 256, 256, 0, stream>>>(  \
            reinterpret_cast<const T*>(x.const_data_ptr()),                                        \
            reinterpret_cast<T*>(output.mutable_data_ptr()),                                       \
            output.numel(),                                                                        \
            shape.back(),                                                                          \
            activation,                                                                            \
            gate_first);                                                                           \
    } else if (shape.back() % 4 == 0 &&                                                            \
               reinterpret_cast<uintptr_t>(x.const_data_ptr()) % (4 * x.element_size()) == 0) {    \
      if (x.numel() <= UINT32_MAX)                                                                 \
        gated_vector_kernel<T, uint32_t, 4><<<(output.numel() / 4 + 255) / 256, 256, 0, stream>>>( \
            reinterpret_cast<const T*>(x.const_data_ptr()),                                        \
            reinterpret_cast<T*>(output.mutable_data_ptr()),                                       \
            output.numel(),                                                                        \
            shape.back(),                                                                          \
            activation,                                                                            \
            gate_first);                                                                           \
      else                                                                                         \
        gated_vector_kernel<T, int64_t, 4><<<(output.numel() / 4 + 255) / 256, 256, 0, stream>>>(  \
            reinterpret_cast<const T*>(x.const_data_ptr()),                                        \
            reinterpret_cast<T*>(output.mutable_data_ptr()),                                       \
            output.numel(),                                                                        \
            shape.back(),                                                                          \
            activation,                                                                            \
            gate_first);                                                                           \
    } else                                                                                         \
      gated_kernel<T><<<(output.numel() + 255) / 256, 256, 0, stream>>>(                           \
          reinterpret_cast<const T*>(x.const_data_ptr()),                                          \
          reinterpret_cast<T*>(output.mutable_data_ptr()),                                         \
          output.numel(),                                                                          \
          shape.back(),                                                                            \
          activation,                                                                              \
          gate_first);                                                                             \
  } while (false)
  if (x.scalar_type() == at::kHalf) {
    TEI_GATE(__half);
  } else if (x.scalar_type() == at::kBFloat16) {
    TEI_GATE(__nv_bfloat16);
  } else if (x.scalar_type() == at::kFloat) {
    TEI_GATE(float);
  } else
    TORCH_CHECK(false, "Unsupported gated activation dtype");
#undef TEI_GATE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::pair<at::Tensor, at::Tensor> fused_rotary(const at::Tensor& q,
                                               const at::Tensor& k,
                                               const at::Tensor& cosine,
                                               const at::Tensor& sine,
                                               bool contract_first_product) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.dim() == 3 && k.dim() == 3 &&
                  q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.stride(2) == 1 &&
                  k.stride(2) == 1 && q.stride(1) == q.size(2) && k.stride(1) == k.size(2),
              "Invalid packed rotary Q/K tensors");
  TORCH_CHECK(k.device() == q.device() && k.scalar_type() == q.scalar_type(),
              "Rotary Q/K dtype/device mismatch");
  TORCH_CHECK(cosine.dim() > 0 && cosine.size(-1) > 0 && 2 * cosine.size(-1) <= q.size(-1) &&
                  cosine.numel() == q.size(0) * cosine.size(-1) && sine.sizes() == cosine.sizes(),
              "Invalid rotary frequency dimensions");
  for (const auto& t : {cosine, sine})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type() && t.is_contiguous(),
                "Rotary frequencies must be contiguous and match Q/K device/dtype");
  c10::cuda::CUDAGuard guard(q.device());
  auto qr = q.size(0) * q.size(1), kr = k.size(0) * k.size(1), dim = q.size(-1);
  auto output = at::empty({qr + kr, dim}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  // Keep the existing path for partial rotary, F32 contraction behavior, or
  // dimensions/strides that cannot be represented by the faster indexing.
  const bool paired = q.scalar_type() != at::kFloat && dim == 2 * cosine.size(-1) &&
                      output.numel() <= UINT32_MAX && q.size(0) * q.stride(0) <= UINT32_MAX &&
                      k.size(0) * k.stride(0) <= UINT32_MAX;
  if (output.numel() > 0) {
#define TEI_ROPE(T)                                                                                \
  if (paired)                                                                                      \
    paired_rotary_kernel<T><<<(output.numel() / 2 + 255) / 256, 256, 0, stream>>>(                 \
        reinterpret_cast<const T*>(q.const_data_ptr()),                                            \
        reinterpret_cast<const T*>(k.const_data_ptr()),                                            \
        reinterpret_cast<const T*>(cosine.const_data_ptr()),                                       \
        reinterpret_cast<const T*>(sine.const_data_ptr()),                                         \
        reinterpret_cast<T*>(output.mutable_data_ptr()),                                           \
        output.numel() / 2,                                                                        \
        qr,                                                                                        \
        q.size(1),                                                                                 \
        k.size(1),                                                                                 \
        q.stride(0),                                                                               \
        k.stride(0),                                                                               \
        cosine.size(-1),                                                                           \
        contract_first_product);                                                                   \
  else                                                                                             \
    rotary_kernel<T><<<(output.numel() + 255) / 256, 256, 0, stream>>>(                            \
        reinterpret_cast<const T*>(q.const_data_ptr()),                                            \
        reinterpret_cast<const T*>(k.const_data_ptr()),                                            \
        reinterpret_cast<const T*>(cosine.const_data_ptr()),                                       \
        reinterpret_cast<const T*>(sine.const_data_ptr()),                                         \
        reinterpret_cast<T*>(output.mutable_data_ptr()),                                           \
        qr,                                                                                        \
        kr,                                                                                        \
        q.size(1),                                                                                 \
        k.size(1),                                                                                 \
        q.stride(0),                                                                               \
        k.stride(0),                                                                               \
        dim,                                                                                       \
        cosine.size(-1),                                                                           \
        contract_first_product)
    if (q.scalar_type() == at::kHalf) {
      TEI_ROPE(__half);
    } else if (q.scalar_type() == at::kBFloat16) {
      TEI_ROPE(__nv_bfloat16);
    } else if (q.scalar_type() == at::kFloat) {
      TEI_ROPE(float);
    } else
      TORCH_CHECK(false, "Unsupported rotary dtype");
#undef TEI_ROPE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.narrow(0, 0, qr).view({q.size(0), q.size(1), dim}),
          output.narrow(0, qr, kr).view({k.size(0), k.size(1), dim})};
}

static void encoder_rotary_dispatch(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& cosine,
                                    const at::Tensor& sine,
                                    const at::Tensor& positions) {
  const bool indexed = positions.defined();
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && k.dim() == 3 && q.size(0) == k.size(0) &&
                  q.size(2) == k.size(2),
              "Invalid packed encoder Q/K");
  TORCH_CHECK(q.device() == k.device() && q.scalar_type() == k.scalar_type(),
              "Encoder Q/K dtype/device mismatch");
  for (const auto& t : {q, k})
    TORCH_CHECK(t.stride(2) == 1 && t.stride(1) == t.size(2) &&
                    t.stride(0) >= t.size(1) * t.size(2),
                "Encoder rotary requires nonoverlapping packed head views");
  TORCH_CHECK(cosine.dim() > 0 && cosine.size(-1) > 0 && 2 * cosine.size(-1) <= q.size(-1) &&
                  (indexed ? (cosine.dim() == 2 && cosine.size(0) > 0)
                           : cosine.numel() == q.size(0) * cosine.size(-1)) &&
                  sine.sizes() == cosine.sizes(),
              "Invalid encoder rotary frequencies");
  for (const auto& t : {cosine, sine})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type() && t.is_contiguous(),
                "Encoder rotary frequencies must match Q/K");
  if (indexed)
    TORCH_CHECK(positions.device() == q.device() && positions.scalar_type() == at::kLong &&
                    positions.is_contiguous() && positions.dim() == 1 &&
                    positions.numel() == q.size(0),
                "Invalid encoder rotary position indices");
  c10::cuda::CUDAGuard guard(q.device());
  const auto count = q.size(0) * (q.size(1) + k.size(1)) * cosine.size(-1);
  if (!count)
    return;
  auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  const bool narrow = count <= UINT32_MAX && uint64_t(q.size(0)) * q.stride(0) <= UINT32_MAX &&
                      uint64_t(k.size(0)) * k.stride(0) <= UINT32_MAX &&
                      uint64_t(cosine.numel()) <= UINT32_MAX;
  const auto aligned = [](const at::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 4 == 0;
  };
  const bool pairs = (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16) &&
                     q.size(2) % 2 == 0 && cosine.size(-1) % 2 == 0 && q.stride(0) % 2 == 0 &&
                     k.stride(0) % 2 == 0 && aligned(q) && aligned(k) && aligned(cosine) &&
                     aligned(sine);
#define TEI_ENCODER_PAIRS(T, I, P)                                                                 \
  encoder_rotary_pairs_kernel<T, I, P><<<(count / 2 + 255) / 256, 256, 0, stream>>>(               \
      reinterpret_cast<T*>(q.mutable_data_ptr()),                                                  \
      reinterpret_cast<T*>(k.mutable_data_ptr()),                                                  \
      reinterpret_cast<const T*>(cosine.const_data_ptr()),                                         \
      reinterpret_cast<const T*>(sine.const_data_ptr()),                                           \
      I(q.size(1)),                                                                                \
      I(k.size(1)),                                                                                \
      I(q.stride(0) / 2),                                                                          \
      I(k.stride(0) / 2),                                                                          \
      I(q.size(0)),                                                                                \
      I(q.size(2) / 2),                                                                            \
      I(cosine.size(-1) / 2),                                                                      \
      indexed ? positions.const_data_ptr<int64_t>() : nullptr,                                     \
      I(cosine.size(0)))
#define TEI_ENCODER_PAIRS_WIDTH(T)                                                                 \
  if (indexed) {                                                                                   \
    if (narrow) {                                                                                  \
      TEI_ENCODER_PAIRS(T, uint32_t, true);                                                        \
    } else {                                                                                       \
      TEI_ENCODER_PAIRS(T, uint64_t, true);                                                        \
    }                                                                                              \
  } else {                                                                                         \
    if (narrow) {                                                                                  \
      TEI_ENCODER_PAIRS(T, uint32_t, false);                                                       \
    } else {                                                                                       \
      TEI_ENCODER_PAIRS(T, uint64_t, false);                                                       \
    }                                                                                              \
  }
  if (pairs) {
    if (q.scalar_type() == at::kHalf) {
      TEI_ENCODER_PAIRS_WIDTH(__half2);
    } else {
      TEI_ENCODER_PAIRS_WIDTH(__nv_bfloat162);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
#undef TEI_ENCODER_PAIRS_WIDTH
#undef TEI_ENCODER_PAIRS
#define TEI_ENCODER_ROPE(T, I, P)                                                                  \
  encoder_rotary_inplace_kernel<T, I, P><<<(count + 255) / 256, 256, 0, stream>>>(                 \
      reinterpret_cast<T*>(q.mutable_data_ptr()),                                                  \
      reinterpret_cast<T*>(k.mutable_data_ptr()),                                                  \
      reinterpret_cast<const T*>(cosine.const_data_ptr()),                                         \
      reinterpret_cast<const T*>(sine.const_data_ptr()),                                           \
      I(q.size(1)),                                                                                \
      I(k.size(1)),                                                                                \
      I(q.stride(0)),                                                                              \
      I(k.stride(0)),                                                                              \
      I(q.size(0)),                                                                                \
      I(q.size(2)),                                                                                \
      I(cosine.size(-1)),                                                                          \
      indexed ? positions.const_data_ptr<int64_t>() : nullptr,                                     \
      I(cosine.size(0)))
#define TEI_ENCODER_ROPE_WIDTH(T)                                                                  \
  if (indexed) {                                                                                   \
    if (narrow) {                                                                                  \
      TEI_ENCODER_ROPE(T, uint32_t, true);                                                         \
    } else {                                                                                       \
      TEI_ENCODER_ROPE(T, uint64_t, true);                                                         \
    }                                                                                              \
  } else {                                                                                         \
    if (narrow) {                                                                                  \
      TEI_ENCODER_ROPE(T, uint32_t, false);                                                        \
    } else {                                                                                       \
      TEI_ENCODER_ROPE(T, uint64_t, false);                                                        \
    }                                                                                              \
  }
  if (q.scalar_type() == at::kHalf) {
    TEI_ENCODER_ROPE_WIDTH(__half);
  } else if (q.scalar_type() == at::kBFloat16) {
    TEI_ENCODER_ROPE_WIDTH(__nv_bfloat16);
  } else if (q.scalar_type() == at::kFloat) {
    TEI_ENCODER_ROPE_WIDTH(float);
  } else
    TORCH_CHECK(false, "Unsupported encoder rotary dtype");
#undef TEI_ENCODER_ROPE_WIDTH
#undef TEI_ENCODER_ROPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void encoder_rotary_inplace(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& cosine,
                            const at::Tensor& sine) {
  encoder_rotary_dispatch(q, k, cosine, sine, {});
}

void encoder_rotary_indexed_inplace(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& cosine,
                                    const at::Tensor& sine,
                                    const at::Tensor& positions) {
  encoder_rotary_dispatch(q, k, cosine, sine, positions);
}

at::Tensor fused_gelu(const at::Tensor& x, bool approximate) {
  return fused_bias_gelu(x, at::Tensor(), approximate);
}

std::pair<at::Tensor, at::Tensor> fused_qk_norm_rope(const at::Tensor& q,
                                                     const at::Tensor& k,
                                                     const at::Tensor& qw,
                                                     const at::Tensor& kw,
                                                     const at::Tensor& cosine,
                                                     const at::Tensor& sine,
                                                     double epsilon) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && q.dim() == 3 && k.dim() == 3 &&
                  q.size(0) == k.size(0) && q.size(2) == k.size(2),
              "Invalid packed Q/K tensors");
  TORCH_CHECK(q.device() == k.device() && q.scalar_type() == k.scalar_type(),
              "Q/K device and dtype must match");
  auto half = q.size(2) / 2;
  if (q.size(2) != 128 || (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16)) {
    auto rotate = [&](const at::Tensor& x, const at::Tensor& weight) {
      auto n = std::get<0>(at::_fused_rms_norm(x.contiguous(), {x.size(-1)}, weight, epsilon));
      auto a = n.narrow(-1, 0, half), b = n.narrow(-1, half, half);
      return at::cat({a * cosine - b * sine, b * cosine + a * sine}, -1);
    };
    return {rotate(q, qw), rotate(k, kw)};
  }
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && q.stride(1) == 128 && k.stride(1) == 128,
              "Q/K head dimensions must be contiguous");
  TORCH_CHECK(qw.numel() == 128 && kw.numel() == 128 && qw.is_contiguous() && kw.is_contiguous(),
              "Invalid Q/K norm weights");
  TORCH_CHECK(cosine.is_contiguous() && sine.is_contiguous() && cosine.numel() == q.size(0) * 64 &&
                  sine.numel() == q.size(0) * 64,
              "Invalid packed cosine/sine tensors");
  for (const auto& t : {qw, kw, cosine, sine})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type(),
                "Q/K norm and RoPE tensor device/dtype mismatch");
  TORCH_CHECK(q.size(0) <= INT32_MAX && q.stride(0) <= INT32_MAX && k.stride(0) <= INT32_MAX,
              "Q/K buffer exceeds kernel range");
  c10::cuda::CUDAGuard guard(q.device());
  const int64_t qrows = q.size(0) * q.size(1), krows = k.size(0) * k.size(1);
  auto output = at::empty({qrows + krows, 128}, q.options());
  auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
  auto blocks = std::min<int64_t>((qrows + krows + 7) / 8, 65535);
  if (blocks > 0) {
    if (q.scalar_type() == at::kHalf) {
      auto p = [](const at::Tensor& t) {
        return reinterpret_cast<const __half*>(t.const_data_ptr());
      };
      qk_norm_rope_f16<<<blocks, 128, 0, stream>>>(
          p(q),
          p(k),
          p(qw),
          p(kw),
          p(cosine),
          p(sine),
          reinterpret_cast<__half*>(output.mutable_data_ptr()),
          q.size(0),
          q.size(1),
          k.size(1),
          q.stride(0),
          k.stride(0),
          epsilon);
    } else {
      auto p = [](const at::Tensor& t) {
        return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
      };
      qk_bf16_bridge<<<blocks, 128, 0, stream>>>(
          p(q),
          p(k),
          p(qw),
          p(kw),
          p(cosine),
          p(sine),
          reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
          q.size(0),
          q.size(1),
          k.size(1),
          q.stride(0),
          k.stride(0),
          epsilon);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output.narrow(0, 0, qrows).view({q.size(0), q.size(1), 128}),
          output.narrow(0, qrows, krows).view({k.size(0), k.size(1), 128})};
}

at::Tensor fused_qkv_norm_rope_into(const at::Tensor& q,
                                    const at::Tensor& k,
                                    const at::Tensor& v,
                                    const at::Tensor& qw,
                                    const at::Tensor& kw,
                                    const at::Tensor& cosine,
                                    const at::Tensor& sine,
                                    double epsilon,
                                    const at::Tensor& key_target,
                                    const at::Tensor& value_target,
                                    const at::Tensor& output_rows) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && k.dim() == 3 && v.sizes() == k.sizes() &&
                  q.size(0) == k.size(0) && q.size(2) == 128 && k.size(2) == 128 &&
                  (q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16),
              "Unsupported direct prefix QKV geometry");
  for (const auto& t : {q, k, v})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type() &&
                    t.stride(2) == 1 && t.stride(1) == 128 && t.stride(0) >= 0 &&
                    t.stride(0) <= INT32_MAX && t.stride(0) % 8 == 0 && !t.is_neg() && !t.is_conj(),
                "Invalid direct prefix QKV layout");
  TORCH_CHECK(q.size(0) > 0 && q.size(0) <= INT32_MAX && q.size(1) > 0 && k.size(1) > 0 &&
                  q.size(1) <= INT32_MAX && k.size(1) <= INT32_MAX &&
                  q.size(0) * (q.size(1) + k.size(1)) <= INT32_MAX,
              "Direct prefix QKV exceeds kernel range");
  const bool indexed = output_rows.defined();
  TORCH_CHECK(key_target.dim() == 3 && value_target.sizes() == key_target.sizes() &&
                  key_target.sizes().slice(1) == k.sizes().slice(1) &&
                  (indexed ? key_target.size(0) >= k.size(0) : key_target.size(0) == k.size(0)) &&
                  key_target.size(0) * k.size(1) <= INT32_MAX && key_target.is_contiguous() &&
                  value_target.is_contiguous(),
              "Invalid direct prefix QKV targets");
  for (const auto& t : {qw, kw, cosine, sine, key_target, value_target})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type() &&
                    t.is_contiguous() && !t.is_neg() && !t.is_conj(),
                "Invalid direct prefix QKV coefficients or targets");
  for (const auto& t : {q, k, v, qw, kw, cosine, sine, key_target, value_target})
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0,
                "Direct prefix QKV requires aligned vectors");
  if (indexed)
    TORCH_CHECK(output_rows.device() == q.device() && output_rows.scalar_type() == at::kInt &&
                    output_rows.dim() == 1 && output_rows.is_contiguous() &&
                    output_rows.numel() == q.size(0) && !output_rows.is_neg() &&
                    !output_rows.is_conj(),
                "Invalid direct prefix output rows");
  TORCH_CHECK(qw.numel() == 128 && kw.numel() == 128 && cosine.numel() == q.size(0) * 64 &&
                  sine.numel() == q.size(0) * 64,
              "Invalid direct prefix normalization coefficients");
  for (const auto& source : {q, k, v, qw, kw, cosine, sine})
    TORCH_CHECK(!key_target.is_alias_of(source) && !value_target.is_alias_of(source),
                "Prefix QKV targets must own independent storage");
  TORCH_CHECK(!key_target.is_alias_of(value_target), "Prefix key and value targets must not alias");
  if (indexed)
    TORCH_CHECK(!key_target.is_alias_of(output_rows) && !value_target.is_alias_of(output_rows),
                "Prefix targets must not alias row metadata");
  c10::cuda::CUDAGuard guard(q.device());
  auto output = at::empty(q.sizes(), q.options());
  const auto blocks = std::min<int64_t>((q.size(0) * (q.size(1) + k.size(1)) + 7) / 8, 65535);
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int32_t* rows = indexed ? output_rows.const_data_ptr<int32_t>() : nullptr;
  auto launch = [&]<class T, bool Indexed>() {
    auto p = [](const at::Tensor& t) { return reinterpret_cast<const T*>(t.const_data_ptr()); };
    prefix_qkv_norm_rope<T, Indexed>
        <<<blocks, 128, 0, stream>>>(p(q),
                                     p(k),
                                     p(qw),
                                     p(kw),
                                     p(cosine),
                                     p(sine),
                                     p(v),
                                     reinterpret_cast<T*>(output.mutable_data_ptr()),
                                     reinterpret_cast<T*>(key_target.mutable_data_ptr()),
                                     reinterpret_cast<T*>(value_target.mutable_data_ptr()),
                                     q.size(0),
                                     q.size(1),
                                     k.size(1),
                                     q.stride(0),
                                     k.stride(0),
                                     v.stride(0),
                                     epsilon,
                                     rows,
                                     key_target.size(0));
  };
  if (q.scalar_type() == at::kHalf) {
    if (indexed)
      launch.template operator()<__half, true>();
    else
      launch.template operator()<__half, false>();
  } else {
    if (indexed)
      launch.template operator()<__nv_bfloat16, true>();
    else
      launch.template operator()<__nv_bfloat16, false>();
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::pair<at::Tensor, at::Tensor> nomic_router_topk8(const at::Tensor& p, int64_t k) {
  TORCH_CHECK(p.is_cuda() && p.scalar_type() == at::kFloat && p.dim() == 2 && p.size(1) == 8 &&
                  p.is_contiguous(),
              "Nomic eight-expert router requires contiguous CUDA FP32 probabilities");
  TORCH_CHECK(k > 0 && k <= 8, "Invalid Nomic eight-expert top-k count");
  c10::cuda::CUDAGuard guard(p.device());
  auto scores = at::empty({p.size(0), k}, p.options());
  auto indices = at::empty({p.size(0), k}, p.options().dtype(at::kLong));
  const auto blocks = std::min<int64_t>((p.size(0) + 127) / 128, 65535);
  if (blocks > 0) {
    auto stream = at::cuda::getCurrentCUDAStream(p.get_device());
    nomic_topk8_kernel<<<blocks, 128, 0, stream>>>(p.const_data_ptr<float>(),
                                                   scores.mutable_data_ptr<float>(),
                                                   indices.mutable_data_ptr<int64_t>(),
                                                   p.size(0),
                                                   int(k));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {scores, indices};
}

}
