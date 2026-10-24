// SPDX-License-Identifier: MIT
// Adapted from original Candle qwen35_gdn.cu and Candle fast_sum at
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original licenses retained.
// Explicit operations preserve both original FMA and separate-rounding modes.
/*
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#include "kernels/qwen35_recurrent.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <type_traits>

namespace tei {
namespace {
template <bool FMA> __device__ float accumulate(float sum, float a, float b) {
  if constexpr (FMA)
    return __fmaf_rn(a, b, sum);
  else
    return __fadd_rn(sum, __fmul_rn(a, b));
}

__device__ float sum_warp(float x) {
  for (int d = 16; d; d >>= 1)
    x = __fadd_rn(x, __shfl_down_sync(0xffffffff, x, d));
  return __shfl_sync(0xffffffff, x, 0);
}

template <int COLUMNS, bool FMA, bool CACHED = false>
__global__ void recurrent(const __nv_bfloat16* values,
                          const float* qk,
                          const float* gates,
                          const int32_t* cu,
                          __nv_bfloat16* output,
                          int kh,
                          int vh,
                          const float* const* initial = nullptr,
                          const int32_t* checkpoints = nullptr,
                          const int32_t* destinations = nullptr,
                          float* states = nullptr) {
  constexpr int WARPS = 8 / COLUMNS;
  const int sequence = blockIdx.x, head = blockIdx.y, lane = threadIdx.x % 32,
            warp = threadIdx.x / 32;
  const int column = blockIdx.z * (COLUMNS * WARPS) + warp * COLUMNS;
  const int qhead = head / (vh / kh), channels = (2 * kh + vh) * 128;
  float state[COLUMNS][4] = {};
  if constexpr (CACHED)
    if (initial && initial[sequence]) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int col = 0; col < COLUMNS; ++col)
          state[col][i] =
              initial[sequence][(int64_t(head) * 128 + lane + i * 32) * 128 + column + col];
      }
    }
  for (int t = cu[sequence]; t < cu[sequence + 1]; ++t) {
    float decay = gates[int64_t(t) * 2 * vh + head];
    float beta = gates[int64_t(t) * 2 * vh + vh + head];
    float q[4], k[4], memory[COLUMNS] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = qk[(int64_t(t) * kh + qhead) * 256 + lane + i * 32];
      k[i] = qk[(int64_t(t) * kh + qhead) * 256 + 128 + lane + i * 32];
#pragma unroll
      for (int col = 0; col < COLUMNS; ++col) {
        state[col][i] = __fmul_rn(state[col][i], decay);
        memory[col] = accumulate<FMA>(memory[col], state[col][i], k[i]);
      }
    }
#pragma unroll
    for (int col = 0; col < COLUMNS; ++col)
      memory[col] = sum_warp(memory[col]);
    float delta[COLUMNS], sum[COLUMNS] = {};
#pragma unroll
    for (int col = 0; col < COLUMNS; ++col) {
      float v = __bfloat162float(
          values[int64_t(t) * channels + 2 * kh * 128 + head * 128 + column + col]);
      delta[col] = __fmul_rn(__fsub_rn(v, memory[col]), beta);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
      for (int col = 0; col < COLUMNS; ++col) {
        state[col][i] = accumulate<FMA>(state[col][i], k[i], delta[col]);
        sum[col] = accumulate<FMA>(sum[col], state[col][i], q[i]);
      }
    }
#pragma unroll
    for (int col = 0; col < COLUMNS; ++col)
      sum[col] = sum_warp(sum[col]);
    if constexpr (CACHED)
      if (states && checkpoints[sequence] == t - cu[sequence] + 1 && destinations[sequence] >= 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
          for (int col = 0; col < COLUMNS; ++col)
            states[((int64_t(destinations[sequence]) * vh + head) * 128 + lane + i * 32) * 128 +
                   column + col] = state[col][i];
        }
      }
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < COLUMNS; ++col)
        output[(int64_t(t) * vh + head) * 128 + column + col] = __float2bfloat16_rn(sum[col]);
    }
  }
}

template <int COLUMNS, bool FMA>
void launch(const at::Tensor& v,
            const at::Tensor& qk,
            const at::Tensor& gates,
            const PackedInput& input,
            at::Tensor& out,
            int kh,
            int vh) {
  recurrent<COLUMNS, FMA>
      <<<dim3(input.batch, vh, 16), (8 / COLUMNS) * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(v.const_data_ptr()),
          qk.const_data_ptr<float>(),
          gates.const_data_ptr<float>(),
          input.cumulative.const_data_ptr<int32_t>(),
          reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
          kh,
          vh);
}

}

bool qwen35_recurrent_supported(c10::Device device, int64_t kh, int64_t vh) {
  // Smaller head counts retain the more parallel one-column fallback.
  if (!device.is_cuda() || device.index() < 0 || kh <= 0 || vh < 48 || vh > 128 || vh % kh != 0)
    return false;
  auto properties = at::cuda::getDeviceProperties(device.index());
  // Register state and warp shuffles require no architecture-specific MMA.
  return properties->major == 8 || properties->major == 9 || properties->major == 10 ||
         properties->major == 12;
}

void qwen35_recurrent_cuda(const at::Tensor& v,
                           const at::Tensor& qk,
                           const at::Tensor& gates,
                           const PackedInput& input,
                           at::Tensor& out,
                           int64_t kh,
                           int64_t vh,
                           bool fma) {
  TORCH_CHECK(qwen35_recurrent_supported(v.device(), kh, vh),
              "Unsupported precise recurrence profile");
  TORCH_CHECK(v.scalar_type() == at::kBFloat16 && v.dim() == 2 && v.is_contiguous(),
              "Recurrence values must be packed BF16 rows");
  const auto tokens = v.size(0);
  TORCH_CHECK(tokens > 0 && tokens <= INT32_MAX && v.size(1) == (2 * kh + vh) * 128,
              "Invalid recurrence value dimensions");
  TORCH_CHECK(qk.device() == v.device() && gates.device() == v.device() &&
                  out.device() == v.device() && qk.scalar_type() == at::kFloat &&
                  gates.scalar_type() == at::kFloat && out.scalar_type() == at::kBFloat16 &&
                  qk.is_contiguous() && gates.is_contiguous() && out.is_contiguous() &&
                  qk.sizes() == at::IntArrayRef({tokens * kh, 256}) &&
                  gates.sizes() == at::IntArrayRef({tokens, 2 * vh}) &&
                  out.sizes() == at::IntArrayRef({tokens, vh, 128}),
              "Invalid prepared recurrence tensors");
  TORCH_CHECK(input.batch > 0 && input.batch <= INT32_MAX &&
                  input.cumulative.device() == v.device() &&
                  input.cumulative.scalar_type() == at::kInt && input.cumulative.is_contiguous() &&
                  input.cumulative.dim() == 1 && input.cumulative.numel() == input.batch + 1,
              "Invalid recurrence sequence metadata");
  TORCH_CHECK(input.offsets && input.offsets[0] == 0 && input.offsets[input.batch] == tokens,
              "Recurrence offsets must span actual token rows");
  for (int64_t row = 0; row < input.batch; ++row)
    TORCH_CHECK(input.offsets[row + 1] > input.offsets[row],
                "Recurrence sequences must be nonempty and ordered");
  c10::cuda::CUDAGuard guard(v.device());
  if (fma)
    launch<4, true>(v, qk, gates, input, out, kh, vh);
  else
    launch<2, false>(v, qk, gates, input, out, kh, vh);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
template <bool FMA>
__global__ void cached_convolution(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   const int32_t* cu,
                                   const __nv_bfloat16* history,
                                   const int32_t* lengths,
                                   __nv_bfloat16* out,
                                   int tokens,
                                   int channels,
                                   int kernel,
                                   int sequences) {
  auto index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= int64_t(tokens) * channels)
    return;
  int t = index / channels, c = index % channels, lo = 0, hi = sequences;
  while (lo + 1 < hi) {
    int m = (lo + hi) / 2;
    if (cu[m] <= t)
      lo = m;
    else
      hi = m;
  }
  float sum = 0;
  for (int j = 0; j < kernel; ++j) {
    int source = t + j - kernel + 1;
    float value;
    if (source >= cu[lo])
      value = __bfloat162float(x[int64_t(source) * channels + c]);
    else {
      int tail = source - cu[lo] + kernel - 1;
      if (tail < kernel - 1 - lengths[lo])
        continue;
      value = __bfloat162float(history[(int64_t(lo) * (kernel - 1) + tail) * channels + c]);
    }
    sum = accumulate<FMA>(sum, value, __bfloat162float(weight[c * kernel + j]));
  }
  sum = __bfloat162float(__float2bfloat16_rn(sum));
  out[index] = __float2bfloat16_rn(sum / (1.0f + expf(-sum)));
}

template <int COLUMNS, bool FMA>
void cached_launch(const at::Tensor& v,
                   const at::Tensor& qk,
                   const at::Tensor& gates,
                   const PackedInput& input,
                   at::Tensor& out,
                   int kh,
                   int vh,
                   const at::Tensor& initial,
                   const at::Tensor& positions,
                   const at::Tensor& destinations,
                   at::Tensor& checkpoints) {
  recurrent<COLUMNS, FMA, true>
      <<<dim3(input.batch, vh, 16), (8 / COLUMNS) * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const __nv_bfloat16*>(v.const_data_ptr()),
          qk.const_data_ptr<float>(),
          gates.const_data_ptr<float>(),
          input.cumulative.const_data_ptr<int32_t>(),
          reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
          kh,
          vh,
          reinterpret_cast<const float* const*>(initial.const_data_ptr<int64_t>()),
          positions.const_data_ptr<int32_t>(),
          destinations.const_data_ptr<int32_t>(),
          checkpoints.defined() ? checkpoints.mutable_data_ptr<float>() : nullptr);
}
}

void qwen35_cached_conv_cuda(const at::Tensor& input,
                             const at::Tensor& weights,
                             const PackedInput& packed,
                             const at::Tensor& history,
                             const at::Tensor& lengths,
                             at::Tensor& output,
                             bool fma) {
  TORCH_CHECK(input.is_cuda(), "Cached convolution requires CUDA tensors");
  c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.is_contiguous(),
              "Cached convolution requires packed BF16 rows");
  TORCH_CHECK(weights.dim() == 2 || weights.dim() == 3, "Invalid cached convolution weights");
  auto tokens = input.size(0), channels = input.size(1), kernel = weights.size(-1);
  TORCH_CHECK(tokens > 0 && tokens <= INT32_MAX && channels > 0 && channels <= INT32_MAX &&
                  tokens * channels <= UINT32_MAX && kernel > 0 && kernel <= 16,
              "Invalid cached convolution geometry");
  TORCH_CHECK(weights.numel() == channels * kernel && weights.is_contiguous() &&
                  weights.scalar_type() == input.scalar_type() &&
                  weights.device() == input.device() && output.sizes() == input.sizes() &&
                  output.is_contiguous() && output.scalar_type() == input.scalar_type() &&
                  output.device() == input.device(),
              "Invalid cached convolution operands");
  TORCH_CHECK(
      packed.batch > 0 && packed.batch <= INT32_MAX &&
          packed.cumulative.device() == input.device() &&
          packed.cumulative.scalar_type() == at::kInt && packed.cumulative.is_contiguous() &&
          packed.cumulative.sizes() == at::IntArrayRef({packed.batch + 1}) && packed.offsets &&
          packed.offsets[0] == 0 && packed.offsets[packed.batch] == tokens,
      "Invalid cached convolution sequence metadata");
  for (int64_t row = 0; row < packed.batch; ++row)
    TORCH_CHECK(packed.offsets[row + 1] > packed.offsets[row],
                "Cached convolution sequences must be nonempty and ordered");
  TORCH_CHECK(lengths.device() == input.device() && lengths.scalar_type() == at::kInt &&
                  lengths.is_contiguous() && lengths.sizes() == at::IntArrayRef({packed.batch}),
              "Invalid cached convolution history lengths");
  TORCH_CHECK(history.sizes() == at::IntArrayRef({packed.batch, kernel - 1, channels}) &&
                  history.is_contiguous() && history.scalar_type() == input.scalar_type() &&
                  history.device() == input.device(),
              "Invalid cached convolution history");
  auto launch = [&](auto mode) {
    constexpr bool FMA = decltype(mode)::value;
    cached_convolution<FMA>
        <<<(tokens * channels + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weights.const_data_ptr()),
            packed.cumulative.const_data_ptr<int32_t>(),
            reinterpret_cast<const __nv_bfloat16*>(history.const_data_ptr()),
            lengths.const_data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
            tokens,
            channels,
            kernel,
            packed.batch);
  };
  if (fma)
    launch(std::true_type{});
  else
    launch(std::false_type{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// Resume each sequence from its retained F32 state pointer and capture requested checkpoint rows.
void qwen35_cached_recurrent_cuda(const at::Tensor& values,
                                  const at::Tensor& qk,
                                  const at::Tensor& gates,
                                  const PackedInput& input,
                                  at::Tensor& output,
                                  int64_t kh,
                                  int64_t vh,
                                  bool fma,
                                  const at::Tensor& initial,
                                  const at::Tensor& positions,
                                  const at::Tensor& destinations,
                                  at::Tensor& checkpoints) {
  TORCH_CHECK(values.is_cuda(), "Cached recurrence requires CUDA tensors");
  c10::cuda::CUDAGuard guard(values.device());
  TORCH_CHECK(kh > 0 && vh > 0 && vh <= 128 && vh % kh == 0 &&
                  values.scalar_type() == at::kBFloat16 && values.dim() == 2 &&
                  values.is_contiguous(),
              "Invalid cached recurrence geometry");
  auto tokens = values.size(0);
  TORCH_CHECK(tokens > 0 && tokens <= INT32_MAX && values.size(1) == (2 * kh + vh) * 128 &&
                  values.numel() <= UINT32_MAX,
              "Invalid cached recurrence values");
  TORCH_CHECK(qk.device() == values.device() && gates.device() == values.device() &&
                  output.device() == values.device() && qk.scalar_type() == at::kFloat &&
                  gates.scalar_type() == at::kFloat && output.scalar_type() == at::kBFloat16 &&
                  qk.is_contiguous() && gates.is_contiguous() && output.is_contiguous() &&
                  qk.sizes() == at::IntArrayRef({tokens * kh, 256}) &&
                  gates.sizes() == at::IntArrayRef({tokens, 2 * vh}) &&
                  output.sizes() == at::IntArrayRef({tokens, vh, 128}),
              "Invalid cached recurrence operands");
  TORCH_CHECK(input.batch > 0 && input.batch <= INT32_MAX &&
                  input.cumulative.device() == values.device() &&
                  input.cumulative.scalar_type() == at::kInt && input.cumulative.is_contiguous() &&
                  input.cumulative.sizes() == at::IntArrayRef({input.batch + 1}) && input.offsets &&
                  input.offsets[0] == 0 && input.offsets[input.batch] == tokens,
              "Invalid cached recurrence sequence metadata");
  for (int64_t row = 0; row < input.batch; ++row)
    TORCH_CHECK(input.offsets[row + 1] > input.offsets[row],
                "Cached recurrence sequences must be nonempty and ordered");
  TORCH_CHECK(initial.scalar_type() == at::kLong && positions.scalar_type() == at::kInt &&
                  destinations.scalar_type() == at::kInt &&
                  initial.sizes() == at::IntArrayRef({input.batch}) &&
                  positions.sizes() == initial.sizes() && destinations.sizes() == initial.sizes() &&
                  initial.device() == values.device() && positions.device() == values.device() &&
                  destinations.device() == values.device() && initial.is_contiguous() &&
                  positions.is_contiguous() && destinations.is_contiguous(),
              "Invalid cached recurrence metadata");
  if (checkpoints.defined())
    TORCH_CHECK(checkpoints.device() == values.device() &&
                    checkpoints.scalar_type() == at::kFloat && checkpoints.is_contiguous() &&
                    checkpoints.dim() == 4 && checkpoints.size(0) > 0 &&
                    checkpoints.size(0) <= input.batch &&
                    checkpoints.sizes().slice(1) == at::IntArrayRef({vh, 128, 128}),
                "Invalid cached recurrence checkpoint storage");
  if (qwen35_recurrent_supported(values.device(), kh, vh)) {
    if (fma)
      cached_launch<4, true>(
          values, qk, gates, input, output, kh, vh, initial, positions, destinations, checkpoints);
    else
      cached_launch<2, false>(
          values, qk, gates, input, output, kh, vh, initial, positions, destinations, checkpoints);
  } else {
    if (fma)
      cached_launch<1, true>(
          values, qk, gates, input, output, kh, vh, initial, positions, destinations, checkpoints);
    else
      cached_launch<1, false>(
          values, qk, gates, input, output, kh, vh, initial, positions, destinations, checkpoints);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
