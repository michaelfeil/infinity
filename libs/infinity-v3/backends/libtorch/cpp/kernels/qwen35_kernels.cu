// SPDX-License-Identifier: MIT
// The included Candle kernels retain their MIT notices. This wrapper dispatches
// those native packed kernels on LibTorch's current stream, without Python.
#include "kernels/qwen35_kernels.h"
#include "kernels/gemma2_norm.h"
#include "kernels/qwen35_recurrent.h"
#include "batching/prefix_cache.h"
#include "kernels/qwen35_chunk.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "../../../candle/src/kernels/qwen35_gdn.cu"
#include "../../../candle/src/kernels/gemma_rms_norm.cu"

namespace tei {
namespace {
/// Aligned four-value BF16 pack used by the native convolution path.
struct alignas(8) ConvValues {
  __nv_bfloat16 values[4];
};

// Cached suffixes read their immutable prefix tail; short prefixes need no padding.
template <bool Cached = false>
__global__ void packed_conv_vector(const __nv_bfloat16* x,
                                   const __nv_bfloat16* weight,
                                   const uint32_t* cumulative,
                                   __nv_bfloat16* output,
                                   int channels,
                                   int kernel,
                                   int sequences,
                                   const int64_t* sources = nullptr,
                                   const __nv_bfloat16* history = nullptr,
                                   const int32_t* history_cu = nullptr) {
  const int token = blockIdx.x, column = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (column >= channels)
    return;
  int lo = 0, hi = sequences;
  while (lo + 1 < hi) {
    const int middle = (lo + hi) / 2;
    if (cumulative[middle] <= token)
      lo = middle;
    else
      hi = middle;
  }
  int history_end = 0, previous = 0;
  if constexpr (Cached) {
    history_end = history_cu[sources[lo] + 1];
    previous = history_end - history_cu[sources[lo]];
  }
  float sums[4] = {};
  for (int j = 0; j < kernel; ++j) {
    const int source = token + j - kernel + 1;
    const __nv_bfloat16* row = nullptr;
    if (source >= int(cumulative[lo]))
      row = x + int64_t(source) * channels;
    else if constexpr (Cached) {
      if (source >= int(cumulative[lo]) - previous)
        row = history + int64_t(history_end + source - int(cumulative[lo])) * channels;
    }
    if (row) {
      const auto values = *reinterpret_cast<const ConvValues*>(row + column);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        sums[i] += __bfloat162float(values.values[i]) *
                   __bfloat162float(weight[(column + i) * kernel + j]);
    }
  }
  ConvValues result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // Preserve convolution -> BF16 -> SiLU rounding and accumulation order.
    const float rounded = __bfloat162float(__float2bfloat16_rn(sums[i]));
    result.values[i] = __float2bfloat16_rn(rounded / (1.f + expf(-rounded)));
  }
  *reinterpret_cast<ConvValues*>(output + int64_t(token) * channels + column) = result;
}
}

// SPDX-License-Identifier: MIT
// Adapted from Candle gemma_rope_reference_bf16 (included above with its full
// MIT notices). Preserve BF16 rounding after each product. Read original
// packed strides and per-token positions directly; no synthetic rows or heads.
namespace {
__global__ void staged_rotary(const __nv_bfloat16* x,
                              const __nv_bfloat16* cos,
                              const __nv_bfloat16* sin,
                              const int64_t* positions,
                              __nv_bfloat16* out,
                              int64_t pairs,
                              int64_t heads,
                              int64_t half,
                              int64_t token_stride,
                              int64_t head_stride,
                              int64_t table_rows) {
  const auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= pairs)
    return;
  const auto column = i % half, head = (i / half) % heads, token = i / (half * heads);
  const auto source = token * token_stride + head * head_stride + column;
  CUDA_KERNEL_ASSERT(positions[token] >= 0 && positions[token] < table_rows);
  const auto frequency = positions[token] * half + column;
  const auto a = __bfloat162float(x[source]), b = __bfloat162float(x[source + half]);
  const auto c = __bfloat162float(cos[frequency]), s = __bfloat162float(sin[frequency]);
  const auto ac = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, c)));
  const auto bs = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, s)));
  const auto bc = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, c)));
  const auto as = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, s)));
  const auto destination = (token * heads + head) * half * 2 + column;
  out[destination] = __float2bfloat16_rn(__fsub_rn(ac, bs));
  out[destination + half] = __float2bfloat16_rn(__fadd_rn(bc, as));
}
}

at::Tensor gemma_rotary_cuda(const at::Tensor& x,
                             const at::Tensor& cos,
                             const at::Tensor& sin,
                             const at::Tensor& positions) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.stride(2) == 1,
              "Gemma rotary requires true BF16 packed T/H/D rows");
  TORCH_CHECK(x.size(2) > 0 && x.numel() / 2 <= int64_t(INT32_MAX) * 256,
              "Gemma rotary launch exceeds kernel range");
  const auto half = x.size(2) / 2;
  TORCH_CHECK(x.size(2) == 2 * half && cos.dim() == 2 && cos.size(1) == half &&
                  cos.sizes() == sin.sizes(),
              "Invalid rotary dimensions");
  TORCH_CHECK(cos.device() == x.device() && sin.device() == x.device() &&
                  positions.device() == x.device() && cos.scalar_type() == x.scalar_type() &&
                  sin.scalar_type() == x.scalar_type() && positions.scalar_type() == at::kLong &&
                  cos.is_contiguous() && sin.is_contiguous() && positions.is_contiguous() &&
                  positions.numel() == x.size(0),
              "Invalid rotary tables/positions");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  const auto pairs = x.numel() / 2;
  if (pairs)
    staged_rotary<<<(pairs + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(cos.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(sin.const_data_ptr()),
        positions.const_data_ptr<int64_t>(),
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        pairs,
        x.size(1),
        half,
        x.stride(0),
        x.stride(1),
        cos.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// SPDX-License-Identifier: MIT
// Adapted from backends/candle/src/kernels/gemma_rms_norm.cu.
// Candle compiles this reference with --fmad=false. Keep the mean and epsilon
// as separate FP32 rounds without changing other native kernel flags.
template <bool VIRTUAL_LANES, bool PAIRED = false, bool RESIDUAL = false>
__device__ __forceinline__ void
tei_gemma_rms_norm_reference_impl(const __nv_bfloat16* x,
                                  const float* scale,
                                  __nv_bfloat16* out,
                                  unsigned int width,
                                  unsigned int heads,
                                  unsigned long long token_stride,
                                  float eps,
                                  const __nv_bfloat16* router_scale = nullptr,
                                  __nv_bfloat16 root = {},
                                  const __nv_bfloat16* residual = nullptr) {
  const size_t base = static_cast<size_t>(blockIdx.x / heads) * token_stride +
                      static_cast<size_t>(blockIdx.x % heads) * width;
  const size_t out_base = static_cast<size_t>(blockIdx.x) * width;
  __shared__ float sums[1024];
  float sum = 0.f;
  if constexpr (VIRTUAL_LANES) {
    // Emulate the original logical lanes before reducing across this smaller
    // block. Pairing lane 0+512, then 0+256 preserves the 1024-lane tree.
    unsigned int logical_threads = 1;
    while (logical_threads < width && logical_threads < 1024)
      logical_threads <<= 1;
    const unsigned int lanes = logical_threads / blockDim.x;
    float partial[4] = {0.f, 0.f, 0.f, 0.f};
    for (unsigned int lane = 0; lane < lanes; ++lane) {
      for (unsigned int col = threadIdx.x + lane * blockDim.x; col < width;
           col += logical_threads) {
        float v = __bfloat162float(x[base + col]);
        partial[lane] += v * v;
      }
    }
    for (unsigned int stride = lanes / 2; stride; stride >>= 1)
      for (unsigned int lane = 0; lane < stride; ++lane)
        partial[lane] += partial[lane + stride];
    sum = partial[0];
  } else {
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
      float v = __bfloat162float(x[base + col]);
      sum += v * v;
    }
  }
  sums[threadIdx.x] = sum;
  for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride)
      sums[threadIdx.x] += sums[threadIdx.x + stride];
  }
  // Finish the same reduction tree within the first warp.
  if (threadIdx.x < 32) {
    sum = sums[threadIdx.x];
    const unsigned int mask = __activemask();
    for (unsigned int stride = blockDim.x < 32 ? blockDim.x / 2 : 16; stride; stride >>= 1) {
      float other = __shfl_down_sync(mask, sum, stride);
      if (threadIdx.x < stride)
        sum += other;
    }
    if (threadIdx.x == 0) {
      float mean = __fmul_rn(sum, static_cast<float>(1.0 / static_cast<double>(width)));
      sums[0] = sqrtf(__fadd_rn(mean, eps));
    }
  }
  __syncthreads();
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    float normalized = __bfloat162float(x[base + col]) / sums[0];
    if constexpr (RESIDUAL) {
      const auto rounded = __float2bfloat16_rn(normalized * scale[col]);
      out[out_base + col] = __hadd(residual[out_base + col], rounded);
    } else
      out[out_base + col] = __float2bfloat16_rn(normalized * scale[col]);
    if constexpr (PAIRED) {
      __nv_bfloat16 router = __float2bfloat16_rn(normalized * 1.f);
      router = router * root + __float2bfloat16_rn(0.f);
      out[static_cast<size_t>(gridDim.x) * width + out_base + col] = router * router_scale[col];
    }
  }
}

static __global__ void tei_gemma_rms_norm_reference_bf16(const __nv_bfloat16* x,
                                                         const float* scale,
                                                         __nv_bfloat16* out,
                                                         unsigned int width,
                                                         unsigned int heads,
                                                         unsigned long long token_stride,
                                                         float eps) {
  tei_gemma_rms_norm_reference_impl<false>(x, scale, out, width, heads, token_stride, eps);
}

// SPDX-License-Identifier: MIT
// Adapted from backends/candle/src/kernels/gemma_rms_norm.cu.
// Candle compiles this reference with --fmad=false. Keep the mean and epsilon
// as separate FP32 rounds without changing other native kernel flags.
template <bool VIRTUAL_LANES, bool PAIRED = false, bool RESIDUAL = false>
__device__ __forceinline__ void
g2_candidate_reference_impl(const __nv_bfloat16* x,
                            const float* scale,
                            __nv_bfloat16* out,
                            unsigned int width,
                            unsigned int heads,
                            unsigned long long token_stride,
                            float eps,
                            const __nv_bfloat16* router_scale = nullptr,
                            __nv_bfloat16 root = {},
                            const __nv_bfloat16* residual = nullptr) {
  const size_t base = static_cast<size_t>(blockIdx.x / heads) * token_stride +
                      static_cast<size_t>(blockIdx.x % heads) * width;
  const size_t out_base = static_cast<size_t>(blockIdx.x) * width;
  __shared__ float sums[1024];
  float sum = 0.f;
  if constexpr (VIRTUAL_LANES) {
    // Emulate the original logical lanes before reducing across this smaller
    // block. Pairing lane 0+512, then 0+256 preserves the 1024-lane tree.
    unsigned int logical_threads = 1;
    while (logical_threads < width && logical_threads < 1024)
      logical_threads <<= 1;
    const unsigned int lanes = logical_threads / blockDim.x;
    float partial[4] = {0.f, 0.f, 0.f, 0.f};
    for (unsigned int lane = 0; lane < lanes; ++lane) {
      for (unsigned int col = threadIdx.x + lane * blockDim.x; col < width;
           col += logical_threads) {
        float v = __bfloat162float(x[base + col]);
        partial[lane] = __fadd_rn(partial[lane], __fmul_rn(v, v));
      }
    }
    for (unsigned int stride = lanes / 2; stride; stride >>= 1)
      for (unsigned int lane = 0; lane < stride; ++lane)
        partial[lane] = __fadd_rn(partial[lane], partial[lane + stride]);
    sum = partial[0];
  } else {
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
      float v = __bfloat162float(x[base + col]);
      sum = __fadd_rn(sum, __fmul_rn(v, v));
    }
  }
  sums[threadIdx.x] = sum;
  for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride)
      sums[threadIdx.x] = __fadd_rn(sums[threadIdx.x], sums[threadIdx.x + stride]);
  }
  // Finish the same reduction tree within the first warp.
  if (threadIdx.x < 32) {
    sum = sums[threadIdx.x];
    const unsigned int mask = __activemask();
    for (unsigned int stride = blockDim.x < 32 ? blockDim.x / 2 : 16; stride; stride >>= 1) {
      float other = __shfl_down_sync(mask, sum, stride);
      if (threadIdx.x < stride)
        sum = __fadd_rn(sum, other);
    }
    if (threadIdx.x == 0) {
      float mean = __fmul_rn(sum, static_cast<float>(1.0 / static_cast<double>(width)));
      sums[0] = sqrtf(__fadd_rn(mean, eps));
    }
  }
  __syncthreads();
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    float normalized = __fdiv_rn(__bfloat162float(x[base + col]), sums[0]);
    if constexpr (RESIDUAL) {
      const auto rounded = __float2bfloat16_rn(__fmul_rn(normalized, scale[col]));
      out[out_base + col] = __hadd(residual[out_base + col], rounded);
    } else
      out[out_base + col] = __float2bfloat16_rn(__fmul_rn(normalized, scale[col]));
    if constexpr (PAIRED) {
      __nv_bfloat16 router = __float2bfloat16_rn(normalized * 1.f);
      router = router * root + __float2bfloat16_rn(0.f);
      out[static_cast<size_t>(gridDim.x) * width + out_base + col] = router * router_scale[col];
    }
  }
}

static __global__ void g2_candidate_reference_bf16(const __nv_bfloat16* x,
                                                   const float* scale,
                                                   __nv_bfloat16* out,
                                                   unsigned int width,
                                                   unsigned int heads,
                                                   unsigned long long token_stride,
                                                   float eps) {
  g2_candidate_reference_impl<false>(x, scale, out, width, heads, token_stride, eps);
}

// Adapted from MIT Candle warp_reference<256>, explicit RN staging for native TU.
static __global__ void g2_candidate_warp(const __nv_bfloat16* x,
                                         const float* scale,
                                         __nv_bfloat16* out,
                                         unsigned width,
                                         unsigned heads,
                                         unsigned long long stride,
                                         float eps,
                                         unsigned rows) {
  unsigned lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
  if (row >= rows)
    return;
  size_t base = size_t(row / heads) * stride + size_t(row % heads) * width;
  float partial[8] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i)
    for (unsigned col = lane + i * 32; col < width; col += 256) {
      float v = __bfloat162float(x[base + col]);
      partial[i] = __fadd_rn(partial[i], __fmul_rn(v, v));
    }
#pragma unroll
  for (int step = 4; step; step >>= 1) {
#pragma unroll
    for (int i = 0; i < step; ++i)
      partial[i] = __fadd_rn(partial[i], partial[i + step]);
  }
  float sum = partial[0];
#pragma unroll
  for (int step = 16; step; step >>= 1) {
    float other = __shfl_down_sync(0xffffffff, sum, step);
    if (lane < step)
      sum = __fadd_rn(sum, other);
  }
  float den = 0;
  if (!lane)
    den = sqrtf(__fadd_rn(__fmul_rn(sum, float(1. / double(width))), eps));
  den = __shfl_sync(0xffffffff, den, 0);
  for (unsigned col = lane; col < width; col += 32)
    out[size_t(row) * width + col] =
        __float2bfloat16_rn(__fmul_rn(__fdiv_rn(__bfloat162float(x[base + col]), den), scale[col]));
}

static __global__ void g2_candidate_compact(const __nv_bfloat16* x,
                                            const float* scale,
                                            __nv_bfloat16* out,
                                            unsigned w,
                                            unsigned h,
                                            unsigned long long s,
                                            float e) {
  g2_candidate_reference_impl<true>(x, scale, out, w, h, s, e);
}

at::Tensor gemma2_norm_cuda(const at::Tensor& x, const at::Tensor& scale, double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                  scale.scalar_type() == at::kFloat && (x.dim() == 2 || x.dim() == 3) &&
                  x.stride(-1) == 1 && scale.device() == x.device(),
              "Gemma2 compact RMS requires original packed BF16/F32 operands");
  const auto width = x.size(-1);
  TORCH_CHECK(width > 0, "Gemma2 norm width must be positive");
  const auto rows = x.numel() / width;
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX, "Gemma2 norm row bounds");
  if (!((width == 256 && rows >= 4096) || (width == 512 && rows >= 2048)))
    return gemma_norm_cuda(x, scale, eps, true);
  TORCH_CHECK(scale.numel() == width && scale.is_contiguous() &&
                  (x.dim() != 3 || x.stride(1) == width),
              "Gemma2 compact RMS scale/layout mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  unsigned w = x.size(-1), h = x.dim() == 3 ? x.size(1) : 1, r = x.numel() / w;
  auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 1;
  while (threads < w && threads < 1024)
    threads *= 2;
  auto a = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  auto b = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  if (w == 256 && r >= 4096)
    g2_candidate_warp<<<(r + 7) / 8, 256, 0, stream>>>(
        a, scale.const_data_ptr<float>(), b, w, h, x.stride(0), eps, r);
  else if (w > 256 && r >= 2048)
    g2_candidate_compact<<<r, 256, 0, stream>>>(
        a, scale.const_data_ptr<float>(), b, w, h, x.stride(0), eps);
  else
    g2_candidate_reference_bf16<<<r, threads, 0, stream>>>(
        a, scale.const_data_ptr<float>(), b, w, h, x.stride(0), eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// SPDX-License-Identifier: MIT
// Same original sqrt/div norm tree and BF16 norm cast; retain residual-first
// BF16 addition while removing the standalone add launch. No rsqrt substitution.
static __global__ void g2_residual_reference(const __nv_bfloat16* x,
                                             const float* scale,
                                             const __nv_bfloat16* residual,
                                             __nv_bfloat16* out,
                                             unsigned width,
                                             float eps) {
  tei_gemma_rms_norm_reference_impl<false, false, true>(
      x, scale, out, width, 1, width, eps, nullptr, {}, residual);
}

static __global__ void g2_residual_compact(const __nv_bfloat16* x,
                                           const float* scale,
                                           const __nv_bfloat16* residual,
                                           __nv_bfloat16* out,
                                           unsigned width,
                                           float eps) {
  g2_candidate_reference_impl<true, false, true>(
      x, scale, out, width, 1, width, eps, nullptr, {}, residual);
}

at::Tensor gemma2_residual_norm_cuda(const at::Tensor& residual,
                                     const at::Tensor& x,
                                     const at::Tensor& scale,
                                     double eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == 512 &&
                  x.is_contiguous(),
              "Gemma2 residual norm requires original contiguous BF16 hidden width512");
  TORCH_CHECK(residual.device() == x.device() && residual.scalar_type() == x.scalar_type() &&
                  residual.sizes() == x.sizes() && residual.is_contiguous() &&
                  scale.device() == x.device() && scale.scalar_type() == at::kFloat &&
                  scale.is_contiguous() && scale.numel() == 512,
              "Gemma2 residual norm operand mismatch");
  TORCH_CHECK(x.size(0) > 0 && x.size(0) <= INT32_MAX, "Gemma2 residual norm row bounds");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  const auto* src = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  const auto* res = reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr());
  auto* dst = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (x.size(0) >= 2048)
    g2_residual_compact<<<x.size(0), 256, 0, stream>>>(
        src, scale.const_data_ptr<float>(), res, dst, 512, eps);
  else
    g2_residual_reference<<<x.size(0), 512, 0, stream>>>(
        src, scale.const_data_ptr<float>(), res, dst, 512, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Adapted from the MIT precise_activation in Candle's gated_activation.cu.
// The original source and its complete notices remain in the repository.
template <bool Gelu>
__device__ __forceinline__ __nv_bfloat16 gemma_precise_activation(__nv_bfloat16 x) {
  using T = __nv_bfloat16;
  if constexpr (Gelu) {
    T square = x * x, cube = square * x;
    T alpha = x + T(.044715) * cube;
    T tanh_input = T(M_2_SQRTPI * M_SQRT1_2) * alpha;
    T tanh_value = T(tanhf(float(tanh_input)));
    return T(.5) * x * (T(1.) + tanh_value);
  } else {
    T exponential = hexp(-x), denominator = T(1.) + exponential;
    return x / denominator;
  }
}

template <bool Gelu>
__global__ void
gemma_precise_gate(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t count, int64_t width) {
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  auto row = i / width, column = i % width;
  out[i] = gemma_precise_activation<Gelu>(x[row * 2 * width + column]) *
           x[row * 2 * width + width + column];
}

template <bool Gelu>
__global__ void gemma_precise_unary(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t count) {
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    out[i] = gemma_precise_activation<Gelu>(x[i]);
}

at::Tensor gemma_activation_cuda(const at::Tensor& value, bool gelu) {
  c10::cuda::CUDAGuard guard(value.device());
  TORCH_CHECK(value.scalar_type() == at::kBFloat16, "Gemma precise activation requires BF16");
  auto x = value.contiguous(), out = at::empty_like(x);
  if (out.numel() == 0)
    return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto input = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto output = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  if (gelu)
    gemma_precise_unary<true>
        <<<(out.numel() + 255) / 256, 256, 0, stream>>>(input, output, out.numel());
  else
    gemma_precise_unary<false>
        <<<(out.numel() + 255) / 256, 256, 0, stream>>>(input, output, out.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor gemma_gated_cuda(const at::Tensor& x, bool gelu) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 &&
                  x.size(1) % 2 == 0,
              "Gemma precise GLU requires contiguous packed BF16 projection");
  auto width = x.size(1) / 2;
  auto out = at::empty({x.size(0), width}, x.options());
  if (out.numel() == 0)
    return out;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  auto input = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  auto output = reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr());
  if (gelu)
    gemma_precise_gate<true>
        <<<(out.numel() + 255) / 256, 256, 0, stream>>>(input, output, out.numel(), width);
  else
    gemma_precise_gate<false>
        <<<(out.numel() + 255) / 256, 256, 0, stream>>>(input, output, out.numel(), width);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor
gemma_norm_cuda(const at::Tensor& x, const at::Tensor& scale, double epsilon, bool reference) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && scale.scalar_type() == at::kFloat &&
                  x.dim() >= 2 && x.dim() <= 3 && x.stride(-1) == 1,
              "Gemma CUDA norm requires packed BF16 rows and FP32 scales");
  auto width = x.size(-1), heads = x.dim() == 3 ? x.size(1) : 1, rows = x.numel() / width;
  TORCH_CHECK(width > 0 && width <= 8192 && scale.numel() == width,
              "Invalid Gemma CUDA norm width");
  auto output = at::empty(x.sizes(), x.options());
  int threads = 1;
  while (threads < width && threads < (reference ? 1024 : 256))
    threads *= 2;
  // The optimized legacy reduction uses a full-warp shuffle mask.
  // Widths below32 still need32 participating lanes (extra lanes sum zero).
  if (!reference && threads < 32)
    threads = 32;
  auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
  auto source = reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr());
  auto destination = reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr());
  if (reference)
    tei_gemma_rms_norm_reference_bf16<<<rows, threads, 0, stream>>>(
        source, scale.const_data_ptr<float>(), destination, width, heads, x.stride(0), epsilon);
  else
    gemma_rms_norm_bf16<<<rows, threads, 0, stream>>>(
        source, scale.const_data_ptr<float>(), destination, width, heads, x.stride(0), epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// SPDX-License-Identifier: MIT
// Adapted from the included Candle gemma_rms_norm_bf16, retaining its
// complete notices and exact reduction tree. BF16 scratch preserves the
// original norm output cast before Candle's contracted BF16 rotary.
namespace {
__global__ void gemma3_qk_norm_rotary(const __nv_bfloat16* q,
                                      const __nv_bfloat16* k,
                                      const float* qs,
                                      const float* ks,
                                      const __nv_bfloat16* cosine,
                                      const __nv_bfloat16* sine,
                                      const int64_t* positions,
                                      __nv_bfloat16* qo,
                                      __nv_bfloat16* ko,
                                      unsigned width,
                                      unsigned qheads,
                                      unsigned kheads,
                                      unsigned long long qstride,
                                      unsigned long long kstride,
                                      unsigned long long qrows,
                                      float eps,
                                      int64_t table_rows) {
  const bool query = blockIdx.x < qrows;
  const auto row = query ? static_cast<unsigned long long>(blockIdx.x) : blockIdx.x - qrows;
  const auto heads = query ? qheads : kheads;
  const auto token = row / heads;
  const auto base = token * (query ? qstride : kstride) + (row % heads) * width;
  const auto* x = query ? q : k;
  const auto* scale = query ? qs : ks;
  auto* out = query ? qo : ko;
  float sum = 0.f;
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float v = __bfloat162float(x[base + col]);
    sum += v * v;
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float partial[8];
  if ((threadIdx.x & 31) == 0)
    partial[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.f;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      partial[0] = rsqrtf(sum / width + eps);
  }
  __syncthreads();
  extern __shared__ __nv_bfloat16 normalized[];
  const float inv = partial[0];
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float value = __bfloat162float(x[base + col]) * inv;
    normalized[col] = __float2bfloat16_rn(value * scale[col]);
  }
  __syncthreads();
  const auto position = positions[token];
  CUDA_KERNEL_ASSERT(position >= 0 && position < table_rows);
  const unsigned half = width / 2;
  for (unsigned col = threadIdx.x; col < half; col += blockDim.x) {
    const auto a = normalized[col], b = normalized[col + half];
    const auto c = cosine[position * half + col], s = sine[position * half + col];
    out[row * width + col] = __hfma(a, c, __nv_bfloat16(-float(__nv_bfloat16(b * s))));
    out[row * width + col + half] = __hfma(b, c, __nv_bfloat16(a * s));
  }
}
}

std::pair<at::Tensor, at::Tensor> gemma3_qk_norm_rotary_cuda(const at::Tensor& q,
                                                             const at::Tensor& k,
                                                             const at::Tensor& qs,
                                                             const at::Tensor& ks,
                                                             const at::Tensor& cosine,
                                                             const at::Tensor& sine,
                                                             const at::Tensor& positions,
                                                             double eps) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && k.dim() == 3 &&
                  k.device() == q.device() && k.scalar_type() == q.scalar_type() &&
                  k.size(0) == q.size(0) && q.size(2) == k.size(2) && q.size(2) > 0 &&
                  q.size(2) <= 8192 && q.size(2) % 2 == 0 && q.size(1) > 0 && k.size(1) > 0 &&
                  q.stride(2) == 1 && k.stride(2) == 1 && q.stride(1) == q.size(2) &&
                  k.stride(1) == k.size(2),
              "Invalid Gemma3 packed Q/K norm layout");
  const auto width = q.size(2), rows = q.size(0) * (q.size(1) + k.size(1));
  TORCH_CHECK(q.size(0) > 0 && rows <= INT32_MAX, "Gemma3 Q/K norm row bounds");
  for (const auto& scale : {qs, ks})
    TORCH_CHECK(scale.device() == q.device() && scale.scalar_type() == at::kFloat &&
                    scale.is_contiguous() && scale.numel() == width,
                "Gemma3 Q/K norm requires contiguous FP32 scales");
  TORCH_CHECK(cosine.dim() == 2 && cosine.size(1) == width / 2 && cosine.sizes() == sine.sizes() &&
                  cosine.device() == q.device() && sine.device() == q.device() &&
                  cosine.scalar_type() == q.scalar_type() &&
                  sine.scalar_type() == q.scalar_type() && cosine.is_contiguous() &&
                  sine.is_contiguous() && positions.device() == q.device() &&
                  positions.scalar_type() == at::kLong && positions.is_contiguous() &&
                  positions.numel() == q.size(0),
              "Invalid Gemma3 rotary tables/positions");
  c10::cuda::CUDAGuard guard(q.device());
  auto qo = at::empty(q.sizes(), q.options()), ko = at::empty(k.sizes(), k.options());
  int threads = 32;
  while (threads < width && threads < 256)
    threads *= 2;
  gemma3_qk_norm_rotary<<<rows,
                          threads,
                          width * sizeof(__nv_bfloat16),
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.const_data_ptr()),
      qs.const_data_ptr<float>(),
      ks.const_data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(cosine.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sine.const_data_ptr()),
      positions.const_data_ptr<int64_t>(),
      reinterpret_cast<__nv_bfloat16*>(qo.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(ko.mutable_data_ptr()),
      width,
      q.size(1),
      k.size(1),
      q.stride(0),
      k.stride(0),
      q.size(0) * q.size(1),
      eps,
      cosine.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qo, ko};
}

// SPDX-License-Identifier: MIT
// Adapted from the unchanged included Candle gemma_rms_norm_bf16 kernel.
// Preserve its legacy Float reduction/rsqrt and BF16-rounded norm result,
// then perform the original BF16 residual addition in this same launch.
namespace {
extern "C" __global__ void gemma3_legacy_residual_norm_bf16(const __nv_bfloat16* x,
                                                            const float* scale,
                                                            const __nv_bfloat16* residual,
                                                            __nv_bfloat16* out,
                                                            unsigned int width,
                                                            unsigned int heads,
                                                            unsigned long long token_stride,
                                                            float eps) {
  const size_t base = static_cast<size_t>(blockIdx.x / heads) * token_stride +
                      static_cast<size_t>(blockIdx.x % heads) * width;
  const size_t out_base = static_cast<size_t>(blockIdx.x) * width;
  float sum = 0.f;
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    float v = __bfloat162float(x[base + col]);
    sum += v * v;
  }
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float partial[8];
  if ((threadIdx.x & 31) == 0)
    partial[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.f;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      partial[0] = rsqrtf(sum / width + eps);
  }
  __syncthreads();
  float inv = partial[0];
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    float normalized = __bfloat162float(x[base + col]) * inv;
    const auto rounded = __float2bfloat16_rn(normalized * scale[col]);
    out[out_base + col] = __hadd(residual[out_base + col], rounded);
  }
}

}

at::Tensor gemma3_legacy_residual_norm_cuda(const at::Tensor& residual,
                                            const at::Tensor& x,
                                            const at::Tensor& scale,
                                            double epsilon) {
  if (!x.is_cuda() || residual.device() != x.device() || scale.device() != x.device() ||
      x.scalar_type() != at::kBFloat16 || residual.scalar_type() != at::kBFloat16 ||
      scale.scalar_type() != at::kFloat || x.dim() != 2 || x.size(1) != 768 ||
      residual.sizes() != x.sizes() || !x.is_contiguous() || !residual.is_contiguous() ||
      !scale.is_contiguous() || scale.numel() != 768 || x.size(0) <= 0 || x.size(0) > INT32_MAX)
    return {};
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty(x.sizes(), x.options());
  gemma3_legacy_residual_norm_bf16<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
      768,
      1,
      x.stride(0),
      epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// SPDX-License-Identifier: MIT
// Adapted from the included Candle gemma_rms_norm_bf16 reduction and the
// legacy residual wrapper above. Retain both BF16 intermediate casts and
// the BF16 residual sum before reducing the following normalization.
namespace {
__device__ __forceinline__ float
gemma3_legacy_inverse(float sum, unsigned width, float epsilon, float* partial) {
  for (int offset = 16; offset; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((threadIdx.x & 31) == 0)
    partial[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.f;
    for (int offset = 16; offset; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0)
      partial[0] = rsqrtf(sum / width + epsilon);
  }
  __syncthreads();
  return partial[0];
}

__global__ void gemma3_residual_prenorm(const __nv_bfloat16* x,
                                        const __nv_bfloat16* residual,
                                        const float* post_scale,
                                        const float* pre_scale,
                                        __nv_bfloat16* states,
                                        __nv_bfloat16* normalized,
                                        unsigned width,
                                        unsigned long long xstride,
                                        unsigned long long rstride,
                                        float epsilon) {
  const auto base = static_cast<unsigned long long>(blockIdx.x) * xstride;
  const auto residual_base = static_cast<unsigned long long>(blockIdx.x) * rstride;
  const auto out_base = static_cast<unsigned long long>(blockIdx.x) * width;
  __shared__ float partial[8];
  __shared__ __nv_bfloat16 rounded_states[768];
  float sum = 0.f;
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float v = __bfloat162float(x[base + col]);
    sum += v * v;
  }
  const float inverse = gemma3_legacy_inverse(sum, width, epsilon, partial);
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float value = __bfloat162float(x[base + col]) * inverse;
    const auto rounded = __float2bfloat16_rn(value * post_scale[col]);
    const auto state = __hadd(residual[residual_base + col], rounded);
    rounded_states[col] = state;
    states[out_base + col] = state;
  }
  __syncthreads();
  sum = 0.f;
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float v = __bfloat162float(rounded_states[col]);
    sum += v * v;
  }
  const float next_inverse = gemma3_legacy_inverse(sum, width, epsilon, partial);
  for (unsigned col = threadIdx.x; col < width; col += blockDim.x) {
    float value = __bfloat162float(rounded_states[col]) * next_inverse;
    normalized[out_base + col] = __float2bfloat16_rn(value * pre_scale[col]);
  }
}
}

std::pair<at::Tensor, at::Tensor> gemma3_residual_prenorm_cuda(const at::Tensor& residual,
                                                               const at::Tensor& x,
                                                               const at::Tensor& post_scale,
                                                               const at::Tensor& pre_scale,
                                                               double epsilon) {
  if (!x.is_cuda() || residual.device() != x.device() || x.scalar_type() != at::kBFloat16 ||
      residual.scalar_type() != at::kBFloat16 || x.dim() != 2 || x.size(1) != 768 ||
      residual.sizes() != x.sizes() || x.stride(1) != 1 || residual.stride(1) != 1 ||
      x.stride(0) < 768 || residual.stride(0) < 768 || x.size(0) <= 0 || x.size(0) > INT32_MAX)
    return {};
  for (const auto& scale : {post_scale, pre_scale})
    if (scale.device() != x.device() || scale.scalar_type() != at::kFloat ||
        !scale.is_contiguous() || scale.numel() != 768)
      return {};
  c10::cuda::CUDAGuard guard(x.device());
  auto states = at::empty(x.sizes(), x.options()), normalized = at::empty(x.sizes(), x.options());
  gemma3_residual_prenorm<<<x.size(0), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      post_scale.const_data_ptr<float>(),
      pre_scale.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(states.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(normalized.mutable_data_ptr()),
      768,
      x.stride(0),
      residual.stride(0),
      epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {states, normalized};
}

// Adapted from Candle rmsnorm_f32_math, fast_sum, upowf and scatter_add at revision
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Their complete MIT notice:
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
// The spatial gather visits each real pooling bin in the original sequential
// row-major scatter order, without synthetic image patches or token padding.
__global__ void exact_vision_norm(
    const __nv_bfloat16* x, const float* alpha, __nv_bfloat16* out, int width, float eps) {
  int row = blockIdx.x, tid = threadIdx.x;
  __shared__ float sums[1024];
  float sum = 0;
  for (int col = tid; col < width; col += blockDim.x) {
    float v = float(x[row * width + col]);
    sum = __fadd_rn(sum, __fmul_rn(v, v));
  }
  sums[tid] = sum;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tid < stride)
      sums[tid] = __fadd_rn(sums[tid], sums[tid + stride]);
  }
  __syncthreads();
  float variance = __fadd_rn(__fmul_rn(sums[0], float(1. / double(width))), eps);
  float denominator = __fsqrt_rn(variance);
  for (int col = tid; col < width; col += blockDim.x) {
    float normalized = __fdiv_rn(float(x[row * width + col]), denominator);
    out[row * width + col] = __float2bfloat16_rn(__fmul_rn(normalized, alpha[col]));
  }
}

at::Tensor gemma_vision_norm_cuda(const at::Tensor& x, const at::Tensor& scale, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && scale.scalar_type() == at::kFloat &&
                  x.is_contiguous() && scale.is_contiguous() && scale.device() == x.device() &&
                  x.dim() >= 2 && x.dim() <= 3 && x.size(-1) > 0 && x.size(-1) <= 8192 &&
                  scale.numel() == x.size(-1) && x.numel() > 0 && x.numel() <= INT32_MAX,
              "Invalid packed Gemma vision FP32 norm geometry");
  auto out = at::empty_like(x);
  int w = x.size(-1), threads = 1;
  while (threads < w && threads < 1024)
    threads *= 2;
  exact_vision_norm<<<x.numel() / w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      w,
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Audio uses runtime powf(-.5), unlike the vision sqrt/divide normalization.
// Keep the exponent a launch argument to preserve the unfused powf contract.
__global__ void exact_audio_norm(const __nv_bfloat16* x,
                                 const float* alpha,
                                 __nv_bfloat16* out,
                                 int width,
                                 float eps,
                                 float exponent) {
  int row = blockIdx.x, tid = threadIdx.x;
  __shared__ float sums[1024];
  float sum = 0;
  for (int col = tid; col < width; col += blockDim.x) {
    float v = float(x[row * width + col]);
    sum = __fadd_rn(sum, __fmul_rn(v, v));
  }
  sums[tid] = sum;
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    if (tid < stride)
      sums[tid] = __fadd_rn(sums[tid], sums[tid + stride]);
  }
  __syncthreads();
  float variance = __fadd_rn(__fmul_rn(sums[0], float(1. / double(width))), eps);
  float inverse = powf(variance, exponent);
  for (int col = tid; col < width; col += blockDim.x) {
    float normalized = __fmul_rn(float(x[row * width + col]), inverse);
    out[row * width + col] = __float2bfloat16_rn(__fmul_rn(normalized, alpha[col]));
  }
}

at::Tensor gemma_audio_norm_cuda(const at::Tensor& x, const at::Tensor& scale, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && scale.scalar_type() == at::kFloat &&
                  x.is_contiguous() && scale.is_contiguous() && scale.device() == x.device() &&
                  x.dim() >= 2 && x.dim() <= 3 && x.size(-1) > 0 && x.size(-1) <= 8192 &&
                  scale.numel() == x.size(-1) && x.numel() > 0 && x.numel() <= INT32_MAX,
              "Invalid packed Gemma audio FP32 norm geometry");
  auto out = at::empty_like(x);
  int w = x.size(-1), threads = 1;
  while (threads < w && threads < 1024)
    threads *= 2;
  exact_audio_norm<<<x.numel() / w, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      w,
      eps,
      -.5f);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

__global__ void exact_vision_pool(const __nv_bfloat16* x,
                                  float* out,
                                  int height,
                                  int width,
                                  int hidden,
                                  int kernel,
                                  float inv,
                                  float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int bins = (height / kernel) * (width / kernel);
  if (i >= bins * hidden)
    return;
  int bin = i / hidden, column = i % hidden, by = bin / (width / kernel),
      bx = bin % (width / kernel);
  float sum = 0;
  for (int y = 0; y < kernel; ++y)
    for (int xcol = 0; xcol < kernel; ++xcol) {
      int patch = (by * kernel + y) * width + bx * kernel + xcol;
      sum = __fadd_rn(sum, __fmul_rn(float(x[patch * hidden + column]), inv));
    }
  float rounded = float(__float2bfloat16_rn(sum));
  out[i] = __fmul_rn(rounded, scale);
}

at::Tensor
gemma_vision_pool_cuda(const at::Tensor& x, int64_t height, int64_t width, int64_t kernel) {
  c10::cuda::CUDAGuard guard(x.device());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.dim() == 2 && height > 0 &&
                  width > 0 && kernel > 0 && height <= INT32_MAX / width &&
                  height * width == x.size(0) && height % kernel == 0 && width % kernel == 0 &&
                  x.size(1) > 0 && x.numel() <= INT32_MAX,
              "Invalid packed Gemma vision spatial pool geometry");
  auto out =
      at::empty({height * width / (kernel * kernel), x.size(1)}, x.options().dtype(at::kFloat));
  exact_vision_pool<<<(out.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      out.mutable_data_ptr<float>(),
      height,
      width,
      x.size(1),
      kernel,
      float(1. / double(kernel * kernel)),
      float(sqrt(double(x.size(1)))));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// SPDX-License-Identifier: MIT
// Adapted from Candle qwen35_gdn_recurrent above; snapshot the complete prefix state.
template <int Lanes> __device__ float gdn_sum(float (&partial)[32 / Lanes]) {
  // Reconstruct the original 32-lane reduction before subgroup shuffles.
#pragma unroll
  for (int d = 16 / Lanes; d; d >>= 1) {
#pragma unroll
    for (int j = 0; j < d; ++j)
      partial[j] += partial[j + d];
  }
  float sum = partial[0];
  for (int d = Lanes / 2; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d, Lanes);
  return __shfl_sync(0xffffffff, sum, 0, Lanes);
}

template <int Lanes>
__global__ void qwen35_gdn_cached(const __nv_bfloat16* qkv,
                                  const float* qk,
                                  const float* decay_beta,
                                  const uint32_t* cu,
                                  __nv_bfloat16* out,
                                  int kh,
                                  int vh,
                                  const float* initial,
                                  float* final_state,
                                  const int64_t* sources) {
  constexpr int columns = 512 / Lanes, entries = 128 / Lanes, partials = 32 / Lanes;
  const int seq = blockIdx.x, head = blockIdx.y, col = blockIdx.z * columns + threadIdx.x / Lanes,
            lane = threadIdx.x % Lanes;
  const int qhead = head / (vh / kh), channels = (2 * kh + vh) * 128;
  // Split accumulators preserve the original 32-lane tree.
  float state[entries] = {};
  const auto source = initial ? sources[seq] : seq;
  if (initial)
    for (int i = 0; i < entries; ++i)
      state[i] = initial[((source * vh + head) * 128 + lane + i * Lanes) * 128 + col];
  constexpr int tile = 32;
  __shared__ float queries_keys[tile][256], gates[tile][2];
  __shared__ __nv_bfloat16 values[tile][columns], outputs[tile][columns];
  const uint32_t begin = cu[seq], end = cu[seq + 1];
  for (uint32_t start = begin; start < end; start += tile) {
    const int count = min(uint32_t(tile), end - start);
    __syncthreads();
    // Cooperatively stage each head once for all state columns.
    for (int i = threadIdx.x; i < count * 256; i += blockDim.x)
      queries_keys[i / 256][i % 256] = qk[(uint64_t(start + i / 256) * kh + qhead) * 256 + i % 256];
    for (int i = threadIdx.x; i < count * 2; i += blockDim.x)
      gates[i / 2][i % 2] = decay_beta[uint64_t(start + i / 2) * 2 * vh + head + (i % 2) * vh];
    for (int i = threadIdx.x; i < count * columns; i += blockDim.x)
      values[i / columns][i % columns] =
          qkv[uint64_t(start + i / columns) * channels + 2 * kh * 128 + head * 128 +
              blockIdx.z * columns + i % columns];
    __syncthreads();
#pragma unroll 2
    for (int row = 0; row < count; ++row) {
      float decay = gates[row][0], beta = gates[row][1];
      float q[entries], k[entries], memory[partials] = {};
#pragma unroll
      for (int i = 0; i < entries; ++i) {
        q[i] = queries_keys[row][lane + i * Lanes];
        k[i] = queries_keys[row][128 + lane + i * Lanes];
        state[i] *= decay;
        memory[i % partials] += state[i] * k[i];
      }
      const float remembered = gdn_sum<Lanes>(memory);
      float delta = (__bfloat162float(values[row][threadIdx.x / Lanes]) - remembered) * beta,
            sum[partials] = {};
#pragma unroll
      for (int i = 0; i < entries; ++i) {
        state[i] += k[i] * delta;
        sum[i % partials] += state[i] * q[i];
      }
      const float value = gdn_sum<Lanes>(sum);
      if (lane == 0)
        outputs[row][threadIdx.x / Lanes] = __float2bfloat16_rn(value);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count * columns; i += blockDim.x)
      out[(uint64_t(start + i / columns) * vh + head) * 128 + blockIdx.z * columns + i % columns] =
          outputs[i / columns][i % columns];
  }
  if (final_state)
    for (int i = 0; i < entries; ++i)
      final_state[((int64_t(seq) * vh + head) * 128 + lane + i * Lanes) * 128 + col] = state[i];
}

/// Build normalized Q/K and decay buffers, then select cached, chunked, or recurrent execution.
/// Convolution history is sequence-local; prefix snapshots retain FP32 recurrent state.
at::Tensor qwen35_delta_cuda(const at::Tensor& qkv,
                             const at::Tensor& z,
                             const at::Tensor& ab,
                             const at::Tensor& conv,
                             const at::Tensor& a_log,
                             const at::Tensor& dt_bias,
                             const at::Tensor& norm,
                             const PackedInput& input,
                             int64_t key_heads,
                             int64_t value_heads,
                             double epsilon,
                             const at::Tensor& initial,
                             at::Tensor* final_state,
                             const at::Tensor& history,
                             const at::Tensor& history_cumulative,
                             bool chunk_prefill,
                             int64_t cache_layer) {
  c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && z.scalar_type() == at::kBFloat16 &&
                  ab.scalar_type() == at::kBFloat16 && conv.scalar_type() == at::kBFloat16 &&
                  norm.scalar_type() == at::kBFloat16,
              "Packed DeltaNet CUDA kernels require BF16");
  TORCH_CHECK(key_heads > 0 && value_heads > 0 && value_heads <= 128 &&
                  value_heads % key_heads == 0,
              "Unsupported packed DeltaNet head geometry");
  auto channels = (2 * key_heads + value_heads) * 128, tokens = qkv.size(0);
  TORCH_CHECK(qkv.sizes() == at::IntArrayRef({tokens, channels}) &&
                  z.sizes() == at::IntArrayRef({tokens, value_heads * 128}) &&
                  ab.sizes() == at::IntArrayRef({tokens, 2 * value_heads}) &&
                  conv.size(0) == channels && conv.size(-1) > 0 && conv.size(-1) <= 16,
              "Invalid packed DeltaNet tensors");
  TORCH_CHECK(tokens <= INT32_MAX && tokens * channels <= UINT32_MAX,
              "DeltaNet token buffer exceeds kernel range");
  auto stream = at::cuda::getCurrentCUDAStream(qkv.device().index());
  auto convolution = at::empty_like(qkv),
       qk = at::empty({tokens * key_heads, 256}, qkv.options().dtype(at::kFloat));
  auto decay = at::empty({tokens, 2 * value_heads}, qkv.options().dtype(at::kFloat));
  auto recurrence = at::empty({tokens, value_heads, 128}, qkv.options()),
       output = at::empty_like(recurrence);
  auto bf = [](const at::Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
  };
  auto mut = [](at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.mutable_data_ptr()); };
  auto cu = reinterpret_cast<const uint32_t*>(input.cumulative.const_data_ptr<int32_t>());
  auto cached = prefix_recurrent(input, qkv, cache_layer, value_heads, conv.size(-1));
  if (input.prefix)
    qwen35_cached_conv_cuda(
        qkv, conv, input, cached.history, cached.history_lengths, convolution, true);
  else if (history.defined()) {
    TORCH_CHECK(history.device() == qkv.device() && history.scalar_type() == qkv.scalar_type() &&
                    history.is_contiguous() && history.dim() == 2 && history.size(1) == channels &&
                    history_cumulative.device() == qkv.device() &&
                    history_cumulative.scalar_type() == at::kInt &&
                    history_cumulative.is_contiguous() && history_cumulative.dim() == 1 &&
                    history_cumulative.numel() > 1 &&
                    input.block_prefix_sources.size() == size_t(input.batch) &&
                    input.block_cache_sources.device() == qkv.device() &&
                    input.block_cache_sources.scalar_type() == at::kLong &&
                    input.block_cache_sources.is_contiguous() &&
                    input.block_cache_sources.numel() == input.batch,
                "Invalid packed GDN convolution history");
    for (const auto source : input.block_prefix_sources)
      TORCH_CHECK(source >= 0 && source < history_cumulative.numel() - 1,
                  "Invalid packed GDN prefix source");
    packed_conv_vector<true><<<dim3(tokens, (channels / 4 + 255) / 256), 256, 0, stream>>>(
        bf(qkv),
        bf(conv),
        cu,
        mut(convolution),
        channels,
        conv.size(-1),
        input.batch,
        input.block_cache_sources.const_data_ptr<int64_t>(),
        bf(history),
        history_cumulative.const_data_ptr<int32_t>());
  } else
    packed_conv_vector<<<dim3(tokens, (channels / 4 + 255) / 256), 256, 0, stream>>>(
        bf(qkv), bf(conv), cu, mut(convolution), channels, conv.size(-1), input.batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  qwen35_gdn_prepare<<<tokens * key_heads, 128, 0, stream>>>(bf(convolution),
                                                             bf(ab),
                                                             a_log.const_data_ptr<float>(),
                                                             dt_bias.const_data_ptr<float>(),
                                                             qk.mutable_data_ptr<float>(),
                                                             decay.mutable_data_ptr<float>(),
                                                             tokens,
                                                             key_heads,
                                                             value_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (initial.defined()) {
    TORCH_CHECK(history.defined() && initial.device() == qkv.device() &&
                    initial.scalar_type() == at::kFloat && initial.is_contiguous() &&
                    initial.dim() == 4 && initial.size(0) == history_cumulative.numel() - 1 &&
                    initial.size(1) == value_heads && initial.size(2) == 128 &&
                    initial.size(3) == 128,
                "Invalid packed cached GDN state");
  }
  // Prefill benefits from chunk-parallel work; short question suffixes keep the
  // register-resident recurrent path. Both consume/produce the same FP32 state.
#ifdef TEI_TORCH_GDN_HOPPER
  if (final_state && !initial.defined() && !input.prefix && tokens >= 256 &&
      at::cuda::getDeviceProperties(qkv.get_device())->major == 9) {
    qwen35_prefill_cuda(
        convolution, qk, decay, input, recurrence, key_heads, value_heads, final_state);
  } else
#endif
      if (input.prefix)
    qwen35_cached_recurrent_cuda(convolution,
                                 qk,
                                 decay,
                                 input,
                                 recurrence,
                                 key_heads,
                                 value_heads,
                                 true,
                                 cached.initial_sources,
                                 cached.checkpoint_positions,
                                 cached.checkpoint_rows,
                                 cached.checkpoints);
  else if (chunk_prefill && input.max_sequence >= 256) {
    qwen35_chunk_delta_cuda(
        convolution, qk, decay, input, recurrence, key_heads, value_heads, initial, final_state);
  } else if (initial.defined() || final_state) {
    if (final_state)
      *final_state =
          at::empty({input.batch, value_heads, 128, 128}, qkv.options().dtype(at::kFloat));
    // Wider column tiles amortize shared loads once enough heads fill the GPU.
#define CACHED(L)                                                                                  \
  qwen35_gdn_cached<L><<<dim3(input.batch, value_heads, L / 4), 512, 0, stream>>>(                 \
      bf(convolution),                                                                             \
      qk.const_data_ptr<float>(),                                                                  \
      decay.const_data_ptr<float>(),                                                               \
      cu,                                                                                          \
      mut(recurrence),                                                                             \
      key_heads,                                                                                   \
      value_heads,                                                                                 \
      initial.defined() ? initial.const_data_ptr<float>() : nullptr,                               \
      final_state ? final_state->mutable_data_ptr<float>() : nullptr,                              \
      initial.defined() ? input.block_cache_sources.const_data_ptr<int64_t>() : nullptr);
    if (input.batch * value_heads >= 256) {
      CACHED(8);
    } else {
      CACHED(16);
    }
#undef CACHED
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else if (qwen35_recurrent_supported(qkv.device(), key_heads, value_heads))
    qwen35_recurrent_cuda(convolution, qk, decay, input, recurrence, key_heads, value_heads, true);
  else {
    qwen35_gdn_recurrent<<<dim3(input.batch, value_heads, 16), 256, 0, stream>>>(
        bf(convolution),
        qk.const_data_ptr<float>(),
        decay.const_data_ptr<float>(),
        cu,
        mut(recurrence),
        key_heads,
        value_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  qwen35_gdn_norm<<<tokens * value_heads, 128, 0, stream>>>(
      bf(recurrence), bf(z), bf(norm), mut(output), epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (input.prefix)
    prefix_recurrent_commit(input, qkv, cache_layer, conv.size(-1), cached);
  return output.view({tokens, value_heads * 128});
}
}

// SPDX-License-Identifier: MIT
// Adapted from Candle036f39e candle-kernels/src/reduce.cu gemma4_rope2d.
// Preserve original MIT terms and explicit BF16 product rounding on Torch stream.
namespace tei {
namespace {
__global__ void rope2d(const __nv_bfloat16* x,
                       const __nv_bfloat16* c,
                       const __nv_bfloat16* s,
                       __nv_bfloat16* out,
                       int64_t count,
                       int64_t heads,
                       int64_t dim,
                       int64_t token_stride,
                       int64_t head_stride) {
  auto i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  auto d = i % dim, head = (i / dim) % heads, token = i / (dim * heads);
  auto part = d % (dim / 2), quarter = dim / 4,
       partner = part < quarter ? d + quarter : d - quarter;
  auto base = token * token_stride + head * head_stride;
  auto a = __bfloat162float(x[base + d]), b = __bfloat162float(x[base + partner]);
  if (part < quarter)
    b = -b;
  auto frequency = token * (dim / 2) + (d / (dim / 2)) * quarter + part % quarter;
  auto p = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, __bfloat162float(c[frequency]))));
  auto q = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, __bfloat162float(s[frequency]))));
  out[i] = __float2bfloat16_rn(__fadd_rn(p, q));
}
}

at::Tensor
gemma_vision_rotary_cuda(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.stride(2) == 1 &&
                  x.size(2) > 0 && x.size(2) % 4 == 0,
              "Invalid packed vision rotary rows");
  TORCH_CHECK(x.numel() <= int64_t(INT32_MAX) * 256, "Vision rotary launch exceeds kernel range");
  auto dim = x.size(2);
  TORCH_CHECK(cos.numel() == x.size(0) * dim / 2 && sin.sizes() == cos.sizes() &&
                  cos.is_contiguous() && sin.is_contiguous() &&
                  cos.scalar_type() == x.scalar_type() && sin.scalar_type() == x.scalar_type() &&
                  cos.device() == x.device() && sin.device() == x.device(),
              "Invalid packed vision rotary frequencies");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  if (x.numel())
    rope2d<<<(x.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)x.const_data_ptr(),
        (const __nv_bfloat16*)cos.const_data_ptr(),
        (const __nv_bfloat16*)sin.const_data_ptr(),
        (__nv_bfloat16*)out.mutable_data_ptr(),
        x.numel(),
        x.size(1),
        dim,
        x.stride(0),
        x.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
