// SPDX-License-Identifier: MIT
#include <cuda_bf16.h>

// Apply the supplied scale and normalization in FP32, rounding only
// the final result to BF16. No full-sized FP32 activation buffers are needed.
extern "C" __global__ void gemma_rms_norm_bf16(const __nv_bfloat16* x,
                                               const float* scale,
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
    out[out_base + col] = __float2bfloat16_rn(normalized * scale[col]);
  }
}

// Preserve the reduction tree and FP32 operations of Candle's unfused path.
template <bool VIRTUAL_LANES, bool PAIRED = false, bool ADD_RESIDUAL = false>
__device__ __forceinline__ void
gemma_rms_norm_reference_impl(const __nv_bfloat16* x,
                              const float* scale,
                              __nv_bfloat16* out,
                              unsigned int width,
                              unsigned int heads,
                              unsigned long long token_stride,
                              float eps,
                              const __nv_bfloat16* router_scale = nullptr,
                              __nv_bfloat16 root = {},
                              const __nv_bfloat16* residual = nullptr,
                              unsigned long long residual_stride = 0) {
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
      float mean = sum * static_cast<float>(1.0 / static_cast<double>(width));
      sums[0] = sqrtf(mean + eps);
    }
  }
  __syncthreads();
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    float normalized = __bfloat162float(x[base + col]) / sums[0];
    __nv_bfloat16 value = __float2bfloat16_rn(normalized * scale[col]);
    if constexpr (ADD_RESIDUAL) {
      size_t residual_base = static_cast<size_t>(blockIdx.x) * residual_stride;
      value = residual[residual_base + col] + value;
    }
    out[out_base + col] = value;
    if constexpr (PAIRED) {
      __nv_bfloat16 router = __float2bfloat16_rn(normalized * 1.f);
      router = router * root + __float2bfloat16_rn(0.f);
      out[static_cast<size_t>(gridDim.x) * width + out_base + col] = router * router_scale[col];
    }
  }
}

extern "C" __global__ void gemma_rms_norm_reference_bf16(const __nv_bfloat16* x,
                                                         const float* scale,
                                                         __nv_bfloat16* out,
                                                         unsigned int width,
                                                         unsigned int heads,
                                                         unsigned long long token_stride,
                                                         float eps) {
  gemma_rms_norm_reference_impl<false>(x, scale, out, width, heads, token_stride, eps);
}

extern "C" __global__ void gemma_rms_norm_pair_bf16(const __nv_bfloat16* x,
                                                    const float* expert_scale,
                                                    const __nv_bfloat16* router_scale,
                                                    __nv_bfloat16* out,
                                                    unsigned int width,
                                                    unsigned long long token_stride,
                                                    float eps,
                                                    __nv_bfloat16 root) {
  gemma_rms_norm_reference_impl<false, true>(
      x, expert_scale, out, width, 1, token_stride, eps, router_scale, root);
}

extern "C" __global__ void gemma_rms_norm_pair_compact_bf16(const __nv_bfloat16* x,
                                                            const float* expert_scale,
                                                            const __nv_bfloat16* router_scale,
                                                            __nv_bfloat16* out,
                                                            unsigned int width,
                                                            unsigned long long token_stride,
                                                            float eps,
                                                            __nv_bfloat16 root) {
  gemma_rms_norm_reference_impl<true, true>(
      x, expert_scale, out, width, 1, token_stride, eps, router_scale, root);
}

// NeoX rotary embedding with BF16 rounding after each product, as in Candle's
// composed expression. Write packed [tokens, heads, width] directly.
extern "C" __global__ void gemma_rope_reference_bf16(const __nv_bfloat16* x,
                                                     const __nv_bfloat16* cos,
                                                     const __nv_bfloat16* sin,
                                                     __nv_bfloat16* out,
                                                     unsigned int count,
                                                     unsigned int heads,
                                                     unsigned int width,
                                                     unsigned long long head_stride,
                                                     unsigned long long token_stride,
                                                     unsigned long long col_stride) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  unsigned int col = i % width;
  unsigned int head = (i / width) % heads;
  unsigned int token = i / (width * heads);
  size_t base = static_cast<size_t>(head) * head_stride + static_cast<size_t>(token) * token_stride;
  unsigned int other_col = col < width / 2 ? col + width / 2 : col - width / 2;
  float a = __bfloat162float(x[base + static_cast<size_t>(col) * col_stride]);
  float b = __bfloat162float(x[base + static_cast<size_t>(other_col) * col_stride]);
  if (col < width / 2)
    b = -b;
  size_t freq = static_cast<size_t>(token) * width + col;
  float first = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a, __bfloat162float(cos[freq]))));
  float second = __bfloat162float(__float2bfloat16_rn(__fmul_rn(b, __bfloat162float(sin[freq]))));
  out[i] = __float2bfloat16_rn(__fadd_rn(first, second));
}

template <int LOGICAL>
__device__ __forceinline__ void warp_reference(const __nv_bfloat16* x,
                                               const float* scale,
                                               __nv_bfloat16* out,
                                               unsigned width,
                                               unsigned heads,
                                               unsigned long long token_stride,
                                               float eps,
                                               unsigned rows) {
  unsigned lane = threadIdx.x % 32;
  unsigned row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows)
    return;
  size_t base = size_t(row / heads) * token_stride + size_t(row % heads) * width;
  float partial[LOGICAL / 32] = {};
#pragma unroll
  for (int i = 0; i < LOGICAL / 32; ++i) {
    for (unsigned col = lane + i * 32; col < width; col += LOGICAL) {
      float v = __bfloat162float(x[base + col]);
      partial[i] += v * v;
    }
  }
#pragma unroll
  for (int stride = LOGICAL / 64; stride; stride >>= 1) {
#pragma unroll
    for (int i = 0; i < stride; ++i)
      partial[i] += partial[i + stride];
  }
  float sum = partial[0];
#pragma unroll
  for (int stride = 16; stride; stride >>= 1) {
    float other = __shfl_down_sync(0xffffffff, sum, stride);
    if (lane < stride)
      sum += other;
  }
  float denominator = 0.f;
  if (!lane)
    denominator = sqrtf(sum * float(1.0 / double(width)) + eps);
  denominator = __shfl_sync(0xffffffff, denominator, 0);
  for (unsigned col = lane; col < width; col += 32) {
    float value = __bfloat162float(x[base + col]) / denominator;
    out[size_t(row) * width + col] = __float2bfloat16_rn(value * scale[col]);
  }
}

extern "C" __global__ void gemma_rms_norm_compact_bf16(const __nv_bfloat16* x,
                                                       const float* scale,
                                                       __nv_bfloat16* out,
                                                       unsigned width,
                                                       unsigned heads,
                                                       unsigned long long stride,
                                                       float eps) {
  gemma_rms_norm_reference_impl<true>(x, scale, out, width, heads, stride, eps);
}

extern "C" __global__ void gemma_rms_norm_warp_bf16(const __nv_bfloat16* x,
                                                    const float* scale,
                                                    __nv_bfloat16* out,
                                                    unsigned width,
                                                    unsigned heads,
                                                    unsigned long long stride,
                                                    float eps,
                                                    unsigned rows) {
  warp_reference<256>(x, scale, out, width, heads, stride, eps, rows);
}

extern "C" __global__ void gemma_rms_norm_residual_bf16(const __nv_bfloat16* x,
                                                        const float* scale,
                                                        const __nv_bfloat16* residual,
                                                        __nv_bfloat16* out,
                                                        unsigned int width,
                                                        unsigned long long token_stride,
                                                        unsigned long long residual_stride,
                                                        float eps) {
  gemma_rms_norm_reference_impl<false, false, true>(
      x, scale, out, width, 1, token_stride, eps, nullptr, {}, residual, residual_stride);
}

extern "C" __global__ void gemma_rms_norm_residual_compact_bf16(const __nv_bfloat16* x,
                                                                const float* scale,
                                                                const __nv_bfloat16* residual,
                                                                __nv_bfloat16* out,
                                                                unsigned int width,
                                                                unsigned long long token_stride,
                                                                unsigned long long residual_stride,
                                                                float eps) {
  gemma_rms_norm_reference_impl<true, false, true>(
      x, scale, out, width, 1, token_stride, eps, nullptr, {}, residual, residual_stride);
}
