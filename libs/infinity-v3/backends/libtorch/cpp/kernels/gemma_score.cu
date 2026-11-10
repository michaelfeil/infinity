/// \file
/// CUDA Gemma sequence-score readout and residual/normalization stages.
/// Keeps the declared BF16 rounding boundaries between fused operations.

#include "kernels/gemma_score.h"
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace infinity {
namespace {
__global__ void gemma4_sequence_readout_kernel(const c10::BFloat16* logits,
                                               const int32_t* ends,
                                               c10::BFloat16* out,
                                               int64_t width,
                                               int64_t count,
                                               float reciprocal,
                                               float cap,
                                               bool capped) {
  const int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= count)
    return;
  const int64_t sequence = n / width, label = n % width;
  auto value = logits[(int64_t(ends[sequence + 1]) - 1) * width + label];
  if (capped) {
    // Preserve the publisher's BF16 rounding after each Torch scalar operation.
    auto divided = c10::BFloat16(float(value) * reciprocal);
    auto tangent = c10::BFloat16(::tanh(float(divided)));
    value = c10::BFloat16(float(tangent) * cap);
  }
  out[n] = value;
}
}

// ATen BF16 add and multiply each use FP32 opmath, rounding at both outputs.
__global__ void gemma4_residual_scale_kernel(const c10::BFloat16* residual,
                                             const c10::BFloat16* normalized,
                                             const c10::BFloat16* scale,
                                             c10::BFloat16* out,
                                             int64_t count) {
  const int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= count)
    return;
  auto added = c10::BFloat16(float(residual[n]) + float(normalized[n]));
  out[n] = c10::BFloat16(float(added) * float(scale[0]));
}

at::Tensor gemma4_bf16_residual_scale(const at::Tensor& residual,
                                      const at::Tensor& normalized,
                                      const at::Tensor& scale) {
  TORCH_CHECK(
      residual.is_cuda() && residual.scalar_type() == at::kBFloat16 && residual.is_contiguous() &&
          normalized.device() == residual.device() && normalized.scalar_type() == at::kBFloat16 &&
          normalized.is_contiguous() && normalized.sizes() == residual.sizes() &&
          scale.device() == residual.device() && scale.scalar_type() == at::kBFloat16 &&
          scale.is_contiguous() && scale.dim() == 1 && scale.numel() == 1,
      "Gemma4 residual scale expects original contiguous BF16 states and one-element layer scalar");
  c10::cuda::CUDAGuard guard(residual.device());
  auto out = at::empty_like(residual);
  const auto n = out.numel();
  if (!n)
    return out;
  gemma4_residual_scale_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      residual.data_ptr<c10::BFloat16>(),
      normalized.data_ptr<c10::BFloat16>(),
      scale.data_ptr<c10::BFloat16>(),
      out.data_ptr<c10::BFloat16>(),
      n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// SPDX-License-Identifier: MIT
// Adapted from the Candle-derived native Gemma reference norm in
// kernels/qwen35_kernels.cu. Its component LICENSE/NOTICE bundles are retained.
namespace {
// Preserve the current native donor's default-FMA accumulation explicitly,
// while retaining its separate FP32 mean/epsilon and sqrt/div tree.
// This TU keeps precise pointwise compilation with --fmad=false.
template <bool VIRTUAL_LANES, bool PAIRED = false, bool RESIDUAL = false>
__device__ __forceinline__ void
gemma4_rms_norm_residual_reference_impl(const __nv_bfloat16* x,
                                        const float* scale,
                                        __nv_bfloat16* out,
                                        unsigned int width,
                                        unsigned int heads,
                                        unsigned long long token_stride,
                                        float eps,
                                        const __nv_bfloat16* router_scale = nullptr,
                                        __nv_bfloat16 root = {},
                                        const __nv_bfloat16* residual = nullptr,
                                        const __nv_bfloat16* scalar = nullptr) {
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
        partial[lane] = __fmaf_rn(v, v, partial[lane]);
      }
    }
    for (unsigned int stride = lanes / 2; stride; stride >>= 1)
      for (unsigned int lane = 0; lane < stride; ++lane)
        partial[lane] += partial[lane + stride];
    sum = partial[0];
  } else {
    for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
      float v = __bfloat162float(x[base + col]);
      sum = __fmaf_rn(v, v, sum);
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
      auto added = __float2bfloat16_rn(__bfloat162float(residual[out_base + col]) +
                                       __bfloat162float(rounded));
      out[out_base + col] =
          scalar ? __float2bfloat16_rn(__bfloat162float(added) * __bfloat162float(scalar[0]))
                 : added;
    } else
      out[out_base + col] = __float2bfloat16_rn(normalized * scale[col]);
    if constexpr (PAIRED) {
      __nv_bfloat16 router = __float2bfloat16_rn(normalized * 1.f);
      router = router * root + __float2bfloat16_rn(0.f);
      out[static_cast<size_t>(gridDim.x) * width + out_base + col] = router * router_scale[col];
    }
  }
}

static __global__ void gemma4_norm_residual_kernel(const __nv_bfloat16* x,
                                                   const float* scale,
                                                   const __nv_bfloat16* residual,
                                                   const __nv_bfloat16* scalar,
                                                   __nv_bfloat16* out,
                                                   unsigned width,
                                                   float eps) {
  gemma4_rms_norm_residual_reference_impl<false, false, true>(
      x, scale, out, width, 1, width, eps, nullptr, {}, residual, scalar);
}
}

at::Tensor gemma4_bf16_norm_residual(const at::Tensor& residual,
                                     const at::Tensor& x,
                                     const at::Tensor& scale,
                                     double epsilon,
                                     const at::Tensor& scalar) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
                  x.is_contiguous() && residual.device() == x.device() &&
                  residual.sizes() == x.sizes() && residual.scalar_type() == x.scalar_type() &&
                  residual.is_contiguous() && scale.device() == x.device() &&
                  scale.scalar_type() == at::kFloat && scale.is_contiguous() &&
                  scale.numel() == x.size(1) && x.size(1) > 0 && x.size(1) <= 8192,
              "Gemma4 norm/residual expects original contiguous BF16 states and FP32 scales");
  TORCH_CHECK(!scalar.defined() ||
                  (scalar.device() == x.device() && scalar.scalar_type() == at::kBFloat16 &&
                   scalar.dim() == 1 && scalar.numel() == 1 && scalar.is_contiguous()),
              "Gemma4 layer scalar must be original one-element BF16 device storage");
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  int threads = 1;
  while (threads < x.size(1) && threads < 1024)
    threads *= 2;
  if (!out.numel())
    return out;
  gemma4_norm_residual_kernel<<<x.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      scale.const_data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(residual.const_data_ptr()),
      scalar.defined() ? reinterpret_cast<const __nv_bfloat16*>(scalar.const_data_ptr()) : nullptr,
      reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
      x.size(1),
      epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// Three pre-feedforward outputs from one reference RMS statistic. The reduction
// tree, separate FP32 mean/epsilon, sqrt/div and BF16 scale rounding mirror
// gemma_norm_cuda(..., reference=true) as used by Gemma4. This TU compiles with
// --fmad=false, so the donor's contracted square accumulation is spelled out
// with __fmaf_rn; the remaining adds are plain FP32 adds as in the donor.
namespace {
__global__ void gemma4_pre_ff_norm_kernel(const __nv_bfloat16* x,
                                          const float* dense_scale,
                                          const float* routed_scale,
                                          const __nv_bfloat16* router_scale,
                                          const __nv_bfloat16* root,
                                          __nv_bfloat16* dense,
                                          __nv_bfloat16* routed,
                                          float* router,
                                          unsigned int width,
                                          float eps) {
  const size_t base = static_cast<size_t>(blockIdx.x) * width;
  __shared__ float sums[1024];
  unsigned int logical_threads = 1;
  while (logical_threads < width && logical_threads < 1024)
    logical_threads <<= 1;
  const unsigned int lanes = logical_threads / blockDim.x;
  float partial[4] = {0.f, 0.f, 0.f, 0.f};
  for (unsigned int lane = 0; lane < lanes; ++lane)
    for (unsigned int col = threadIdx.x + lane * blockDim.x; col < width; col += logical_threads) {
      float v = __bfloat162float(x[base + col]);
      partial[lane] = __fmaf_rn(v, v, partial[lane]);
    }
  for (unsigned int stride = lanes / 2; stride; stride >>= 1)
    for (unsigned int lane = 0; lane < stride; ++lane)
      partial[lane] += partial[lane + stride];
  float sum = partial[0];
  sums[threadIdx.x] = sum;
  for (unsigned int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
    __syncthreads();
    if (threadIdx.x < stride)
      sums[threadIdx.x] += sums[threadIdx.x + stride];
  }
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
  const float root_value = __bfloat162float(root[0]);
  for (unsigned int col = threadIdx.x; col < width; col += blockDim.x) {
    const float normalized = __bfloat162float(x[base + col]) / sums[0];
    dense[base + col] = __float2bfloat16_rn(normalized * dense_scale[col]);
    routed[base + col] = __float2bfloat16_rn(normalized * routed_scale[col]);
    // norm(..., unit) rounds to BF16, then root and the learned router scale
    // each round again before the final FP32 cast.
    const __nv_bfloat16 unit = __float2bfloat16_rn(normalized * 1.f);
    const __nv_bfloat16 rooted = __float2bfloat16_rn(__bfloat162float(unit) * root_value);
    const __nv_bfloat16 scaled =
        __float2bfloat16_rn(__bfloat162float(rooted) * __bfloat162float(router_scale[col]));
    router[base + col] = __bfloat162float(scaled);
  }
}
} // namespace

Gemma4PreFeedforwardNorm gemma4_bf16_pre_feedforward_norm(const at::Tensor& x,
                                                          const at::Tensor& dense_scale,
                                                          const at::Tensor& routed_scale,
                                                          const at::Tensor& router_scale,
                                                          const at::Tensor& root,
                                                          double epsilon) {
  Gemma4PreFeedforwardNorm result;
  const bool supported =
      x.defined() && x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 &&
      x.is_contiguous() && x.size(1) > 0 && x.size(1) <= 8192 && dense_scale.defined() &&
      routed_scale.defined() && router_scale.defined() && root.defined() &&
      dense_scale.device() == x.device() && routed_scale.device() == x.device() &&
      router_scale.device() == x.device() && root.device() == x.device() &&
      dense_scale.scalar_type() == at::kFloat && routed_scale.scalar_type() == at::kFloat &&
      dense_scale.is_contiguous() && routed_scale.is_contiguous() &&
      dense_scale.numel() == x.size(1) && routed_scale.numel() == x.size(1) &&
      router_scale.scalar_type() == at::kBFloat16 && router_scale.is_contiguous() &&
      router_scale.numel() == x.size(1) && root.scalar_type() == at::kBFloat16 &&
      root.is_contiguous() && root.numel() == 1;
  if (!supported)
    return result;
  c10::cuda::CUDAGuard guard(x.device());
  result.dense = at::empty_like(x);
  result.routed = at::empty_like(x);
  result.router = at::empty(x.sizes(), x.options().dtype(at::kFloat));
  if (!x.numel())
    return result;
  const auto width = x.size(1);
  int threads = 1;
  while (threads < width && threads < 256)
    threads *= 2;
  gemma4_pre_ff_norm_kernel<<<x.size(0), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      dense_scale.const_data_ptr<float>(),
      routed_scale.const_data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(router_scale.const_data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(root.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(result.dense.mutable_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(result.routed.mutable_data_ptr()),
      result.router.mutable_data_ptr<float>(),
      static_cast<unsigned int>(width),
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

/// Select sequence-end scores from the full score projection, preserving BF16 soft-cap stages.
at::Tensor gemma4_bf16_sequence_score_readout(const at::Tensor& logits,
                                              const at::Tensor& cumulative,
                                              int64_t batch,
                                              double cap,
                                              bool capped) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kBFloat16 && logits.dim() == 2 &&
                  logits.is_contiguous(),
              "Gemma4 sequence readout expects complete contiguous CUDA BF16 score logits");
  TORCH_CHECK(cumulative.device() == logits.device() && cumulative.scalar_type() == at::kInt &&
                  cumulative.dim() == 1 && cumulative.is_contiguous() && batch >= 0 &&
                  cumulative.numel() >= batch + 1,
              "Gemma4 sequence readout expects current packed sequence ends on the score device");
  c10::cuda::CUDAGuard guard(logits.device());
  auto out = at::empty({batch, logits.size(1)}, logits.options());
  const auto n = out.numel();
  if (!n)
    return out;
  const float reciprocal = capped ? float(1. / cap) : 0.f;
  gemma4_sequence_readout_kernel<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.data_ptr<c10::BFloat16>(),
      cumulative.data_ptr<int32_t>(),
      out.data_ptr<c10::BFloat16>(),
      logits.size(1),
      n,
      reciprocal,
      float(cap),
      capped);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
}
