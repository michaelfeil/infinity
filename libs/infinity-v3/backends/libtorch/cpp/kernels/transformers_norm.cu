// Copyright Infinity Core Contributors. See the repository LICENSE.
/// \file
/// CUDA RMS normalization with Transformers residual and affine rounding semantics.

#include "kernels/transformers_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

namespace tei {
namespace {
/// Aligned four-scalar pack used by the RMS-normalization kernel.
template <class T> struct alignas(sizeof(T) * 4) Four {
  T value[4];
};

template <class T>
__global__ void transformers_norm_kernel(const T* input,
                                         const T* residual,
                                         const T* weight,
                                         T* output,
                                         T* summed,
                                         int width,
                                         float epsilon) {
  const int64_t row = blockIdx.x;
  float sum = 0;
  for (int col = threadIdx.x * 4; col < width; col += blockDim.x * 4) {
    auto x = *reinterpret_cast<const Four<T>*>(input + row * width + col);
    Four<T> r{};
    if (residual)
      r = *reinterpret_cast<const Four<T>*>(residual + row * width + col);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (residual)
        x.value[i] = T(float(x.value[i]) + float(r.value[i]));
      const float value = float(x.value[i]);
      sum += __fmul_rn(value, value);
    }
    if (residual)
      *reinterpret_cast<Four<T>*>(summed + row * width + col) = x;
  }
  for (int delta = 16; delta; delta /= 2)
    sum += __shfl_down_sync(0xffffffff, sum, delta);
  __shared__ float warps[8];
  if (!(threadIdx.x % 32))
    warps[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < blockDim.x / 32 ? warps[threadIdx.x] : 0.f;
    for (int delta = 16; delta; delta /= 2)
      sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (!threadIdx.x)
      warps[0] = rsqrtf(sum / width + epsilon);
  }
  __syncthreads();
  const float inverse = warps[0];
  const T* values = residual ? summed : input;
  for (int col = threadIdx.x * 4; col < width; col += blockDim.x * 4) {
    auto x = *reinterpret_cast<const Four<T>*>(values + row * width + col);
    auto gamma = *reinterpret_cast<const Four<T>*>(weight + col);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const T normalized = T(float(x.value[i]) * inverse);
      x.value[i] = T(float(normalized) * float(gamma.value[i]));
    }
    *reinterpret_cast<Four<T>*>(output + row * width + col) = x;
  }
}
}

std::pair<at::Tensor, at::Tensor> transformers_rms_norm(const at::Tensor& input,
                                                        const at::Tensor& residual,
                                                        const at::Tensor& weight,
                                                        double epsilon) {
  const auto width = input.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.numel() == width && weight.device() == input.device() &&
                  weight.scalar_type() == input.scalar_type(),
              "Invalid Transformers RMSNorm weight");
  TORCH_CHECK(!residual.defined() || (residual.device() == input.device() &&
                                      residual.scalar_type() == input.scalar_type() &&
                                      residual.sizes() == input.sizes()),
              "Invalid Transformers RMSNorm residual");
  const auto alignment = input.element_size() * 4;
  const bool aligned = reinterpret_cast<uintptr_t>(input.const_data_ptr()) % alignment == 0 &&
                       reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % alignment == 0 &&
                       (!residual.defined() ||
                        reinterpret_cast<uintptr_t>(residual.const_data_ptr()) % alignment == 0);
  const bool supported =
      input.is_cuda() && aligned && input.is_contiguous() && weight.is_contiguous() &&
      (!residual.defined() || residual.is_contiguous()) && width > 0 && width % 4 == 0 &&
      width <= 32768 && (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16);
  if (!supported) {
    auto summed = residual.defined() ? input + residual : input;
    auto f = summed.to(at::kFloat);
    return {(f * at::rsqrt(f.square().mean(-1, true) + epsilon)).to(input.scalar_type()) * weight,
            summed};
  }
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input), summed = residual.defined() ? at::empty_like(input) : input;
  if (!input.numel())
    return {output, summed};
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto rows = input.numel() / width;
#define RUN(T)                                                                                     \
  transformers_norm_kernel<T><<<rows, 256, 0, stream>>>(                                           \
      reinterpret_cast<const T*>(input.const_data_ptr()),                                          \
      residual.defined() ? reinterpret_cast<const T*>(residual.const_data_ptr()) : nullptr,        \
      reinterpret_cast<const T*>(weight.const_data_ptr()),                                         \
      reinterpret_cast<T*>(output.mutable_data_ptr()),                                             \
      reinterpret_cast<T*>(summed.mutable_data_ptr()),                                             \
      width,                                                                                       \
      epsilon)
  if (input.scalar_type() == at::kHalf) {
    RUN(__half);
  } else {
    RUN(__nv_bfloat16);
  }
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, summed};
}
}
