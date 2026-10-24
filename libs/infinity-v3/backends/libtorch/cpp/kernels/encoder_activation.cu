// SPDX-License-Identifier: MIT
// GELU arithmetic follows candle-kernels/src/unary.cu from Candle revision
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original permission notice:
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
/// \file
/// CUDA encoder GELU and bias-plus-GELU kernels with reference rounding.

#include "kernels/encoder_activation.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tei {
namespace {
__device__ __forceinline__ half model_fma(half a, half b, half c) {
  return __hfma(a, b, c);
}

__device__ __forceinline__ nv_bfloat16 model_fma(nv_bfloat16 a, nv_bfloat16 b, nv_bfloat16 c) {
  return __hfma(a, b, c);
}

__device__ __forceinline__ float model_fma(float a, float b, float c) {
  return fmaf(a, b, c);
}

template <class T> __device__ __forceinline__ T gelu(T x) {
  T square = T(x * x), cube = T(square * x);
  T alpha = model_fma(T(0.044715), cube, x);
  T argument = T(T(0.7978845608028654) * alpha);
  T tangent = T(tanhf(float(argument)));
  return T(T(T(0.5) * x) * T(T(1.0) + tangent));
}

/// Aligned group of four scalar values used by vectorized activation loads and stores.
template <class T> struct alignas(sizeof(T) * 4) Four {
  T values[4];
};

template <class T> __global__ void gelu_kernel(const T* input, T* output, int64_t count) {
  const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i + 3 < count && reinterpret_cast<uintptr_t>(input) % alignof(Four<T>) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Four<T>) == 0) {
    auto values = *reinterpret_cast<const Four<T>*>(input + i);
#pragma unroll
    for (int j = 0; j < 4; ++j)
      values.values[j] = gelu(values.values[j]);
    *reinterpret_cast<Four<T>*>(output + i) = values;
  } else {
    for (int64_t j = i; j < count && j < i + 4; ++j)
      output[j] = gelu(input[j]);
  }
}

template <class T> void launch(const at::Tensor& input, at::Tensor& output, cudaStream_t stream) {
  const int64_t blocks = (input.numel() + 1023) / 1024;
  gelu_kernel<T><<<blocks, 256, 0, stream>>>(reinterpret_cast<const T*>(input.const_data_ptr()),
                                             reinterpret_cast<T*>(output.mutable_data_ptr()),
                                             input.numel());
}

template <class T>
__global__ void
bias_gelu_kernel(const T* input, const T* bias, T* output, int64_t count, int64_t width) {
  const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    if (i + j < count) {
      // ATen add rounds model-dtype GEMM output+bias before Candle unary GELU.
      T added = T(input[i + j] + bias[(i + j) % width]);
      output[i + j] = gelu(added);
    }
}

template <class T>
void launch_bias(const at::Tensor& input,
                 const at::Tensor& bias,
                 at::Tensor& output,
                 cudaStream_t stream) {
  bias_gelu_kernel<T><<<(input.numel() + 1023) / 1024, 256, 0, stream>>>(
      reinterpret_cast<const T*>(input.const_data_ptr()),
      reinterpret_cast<const T*>(bias.const_data_ptr()),
      reinterpret_cast<T*>(output.mutable_data_ptr()),
      input.numel(),
      input.size(-1));
}
}

at::Tensor encoder_gelu(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
              "Exact encoder GELU requires contiguous CUDA input");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Exact encoder GELU supports FP16/BF16/FP32");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  if (!input.numel())
    return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  if (input.scalar_type() == at::kHalf)
    launch<half>(input, output, stream);
  else if (input.scalar_type() == at::kBFloat16)
    launch<nv_bfloat16>(input, output, stream);
  else
    launch<float>(input, output, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor encoder_bias_gelu(const at::Tensor& input, const at::Tensor& bias) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dim() == 2 && bias.dim() == 1 &&
                  bias.size(0) == input.size(1) && bias.is_contiguous() &&
                  bias.device() == input.device() && bias.scalar_type() == input.scalar_type(),
              "Invalid fused bias-GELU");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 ||
                  input.scalar_type() == at::kFloat,
              "Fused bias-GELU supports FP16/BF16/FP32");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty_like(input);
  if (!input.numel())
    return output;
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  if (input.scalar_type() == at::kHalf)
    launch_bias<half>(input, bias, output, stream);
  else if (input.scalar_type() == at::kBFloat16)
    launch_bias<nv_bfloat16>(input, bias, output, stream);
  else if (input.scalar_type() == at::kFloat)
    launch_bias<float>(input, bias, output, stream);
  else
    TORCH_CHECK(false, "Invalid fused bias-GELU dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}
