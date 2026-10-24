// SPDX-License-Identifier: Apache-2.0 AND MIT
#include "kernels/decoder_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

// The following softmax is adapted from Candle 036f39e candle-kernels/src/reduce.cu,
// itself adapted from ggml. The original component attribution is retained.
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
extern "C" __device__ float __nv_expf(float);

namespace tei {
namespace {
__global__ void qwen_half_moe_softmax(const float* x, float* dst) {
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.y;
  float maximum = -INFINITY;
  for (int col = tid; col < 128; col += 32)
    maximum = fmaxf(maximum, x[row * 128 + col]);
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, mask, 32));
  float total = 0.f;
  for (int col = tid; col < 128; col += 32) {
    // Candle builds this kernel without --use_fast_math; preserve its expf.
    const float value = __nv_expf(x[row * 128 + col] - maximum);
    total += value;
    dst[row * 128 + col] = value;
  }
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    total += __shfl_xor_sync(0xffffffff, total, mask, 32);
  // Original Candle promotes the reciprocal division through its double literal.
  const float reciprocal = 1. / total;
  for (int col = tid; col < 128; col += 32)
    dst[row * 128 + col] *= reciprocal;
}

__global__ void
qwen_half_moe_weights(const float* selected, __half* output, int64_t rows, bool renormalize) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  float divisor = renormalize ? 0.f : 1.f;
  if (renormalize)
    for (int j = 0; j < 8; ++j)
      divisor += selected[row * 8 + j];
  for (int j = 0; j < 8; ++j)
    output[row * 8 + j] = __float2half_rn(__fdiv_rn(selected[row * 8 + j], divisor));
}
}

at::Tensor decoder_half_moe_probabilities(const at::Tensor& logits) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 2 &&
                  logits.size(1) == 128 && logits.is_contiguous() && logits.size(0) <= UINT32_MAX,
              "Original Half MoE softmax requires real contiguous FP32 rows with128 experts");
  c10::cuda::CUDAGuard guard(logits.device());
  auto result = at::empty_like(logits);
  if (!logits.size(0))
    return result;
  qwen_half_moe_softmax<<<logits.size(0), dim3(1, 32), 0, at::cuda::getCurrentCUDAStream()>>>(
      logits.const_data_ptr<float>(), result.mutable_data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}

at::Tensor decoder_half_moe_route_weights(const at::Tensor& selected, bool renormalize) {
  TORCH_CHECK(selected.is_cuda() && selected.scalar_type() == at::kFloat && selected.dim() == 2 &&
                  selected.size(1) == 8 && selected.is_contiguous(),
              "Original Half MoE normalization requires contiguous FP32 top8 probabilities");
  c10::cuda::CUDAGuard guard(selected.device());
  auto result = at::empty(selected.sizes(), selected.options().dtype(at::kHalf));
  if (!selected.size(0))
    return result;
  qwen_half_moe_weights<<<(selected.size(0) + 255) / 256,
                          256,
                          0,
                          at::cuda::getCurrentCUDAStream()>>>(
      selected.const_data_ptr<float>(),
      reinterpret_cast<__half*>(result.mutable_data_ptr()),
      selected.size(0),
      renormalize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
}
