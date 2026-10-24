// SPDX-License-Identifier: Apache-2.0
// Reuses the original packed reduction implementation from this TEI fork.
#include "batching/packed_pool.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
extern "C" __global__ void
packed_mean_pool_bf16(const __nv_bfloat16*, const uint32_t*, __nv_bfloat16*, uint32_t);
#include "../../../candle/src/pooling_kernels/mean_pool.cu"

namespace {
// Same model-dtype tree as Candle, reading existing cumulative offsets.
template <typename T>
__device__ void
packed_mean_cumulative(const T* input, const uint32_t* spans, T* output, uint32_t width) {
  __shared__ T partial[1024 * 8];
  const uint32_t start = spans[blockIdx.y];
  const uint32_t length = spans[blockIdx.y + 1] - start;
  uint32_t threads = 1;
  while (threads < length && threads < 1024)
    threads *= 2;
  const uint32_t base = blockIdx.x * 8;
  for (uint32_t i = threadIdx.x; i < threads * 8; i += blockDim.x) {
    const uint32_t feature = base + i % 8;
    T sum = T(0.f);
    if (feature < width) {
      for (uint64_t row = i / 8; row < length; row += threads)
        sum += input[(uint64_t(start) + row) * width + feature];
    }
    partial[i] = sum;
  }
  for (uint32_t step = threads / 2; step; step /= 2) {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < step * 8; i += blockDim.x)
      partial[i] += partial[i + step * 8];
  }
  __syncthreads();
  if (threadIdx.x < 8 && base + threadIdx.x < width) {
    const T reciprocal = T(1.0 / double(length));
    const T original = partial[threadIdx.x] * reciprocal + T(0.f);
    output[uint64_t(blockIdx.y) * width + base + threadIdx.x] =
        packed_mean_finite_fallback(input, start, length, width, base + threadIdx.x, original);
  }
}

__global__ void
packed_mean_cumulative_f16(const __half* x, const uint32_t* cu, __half* y, uint32_t width) {
  packed_mean_cumulative<__half>(x, cu, y, width);
}

__global__ void packed_mean_cumulative_bf16(const __nv_bfloat16* x,
                                            const uint32_t* cu,
                                            __nv_bfloat16* y,
                                            uint32_t width) {
  packed_mean_cumulative<__nv_bfloat16>(x, cu, y, width);
}
}

namespace tei {
at::Tensor packed_mean_pool(const at::Tensor& hidden, const at::Tensor& spans) {
  TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 && hidden.is_contiguous(),
              "Packed mean pooling requires contiguous CUDA hidden states");
  TORCH_CHECK(spans.device() == hidden.device() && spans.scalar_type() == at::kInt &&
                  spans.is_contiguous() && spans.dim() == 2 && spans.size(1) == 2,
              "Invalid packed mean pooling spans");
  TORCH_CHECK(hidden.size(1) > 0 && hidden.size(1) <= UINT32_MAX && spans.size(0) <= 65535,
              "Packed mean pooling dimensions exceed launch bounds");
  c10::cuda::CUDAGuard guard(hidden.device());
  auto output = at::empty({spans.size(0), hidden.size(1)}, hidden.options());
  if (!spans.size(0))
    return output;
  auto stream = at::cuda::getCurrentCUDAStream();
  auto boundaries = reinterpret_cast<const uint32_t*>(spans.const_data_ptr<int32_t>());
  dim3 grid((hidden.size(1) + 7) / 8, spans.size(0));
  if (hidden.scalar_type() == at::kHalf) {
    ::packed_mean_pool_f16<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(hidden.const_data_ptr()),
        boundaries,
        reinterpret_cast<__half*>(output.data_ptr()),
        hidden.size(1));
  } else if (hidden.scalar_type() == at::kBFloat16) {
    // The shared implementation guards the exported BF16 entry point for PTX
    // compilation. Instantiate its device routine here for host CUDA builds.
    TORCH_CHECK(at::cuda::getDeviceProperties(hidden.device().index())->major >= 8,
                "BF16 pooling requires SM80 or newer");
    ::packed_mean_pool_bf16<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr()),
        boundaries,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        hidden.size(1));
  } else {
    TORCH_CHECK(false, "Packed mean pooling supports float16 and bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor packed_mean_pool_cumulative(const at::Tensor& hidden, const at::Tensor& cumulative) {
  TORCH_CHECK(hidden.is_cuda() && hidden.dim() == 2 && hidden.is_contiguous(),
              "Packed cumulative mean requires contiguous CUDA hidden states");
  TORCH_CHECK(cumulative.device() == hidden.device() && cumulative.scalar_type() == at::kInt &&
                  cumulative.is_contiguous() && cumulative.dim() == 1 && cumulative.numel() >= 1,
              "Invalid packed cumulative mean offsets");
  const auto batch = cumulative.numel() - 1;
  TORCH_CHECK(hidden.size(1) > 0 && hidden.size(1) <= UINT32_MAX && batch <= 65535,
              "Packed cumulative mean dimensions exceed launch bounds");
  // Caller supplies already-validated strictly increasing packed boundaries;
  // never inspect device offsets on host or allocate synthetic metadata rows.
  c10::cuda::CUDAGuard guard(hidden.device());
  auto output = at::empty({batch, hidden.size(1)}, hidden.options());
  if (!batch)
    return output;
  const auto* boundaries = reinterpret_cast<const uint32_t*>(cumulative.const_data_ptr<int32_t>());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid((hidden.size(1) + 7) / 8, batch);
  if (hidden.scalar_type() == at::kHalf) {
    packed_mean_cumulative_f16<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __half*>(hidden.const_data_ptr()),
        boundaries,
        reinterpret_cast<__half*>(output.mutable_data_ptr()),
        hidden.size(1));
  } else if (hidden.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(at::cuda::getDeviceProperties(hidden.device().index())->major >= 8,
                "BF16 pooling requires SM80 or newer");
    packed_mean_cumulative_bf16<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.const_data_ptr()),
        boundaries,
        reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),
        hidden.size(1));
  } else {
    TORCH_CHECK(false, "Packed mean pooling supports float16 and bfloat16");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
}
