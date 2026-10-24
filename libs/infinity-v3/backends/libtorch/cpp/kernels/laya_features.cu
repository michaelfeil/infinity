// Fork contribution; see LICENSE and LICENSING.md at the repository root.
#include "kernels/laya_features.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

namespace tei {
namespace {
constexpr int threads = 128;

__global__ void features_kernel(const float* logits, const int64_t* offsets, float* features) {
  const auto row = blockIdx.x;
  const auto start = offsets[row], end = offsets[row + 1], count = end - start;
  __shared__ float first[threads], second[threads], sum[threads];
  float a = -INFINITY, b = -INFINITY;
  for (auto i = start + threadIdx.x; i < end; i += blockDim.x) {
    const auto value = logits[i];
    if (value >= a) {
      b = a;
      a = value;
    } else
      b = fmaxf(b, value);
  }
  first[threadIdx.x] = a;
  second[threadIdx.x] = b;
  __syncthreads();
  for (int stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      const auto other = threadIdx.x + stride;
      second[threadIdx.x] =
          fmaxf(fminf(first[threadIdx.x], first[other]), fmaxf(second[threadIdx.x], second[other]));
      first[threadIdx.x] = fmaxf(first[threadIdx.x], first[other]);
    }
    __syncthreads();
  }
  const auto maximum = first[0], runner_up = second[0];
  float total = 0;
  for (auto i = start + threadIdx.x; i < end; i += blockDim.x)
    total += expf(logits[i] - maximum);
  sum[threadIdx.x] = total;
  __syncthreads();
  for (int stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  const auto denominator = sum[0];
  float entropy = 0;
  for (auto i = start + threadIdx.x; i < end; i += blockDim.x) {
    const auto p = expf(logits[i] - maximum) / denominator;
    entropy -= p * logf(fmaxf(p, 1e-9f));
  }
  // Each thread has captured the denominator before reusing the shared reduction.
  __syncthreads();
  sum[threadIdx.x] = entropy;
  __syncthreads();
  for (int stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const auto top1 = 1.f / denominator,
               top2 = count > 1 ? expf(runner_up - maximum) / denominator : 0.f;
    features[row * 4] = top1;
    features[row * 4 + 1] = top1 - top2;
    features[row * 4 + 2] = sum[0] / float(log(double(count > 2 ? count : 2)));
    features[row * 4 + 3] = float(double(count > 2 ? count : 2) / 255.);
  }
}
} // namespace

at::Tensor laya_action_features(const at::Tensor& logits, const at::Tensor& offsets) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kFloat && logits.dim() == 1 &&
                  logits.is_contiguous(),
              "Laya features require packed contiguous CUDA F32 logits");
  TORCH_CHECK(offsets.device() == logits.device() && offsets.scalar_type() == at::kLong &&
                  offsets.dim() == 1 && offsets.is_contiguous() && offsets.numel() > 1,
              "Laya features require same-device contiguous Int64 offsets");
  const auto batch = offsets.numel() - 1;
  TORCH_CHECK(batch <= std::numeric_limits<unsigned int>::max(),
              "Laya feature batch exceeds CUDA grid range");
  c10::cuda::CUDAGuard guard(logits.device());
  auto output = at::empty({batch, 4}, logits.options());
  features_kernel<<<static_cast<unsigned int>(batch),
                    threads,
                    0,
                    at::cuda::getCurrentCUDAStream(logits.device().index())>>>(
      logits.const_data_ptr<float>(),
      offsets.const_data_ptr<int64_t>(),
      output.mutable_data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
} // namespace tei
