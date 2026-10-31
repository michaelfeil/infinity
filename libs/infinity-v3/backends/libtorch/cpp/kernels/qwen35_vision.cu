// Adapted from Candle candle-kernels/src/reduce.cu, revision 036f39e.
// Original Apache-2.0/MIT licenses retained in vendor/qwen35_vision.
// Original ggml attribution retained below. Match default FMA, no fast math.
#include "qwen35_vision.h"
#include <climits>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#define WARP_SIZE 32

__device__ __forceinline__ float maxg(float a, float b) {
  return fmaxf(a, b);
}

__device__ __forceinline__ float expg(float a) {
  return expf(a);
}

// Softmax implementation adapted from ggml.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L4159
template <typename T, typename ACC> __device__ void softmax(const T* x, T* dst, const int ncols) {
  const int row = blockDim.x * blockIdx.x + threadIdx.x;
  const int block_size = blockDim.y;
  const int tid = threadIdx.y;

  T max_val = -INFINITY;

  for (int col = tid; col < ncols; col += block_size) {
    const int i = row * ncols + col;
    max_val = maxg(max_val, x[i]);
  }

  // find the max value in the block
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    max_val = maxg(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
  }

  ACC tmp = 0.;

  for (int col = tid; col < ncols; col += block_size) {
    const int i = row * ncols + col;
    const T val = expg(x[i] - max_val);
    tmp += static_cast<ACC>(val);
    dst[i] = val;
  }

  // sum up partial sums
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
  }

  const ACC inv_tmp = 1. / tmp;

  for (int col = tid; col < ncols; col += block_size) {
    const int i = row * ncols + col;
    dst[i] *= inv_tmp;
  }
}

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
    a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
  }
  return a;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, 32);
  }
  return x;
}

// LayerNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L477
template <typename T>
__device__ void layernorm(const T* x,
                          T* dst,
                          const T* alpha,
                          const T* beta,
                          const int ncols,
                          const int block_size,
                          const float eps) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  const int tid = threadIdx.x;

  float2 mean_var = make_float2(0.f, 0.f);

  for (int col = tid; col < ncols; col += block_size) {
    const float xi = x[row * ncols + col];
    mean_var.x += xi;
    mean_var.y += xi * xi;
  }

  // sum up partial sums
  mean_var = warp_reduce_sum(mean_var);
  if (block_size > WARP_SIZE) {
    __shared__ float2 s_sum[32];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
      s_sum[warp_id] = mean_var;
    }
    __syncthreads();
    mean_var = s_sum[lane_id];
    mean_var = warp_reduce_sum(mean_var);
  }

  const float mean = mean_var.x / ncols;
  const float var = mean_var.y / ncols - mean * mean;
  const float inv_std = rsqrtf(var + eps);

  if (alpha == nullptr && beta == nullptr) {
    for (int col = tid; col < ncols; col += block_size) {
      float lhs = (static_cast<float>(x[row * ncols + col]) - mean) * inv_std;
      dst[row * ncols + col] = static_cast<T>(lhs);
    }
  } else if (alpha == nullptr && beta != nullptr) {
    for (int col = tid; col < ncols; col += block_size) {
      float b = static_cast<float>(beta[col]);
      float lhs = (static_cast<float>(x[row * ncols + col]) - mean) * inv_std;
      dst[row * ncols + col] = static_cast<T>(lhs + b);
    }
  } else if (alpha != nullptr && beta == nullptr) {
    for (int col = tid; col < ncols; col += block_size) {
      float a = static_cast<float>(alpha[col]);
      float lhs = (static_cast<float>(x[row * ncols + col]) - mean) * inv_std;
      dst[row * ncols + col] = static_cast<T>(lhs * a);
    }
  } else {
    for (int col = tid; col < ncols; col += block_size) {
      float a = static_cast<float>(alpha[col]);
      float b = static_cast<float>(beta[col]);
      float lhs = (static_cast<float>(x[row * ncols + col]) - mean) * inv_std;
      dst[row * ncols + col] = static_cast<T>(lhs * a + b);
    }
  }
}

__global__ void exact_softmax_kernel(const float* x, float* y, int cols) {
  softmax<float, float>(x, y, cols);
}

__global__ void exact_ln_kernel(const __half* x,
                                __half* y,
                                const __half* weight,
                                const __half* bias,
                                int cols,
                                int threads,
                                float eps) {
  layernorm<__half>(x, y, weight, bias, cols, threads, eps);
}

namespace infinity {
namespace {
template <typename Input, typename Output>
__global__ void vision_rotary_kernel(const Input* qkv,
                                     const float* cosine,
                                     const float* sine,
                                     Output* output,
                                     int count,
                                     int heads,
                                     int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const int lane = index % dim, row = index / dim, token = row / heads;
  const int half = dim / 2, pair = lane % half;
  const int source = token * 3 * heads * dim + (row % heads) * dim;
  const float c = cosine[token * half + pair], s = sine[token * half + pair];
  for (int projection = 0; projection < 2; ++projection) {
    const int start = source + projection * heads * dim;
    const float a = float(qkv[start + pair]), b = float(qkv[start + pair + half]);
    // Match the separate FP32 tensor products before the final model-dtype cast.
    const float value = lane < half ? __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s))
                                    : __fadd_rn(__fmul_rn(b, c), __fmul_rn(a, s));
    output[projection * count + index] = Output(value);
  }
  output[2 * count + index] = Output(qkv[source + 2 * heads * dim + lane]);
}
}

at::Tensor vision_rotary_qkv(const at::Tensor& qkv,
                             const at::Tensor& cosine,
                             const at::Tensor& sine,
                             at::ScalarType output_type) {
  TORCH_CHECK(qkv.is_cuda() && qkv.is_contiguous() && qkv.dim() == 4 && qkv.size(1) == 3 &&
                  qkv.size(2) > 0 && qkv.size(3) > 0 && qkv.size(3) % 2 == 0 &&
                  qkv.numel() <= INT_MAX &&
                  (qkv.scalar_type() == at::kHalf || qkv.scalar_type() == at::kBFloat16 ||
                   qkv.scalar_type() == at::kFloat) &&
                  (output_type == qkv.scalar_type() || output_type == at::kFloat),
              "Invalid vision rotary QKV geometry or dtype");
  for (const auto& phase : {cosine, sine})
    TORCH_CHECK(phase.device() == qkv.device() && phase.scalar_type() == at::kFloat &&
                    phase.is_contiguous() &&
                    phase.sizes() == at::IntArrayRef({qkv.size(0), 1, qkv.size(3) / 2}),
                "Invalid vision rotary phases");
  c10::cuda::CUDAGuard guard(qkv.device());
  auto output =
      at::empty({3, qkv.size(0), qkv.size(2), qkv.size(3)}, qkv.options().dtype(output_type));
  const int count = qkv.numel() / 3;
  if (count == 0)
    return output;
  auto launch = [&]<typename Input, typename Output>() {
    vision_rotary_kernel<<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        qkv.const_data_ptr<Input>(),
        cosine.const_data_ptr<float>(),
        sine.const_data_ptr<float>(),
        output.mutable_data_ptr<Output>(),
        count,
        int(qkv.size(2)),
        int(qkv.size(3)));
  };
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, qkv.scalar_type(), "vision_rotary_qkv", [&] {
        if (output_type == at::kFloat)
          launch.template operator()<scalar_t, float>();
        else
          launch.template operator()<scalar_t, scalar_t>();
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

/// Apply the reference warp softmax to actual last-dimension rows without padding scores.
at::Tensor qwen35_vision_softmax(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(),
              "Contiguous CUDA F32 scores required");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  TORCH_CHECK(x.dim() > 0 && x.size(-1) > 0 && x.size(-1) <= INT_MAX && x.numel() <= INT_MAX,
              "Invalid Qwen3.5 vision softmax indexing range");
  int cols = x.size(-1);
  if (x.numel() == 0)
    return y;
  exact_softmax_kernel<<<x.numel() / cols, dim3(1, 32, 1), 0, at::cuda::getCurrentCUDAStream()>>>(
      x.const_data_ptr<float>(), y.data_ptr<float>(), cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor
qwen35_vision_layer_norm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, float eps) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous(),
              "Contiguous CUDA Half required");
  c10::cuda::CUDAGuard guard(x.device());
  auto y = at::empty_like(x);
  TORCH_CHECK(x.dim() > 0 && x.size(-1) > 0 && x.size(-1) <= INT_MAX && x.numel() <= INT_MAX &&
                  w.dim() == 1 && b.dim() == 1 && w.numel() == x.size(-1) &&
                  b.numel() == x.size(-1) && w.is_contiguous() && b.is_contiguous() &&
                  w.device() == x.device() && b.device() == x.device() &&
                  w.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
              "Invalid Qwen3.5 vision LayerNorm geometry");
  if (x.numel() == 0)
    return y;
  int cols = x.size(-1), threads = cols < 1024 ? 32 : 1024;
  exact_ln_kernel<<<x.numel() / cols, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(x.const_data_ptr<at::Half>()),
      reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w.const_data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(b.const_data_ptr<at::Half>()),
      cols,
      threads,
      eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
}
