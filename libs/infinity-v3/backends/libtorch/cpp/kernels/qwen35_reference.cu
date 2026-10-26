#include "batching/prefix_cache.h"
// SPDX-License-Identifier: MIT
// Adapted from original Candle qwen35_gdn.cu and Candle fast_sum at
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original licenses retained.
// This translation unit alone uses the original --fmad=false contract.
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
#include "kernels/qwen35_reference.h"
#include "kernels/qwen35_recurrent.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#define gdn_sum tei_reference_gdn_sum
#define qwen35_conv_bf16 tei_reference_qwen35_conv_bf16
#define qwen35_gdn_prepare tei_reference_qwen35_gdn_prepare
#define qwen35_gdn_recurrent tei_reference_qwen35_gdn_recurrent
#define qwen35_gdn_norm tei_reference_qwen35_gdn_norm
#include "../../../candle/src/kernels/qwen35_gdn.cu"
#undef gdn_sum
#undef qwen35_conv_bf16
#undef qwen35_gdn_prepare
#undef qwen35_gdn_recurrent
#undef qwen35_gdn_norm

namespace infinity {
namespace {
// Each thread owns V adjacent logical lanes; preserve the original reduction tree.
template <int V> struct alignas(V * 2) NormValues {
  __nv_bfloat16 values[V];
};

template <int V, bool Residual = false>
__global__ void norm_tree(const __nv_bfloat16* x,
                          const __nv_bfloat16* gamma,
                          __nv_bfloat16* out,
                          int rows,
                          int width,
                          int heads,
                          long stride0,
                          long stride1,
                          float eps,
                          const __nv_bfloat16* addition = nullptr,
                          __nv_bfloat16* updated = nullptr) {
  __shared__ float sums[1024];
  int row = blockIdx.x, tid = threadIdx.x * V, threads = blockDim.x * V;
  long base = int64_t(row / heads) * stride0 + (row % heads) * stride1;
  float sum[V] = {};
  for (int d = tid; d < width; d += threads) {
    auto values = *reinterpret_cast<const NormValues<V>*>(x + base + d);
    if constexpr (Residual) {
      auto delta = *reinterpret_cast<const NormValues<V>*>(addition + base + d);
#pragma unroll
      for (int j = 0; j < V; ++j)
        values.values[j] = __float2bfloat16_rn(
            __fadd_rn(__bfloat162float(values.values[j]), __bfloat162float(delta.values[j])));
      *reinterpret_cast<NormValues<V>*>(updated + base + d) = values;
    }
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float z = __bfloat162float(values.values[j]);
      sum[j] = __fadd_rn(sum[j], __fmul_rn(z, z));
    }
  }
#pragma unroll
  for (int j = 0; j < V; ++j)
    sums[tid + j] = sum[j];
  for (int s = threads / 2; s >= 32; s >>= 1) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < V; ++j)
      if (tid + j < s)
        sums[tid + j] = __fadd_rn(sums[tid + j], sums[tid + j + s]);
  }
  if (tid < 32) {
    float value[V];
#pragma unroll
    for (int j = 0; j < V; ++j)
      value[j] = sums[tid + j];
    const unsigned mask = __activemask();
    for (int s = min(threads / 2, 16); s >= V; s >>= 1)
#pragma unroll
      for (int j = 0; j < V; ++j)
        value[j] = __fadd_rn(value[j], __shfl_down_sync(mask, value[j], s / V));
#pragma unroll
    for (int s = V / 2; s; s >>= 1)
#pragma unroll
      for (int j = 0; j < s; ++j)
        value[j] = __fadd_rn(value[j], value[j + s]);
    if (!tid) {
      float mean = __fmul_rn(value[0], 1.f / width);
      sums[0] = __fdiv_rn(1.f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
  }
  __syncthreads();
  float inv = sums[0];
  for (int d = tid; d < width; d += threads) {
    auto values = *reinterpret_cast<const NormValues<V>*>((Residual ? updated : x) + base + d);
    auto weights = *reinterpret_cast<const NormValues<V>*>(gamma + d);
    NormValues<V> result;
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float z = __bfloat162float(values.values[j]),
            weight = __fadd_rn(__bfloat162float(weights.values[j]), 1.f);
      result.values[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(z, inv), weight));
    }
    *reinterpret_cast<NormValues<V>*>(out + int64_t(row) * width + d) = result;
  }
}
}

at::Tensor qwen35_reference_norm_cuda(const at::Tensor& x, const at::Tensor& gamma, double eps) {
  TORCH_CHECK(x.is_cuda() && gamma.device() == x.device() && x.scalar_type() == at::kBFloat16 &&
              gamma.scalar_type() == at::kBFloat16 && gamma.dim() == 1 && gamma.is_contiguous());
  TORCH_CHECK((x.dim() == 2 || x.dim() == 3) && x.stride(-1) == 1);
  long width = x.size(-1);
  TORCH_CHECK(width > 0 && width <= 8192 && gamma.numel() == width);
  long heads = x.dim() == 3 ? x.size(1) : 1, rows = x.numel() / width;
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX);
  int threads = 1;
  while (threads < width && threads < 1024)
    threads *= 2;
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty(x.sizes(), x.options());
  bool vector = width % 4 == 0 && x.stride(0) % 4 == 0 && (x.dim() == 2 || x.stride(1) % 4 == 0) &&
                (reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 8 == 0) &&
                (reinterpret_cast<uintptr_t>(gamma.const_data_ptr()) % 8 == 0);
  if (vector)
    norm_tree<4><<<rows, threads / 4, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        rows,
        width,
        heads,
        x.stride(0),
        x.dim() == 3 ? x.stride(1) : 0,
        static_cast<float>(eps));
  else
    norm_tree<1><<<rows, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
        rows,
        width,
        heads,
        x.stride(0),
        x.dim() == 3 ? x.stride(1) : 0,
        static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::pair<at::Tensor, at::Tensor> qwen35_reference_residual_norm_cuda(const at::Tensor& x,
                                                                      const at::Tensor& addition,
                                                                      const at::Tensor& gamma,
                                                                      double eps) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == at::kBFloat16 &&
                  x.is_contiguous() && addition.device() == x.device() &&
                  addition.scalar_type() == x.scalar_type() && addition.sizes() == x.sizes() &&
                  addition.is_contiguous() && gamma.device() == x.device() &&
                  gamma.scalar_type() == x.scalar_type() && gamma.dim() == 1 &&
                  gamma.is_contiguous(),
              "Invalid Qwen3.5 residual norm operands");
  const auto rows = x.size(0), width = x.size(1);
  TORCH_CHECK(rows > 0 && rows <= INT32_MAX && width > 0 && width <= 8192 && gamma.numel() == width,
              "Invalid Qwen3.5 residual norm geometry");
  c10::cuda::CUDAGuard guard(x.device());
  auto updated = at::empty_like(x), normalized = at::empty_like(x);
  int threads = 1;
  while (threads < width && threads < 1024)
    threads *= 2;
  auto launch = [&]<int V>() {
    norm_tree<V, true><<<rows, threads / V, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(gamma.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(normalized.mutable_data_ptr()),
        rows,
        width,
        1,
        width,
        0,
        static_cast<float>(eps),
        reinterpret_cast<const __nv_bfloat16*>(addition.const_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(updated.mutable_data_ptr()));
  };
  const bool vector = width % 4 == 0 && reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(addition.const_data_ptr()) % 8 == 0 &&
                      reinterpret_cast<uintptr_t>(gamma.const_data_ptr()) % 8 == 0;
  if (vector)
    launch.template operator()<4>();
  else
    launch.template operator()<1>();
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {updated, normalized};
}

/// Keep convolution and recurrence arithmetic in the reference order,
/// including prefix checkpoint publication.
at::Tensor qwen35_reference_delta_cuda(const at::Tensor& qkv,
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
        qkv, conv, input, cached.history, cached.history_lengths, convolution, false);
  else
    tei_reference_qwen35_conv_bf16<<<(tokens * channels + 255) / 256, 256, 0, stream>>>(
        bf(qkv), bf(conv), cu, mut(convolution), tokens, channels, conv.size(-1), input.batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tei_reference_qwen35_gdn_prepare<<<tokens * key_heads, 128, 0, stream>>>(
      bf(convolution),
      bf(ab),
      a_log.const_data_ptr<float>(),
      dt_bias.const_data_ptr<float>(),
      qk.mutable_data_ptr<float>(),
      decay.mutable_data_ptr<float>(),
      tokens,
      key_heads,
      value_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (input.prefix)
    qwen35_cached_recurrent_cuda(convolution,
                                 qk,
                                 decay,
                                 input,
                                 recurrence,
                                 key_heads,
                                 value_heads,
                                 false,
                                 cached.initial_sources,
                                 cached.checkpoint_positions,
                                 cached.checkpoint_rows,
                                 cached.checkpoints);
  else if (qwen35_recurrent_supported(qkv.device(), key_heads, value_heads))
    qwen35_recurrent_cuda(convolution, qk, decay, input, recurrence, key_heads, value_heads, false);
  else {
    tei_reference_qwen35_gdn_recurrent<<<dim3(input.batch, value_heads, 16), 256, 0, stream>>>(
        bf(convolution),
        qk.const_data_ptr<float>(),
        decay.const_data_ptr<float>(),
        cu,
        mut(recurrence),
        key_heads,
        value_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  tei_reference_qwen35_gdn_norm<<<tokens * value_heads, 128, 0, stream>>>(
      bf(recurrence), bf(z), bf(norm), mut(output), epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (input.prefix)
    prefix_recurrent_commit(input, qkv, cache_layer, conv.size(-1), cached);
  return output.view({tokens, value_heads * 128});
}
}
