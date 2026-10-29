/*
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This adapter retains and invokes the fork's BSD-3-Clause Candle kernel.
// Private kernel names prevent preemption of Candle's own CUDA registration.
/// \file
/// CUDA decoder RMS normalization with reference residual and reduction semantics.
/// Dispatches the registered row-width and dtype specializations.

#include "kernels/decoder_norm.h"
#include "kernels/fp8_quant_pair.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <type_traits>
#include <cstdint>
#define layer_norm tei_decoder_exact_norm
#include "../../../candle/extensions/candle-layer-norm/kernels/ln_fwd_kernels.cuh"
#undef layer_norm

namespace infinity {
namespace {
// Specialize the retained Candle 1x4-warp contract: the same 32 values per
// thread, XOR warp reductions and Chan merges, with fixed group sizes.
// One block owns each row; other widths and unaligned views use the adapter.
struct alignas(16) ExactBf16Vector {
  __nv_bfloat16 v[8];
};

__device__ float2 combine_exact_stats(float2 a, float2 b, float count, float inverse) {
  float delta = a.x - b.x;
  float m2 = a.y + b.y + delta * delta * count * count * inverse;
  float mean = (count * a.x + count * b.x) * inverse;
  return {mean, m2};
}

template <bool Quantize>
__global__ void exact_bf16_4096_kernel(const __nv_bfloat16* x,
                                       const __nv_bfloat16* residual,
                                       const __nv_bfloat16* weight,
                                       __nv_bfloat16* out,
                                       __nv_bfloat16* summed,
                                       float epsilon,
                                       float* scales) {
  __shared__ float2 warp_stats[4];
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float values[32];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int group = threadIdx.x + j * 128;
    auto input = reinterpret_cast<const ExactBf16Vector*>(x)[row * 512 + group];
    ExactBf16Vector add{}, sum;
    if (residual)
      add = reinterpret_cast<const ExactBf16Vector*>(residual)[row * 512 + group];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float value = float(input.v[i]);
      if (residual)
        value += float(add.v[i]);
      values[j * 8 + i] = value;
      sum.v[i] = __nv_bfloat16(value);
    }
    if (residual)
      reinterpret_cast<ExactBf16Vector*>(summed)[row * 512 + group] = sum;
  }
  float mean = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    mean += values[i];
#pragma unroll
  for (int i = 1; i < 32; i *= 2)
    mean += __shfl_xor_sync(0xffffffff, mean, i);
  mean *= 1.f / 1024;
  float m2 = 0;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    float delta = values[i] - mean;
    m2 += delta * delta;
  }
#pragma unroll
  for (int i = 1; i < 32; i *= 2)
    m2 += __shfl_xor_sync(0xffffffff, m2, i);
  if (!lane)
    warp_stats[warp] = {mean, m2};
  __syncthreads();
  auto even = combine_exact_stats(warp_stats[0], warp_stats[2], 1024, 1.f / 2048);
  auto odd = combine_exact_stats(warp_stats[1], warp_stats[3], 1024, 1.f / 2048);
  auto stats = combine_exact_stats(even, odd, 2048, 1.f / 4096);
  float inverse = rsqrtf(stats.y * (1.f / 4096) + epsilon + stats.x * stats.x);
  float amax = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int group = threadIdx.x + j * 128;
    auto gamma = reinterpret_cast<const ExactBf16Vector*>(weight)[group];
    ExactBf16Vector result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      result.v[i] = __nv_bfloat16(float(gamma.v[i]) * (inverse * (values[j * 8 + i] - 0.f)) + 0.f);
      if constexpr (Quantize) {
        values[j * 8 + i] = float(result.v[i]);
        amax = fmaxf(amax, fabsf(values[j * 8 + i]));
      }
    }
    if constexpr (!Quantize)
      reinterpret_cast<ExactBf16Vector*>(out)[row * 512 + group] = result;
  }
  if constexpr (Quantize) {
    __shared__ float maxima[4];
    for (int d = 16; d; d /= 2)
      amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, d));
    if (!lane)
      maxima[warp] = amax;
    __syncthreads();
    float scale =
        __fmul_rn(fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(fmaxf(maxima[2], maxima[3]), 1e-12f)),
                  1.f / 448.f);
    if (!threadIdx.x)
      scales[row] = scale;

    struct alignas(8) Fp8Vector {
      __nv_fp8x2_e4m3 value[4];
    };

#pragma unroll
    for (int j = 0; j < 4; ++j) {
      Fp8Vector result;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        result.value[i] =
            fp8_quantize_pair(values[j * 8 + 2 * i], values[j * 8 + 2 * i + 1], scale);
      reinterpret_cast<Fp8Vector*>(out)[row * 512 + threadIdx.x + j * 128] = result;
    }
  }
}

template <int Width, int WarpsM, int WarpsN, class T = nv_bfloat16, bool Even = true>
std::pair<at::Tensor, at::Tensor>
exact_norm(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& w, double epsilon) {
  const c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x), sum = residual.defined() ? at::empty_like(x) : x;
  if (x.size(0) == 0)
    return {out, sum};
  tei_decoder_exact_norm::FwdParams p;
  p.rows = x.size(0);
  p.cols = Even ? Width : x.size(1);
  p.x0 = x.data_ptr();
  p.x1 = nullptr;
  p.residual = residual.defined() ? residual.data_ptr() : nullptr;
  p.x = residual.defined() ? sum.data_ptr() : nullptr;
  p.z = out.data_ptr();
  p.dmask = nullptr;
  p.dmask1 = nullptr;
  p.x0_subset = nullptr;
  p.z_subset = nullptr;
  p.gamma = w.data_ptr();
  p.epsilon = epsilon;
  p.inverse_cols = Even ? 1.f / Width : 1.f / x.size(1);
  p.dropout_keep_p = 1.f;
  p.dropout_scale = 1.f;
  p.rowscale_const = 1.f;
  p.round_residual = false;
  p.is_rms_norm = true;
  using Traits =
      tei_decoder_exact_norm::Kernel_traits<T, T, T, T, float, int, Width, 1, WarpsM, WarpsN, 16>;
  auto kernel = tei_decoder_exact_norm::ln_fwd_kernel<Traits, false, false, false, Even>;
  // Each registered width uses its original Candle warp geometry. Cache
  // occupancy separately per specialization and device, before graph capture.
  static thread_local std::vector<int> cache;
  const int device = x.get_device();
  if (cache.size() <= static_cast<size_t>(device))
    cache.resize(device + 1, -1);
  auto& blocks = cache[device];
  if (blocks < 0)
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, Traits::THREADS_PER_CTA, Traits::SMEM_BYTES_FWD));
  p.ctas_per_col = at::cuda::getCurrentDeviceProperties()->multiProcessorCount * blocks;
  kernel<<<p.ctas_per_col,
           Traits::THREADS_PER_CTA,
           Traits::SMEM_BYTES_FWD,
           at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, sum};
}
}

std::pair<at::Tensor, at::Tensor> decoder_exact_rms_norm(const at::Tensor& x,
                                                         const at::Tensor& residual,
                                                         const at::Tensor& w,
                                                         double epsilon) {
  return decoder_exact_rms_norm(x, residual, w, epsilon, nullptr);
}

/// Select the registered dtype/width reduction geometry after validating contiguous row layout. The
/// vectorized BF16-4096 route retains the same statistics and residual-rounding boundaries.
std::pair<at::Tensor, at::Tensor> decoder_exact_rms_norm(const at::Tensor& x,
                                                         const at::Tensor& residual,
                                                         const at::Tensor& w,
                                                         double epsilon,
                                                         at::Tensor* scales) {
  if (scales)
    *scales = at::Tensor();
  // The following width list matches explicit template registrations below, including masked width
  // 896. Adding a width requires a matching reduction/warp layout; accepting it here alone would
  // select the wrong registration.
  TORCH_CHECK(
      x.is_cuda() && x.dim() == 2 && x.is_contiguous() &&
          ((x.scalar_type() == at::kBFloat16 &&
            (x.size(1) == 896 || x.size(1) == 1024 || x.size(1) == 2048 || x.size(1) == 3072 ||
             x.size(1) == 4096)) ||
           (x.scalar_type() == at::kHalf &&
            (x.size(1) == 896 || x.size(1) == 1024 || x.size(1) == 2048))),
      "Exact decoder RMS expects CUDA BF16 width 896, 1024, 2048, 3072 or 4096, or FP16 width 896, 1024 or 2048");
  TORCH_CHECK(w.device() == x.device() && w.scalar_type() == x.scalar_type() && w.dim() == 1 &&
                  w.numel() == x.size(1) && w.is_contiguous(),
              "Invalid exact decoder RMS weight");
  TORCH_CHECK(!residual.defined() ||
                  (residual.device() == x.device() && residual.scalar_type() == x.scalar_type() &&
                   residual.sizes() == x.sizes() && residual.is_contiguous()),
              "Invalid exact decoder RMS residual");
  // The retained registration also uses vector loads; contiguous offset views
  // must be aligned before either implementation accesses their storage.
  auto aligned = [](const at::Tensor& t) {
    return !t.defined() || reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0;
  };
  if (!aligned(x) || !aligned(w) || !aligned(residual)) {
    auto result = decoder_exact_rms_norm(
        x.clone(), residual.defined() ? residual.clone() : residual, w.clone(), epsilon, scales);
    if (!residual.defined())
      result.second = x;
    return result;
  }
  // Width and 16-byte alignment match the vectorized kernel's loads. The row threshold selects that
  // implementation; smaller batches retain the registered reduction path
  // instead of rejecting the input.
  if (x.scalar_type() == at::kBFloat16 && x.size(1) == 4096 && x.size(0) >= 128 &&
      reinterpret_cast<uintptr_t>(x.const_data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(w.const_data_ptr()) % 16 == 0 &&
      (!residual.defined() || reinterpret_cast<uintptr_t>(residual.const_data_ptr()) % 16 == 0)) {
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x, x.options().dtype(scales ? at::kFloat8_e4m3fn : at::kBFloat16));
    auto sum = residual.defined() ? at::empty_like(x) : x;
    if (scales)
      *scales = at::empty({x.size(0), 1}, x.options().dtype(at::kFloat));
    auto ptr = [](const at::Tensor& t) {
      return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
    };
    auto launch = [&](auto quantize) {
      exact_bf16_4096_kernel<decltype(quantize)::value>
          <<<x.size(0), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
              ptr(x),
              residual.defined() ? ptr(residual) : nullptr,
              ptr(w),
              reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),
              reinterpret_cast<__nv_bfloat16*>(sum.mutable_data_ptr()),
              epsilon,
              scales ? scales->data_ptr<float>() : nullptr);
    };
    if (scales)
      launch(std::true_type{});
    else
      launch(std::false_type{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, sum};
  }
  // Retain Candle's registered 4x1 warp geometry and FP16 rounding.
  if (x.scalar_type() == at::kHalf && x.size(1) == 896)
    return exact_norm<1024, 4, 1, __half, false>(x, residual, w, epsilon);
  if (x.scalar_type() == at::kHalf && x.size(1) == 2048)
    return exact_norm<2048, 4, 1, __half>(x, residual, w, epsilon);
  if (x.scalar_type() == at::kHalf)
    return exact_norm<1024, 4, 1, __half>(x, residual, w, epsilon);
  // Candle masks actual BF16 width896 within its original1024 registration.
  if (x.size(1) == 896)
    return exact_norm<1024, 4, 1, nv_bfloat16, false>(x, residual, w, epsilon);
  if (x.size(1) == 1024)
    return exact_norm<1024, 4, 1>(x, residual, w, epsilon);
  if (x.size(1) == 2048)
    return exact_norm<2048, 4, 1>(x, residual, w, epsilon);
  if (x.size(1) == 3072)
    return exact_norm<3072, 1, 4>(x, residual, w, epsilon);
  return exact_norm<4096, 1, 4>(x, residual, w, epsilon);
}
}
