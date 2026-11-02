// SPDX-License-Identifier: MIT
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//
// Bounded raw CUDA foundation for native Qwen3.5 Gated DeltaNet. This file is
// additive to the existing oxide device bundle and is not wired into an
// executor, factory, admission policy, or model route.
//
// The four entrypoints below are copied byte-identically from
// libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/kernels/
// qwen35_gdn.cu (Candle source revision
// d9073968443369d57cdb378db2e0ff957c5ec027, retained MIT notice). Their
// arithmetic, FMA/rounding order, and BF16/F32 boundaries are unchanged.
//
// The packed fresh/cached convolution and cached recurrence are extracted from
// libs/infinity-v3/backends/libtorch/cpp/kernels/qwen35_kernels.cu at the
// c4c90b9 source revision (SPDX MIT). Only the launchable symbol spelling and
// a device-inline body extraction were added; the arithmetic and reduction
// trees are unchanged.
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the
// Software without restriction, including without
// limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice
// shall be included in all copies or substantial portions
// of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
// ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
// PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ── Unchanged Candle span: conv, prepare, fresh recurrence, gated norm ──────
// The following four kernel bodies are byte-identical to the vendored Candle
// source named above. Do not add intermediate BF16 casts: the prepare kernel
// reads native BF16 gate inputs but evaluates softplus/exp and decay in F32.
__device__ float gdn_sum(float x) {
  for (int d = 16; d; d >>= 1)
    x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}

extern "C" __global__ void qwen35_conv_bf16(const __nv_bfloat16* x,
                                            const __nv_bfloat16* weight,
                                            const uint32_t* cu,
                                            __nv_bfloat16* out,
                                            int tokens,
                                            int channels,
                                            int kernel,
                                            int sequences) {
  const uint64_t idx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= uint64_t(tokens) * channels)
    return;
  int t = idx / channels, c = idx % channels, lo = 0, hi = sequences;
  while (lo + 1 < hi) {
    int m = (lo + hi) / 2;
    if (cu[m] <= t)
      lo = m;
    else
      hi = m;
  }
  float sum = 0;
  for (int j = 0; j < kernel; ++j) {
    int src = t + j - kernel + 1;
    if (src >= int(cu[lo]))
      sum += __bfloat162float(x[uint64_t(src) * channels + c]) *
             __bfloat162float(weight[c * kernel + j]);
  }
  // Preserve the Transformers conv -> BF16 -> SiLU rounding contract.
  sum = __bfloat162float(__float2bfloat16_rn(sum));
  out[idx] = __float2bfloat16_rn(sum / (1.f + expf(-sum)));
}

extern "C" __global__ void qwen35_gdn_prepare(const __nv_bfloat16* qkv,
                                              const __nv_bfloat16* ab,
                                              const float* alog,
                                              const float* dt,
                                              float* qk,
                                              float* decay_beta,
                                              int tokens,
                                              int kh,
                                              int vh) {
  const int row = blockIdx.x, head = row % kh, t = row / kh, j = threadIdx.x;
  const int channels = (2 * kh + vh) * 128;
  float q = __bfloat162float(qkv[uint64_t(t) * channels + head * 128 + j]);
  float k = __bfloat162float(qkv[uint64_t(t) * channels + kh * 128 + head * 128 + j]);
  __shared__ float qs[4], ks[4];
  float qsum = gdn_sum(q * q), ksum = gdn_sum(k * k);
  if (j % 32 == 0) {
    qs[j / 32] = qsum;
    ks[j / 32] = ksum;
  }
  __syncthreads();
  qsum = qs[0] + qs[1] + qs[2] + qs[3];
  ksum = ks[0] + ks[1] + ks[2] + ks[3];
  qk[uint64_t(row) * 256 + j] = q * rsqrtf(qsum + 1e-6f) * rsqrtf(128.f);
  qk[uint64_t(row) * 256 + 128 + j] = k * rsqrtf(ksum + 1e-6f);
  if (head == 0 && j < vh) {
    float a = __bfloat162float(ab[uint64_t(t) * 2 * vh + j]) + dt[j];
    float softplus = a > 20.f ? a : log1pf(expf(a));
    float b = __bfloat162float(ab[uint64_t(t) * 2 * vh + vh + j]);
    decay_beta[uint64_t(t) * 2 * vh + j] = expf(-expf(alog[j]) * softplus);
    decay_beta[uint64_t(t) * 2 * vh + vh + j] =
        __bfloat162float(__float2bfloat16_rn(1.f / (1.f + expf(-b))));
  }
}

extern "C" __global__ void qwen35_gdn_recurrent(const __nv_bfloat16* qkv,
                                                const float* qk,
                                                const float* decay_beta,
                                                const uint32_t* cu,
                                                __nv_bfloat16* out,
                                                int kh,
                                                int vh) {
  const int seq = blockIdx.x, head = blockIdx.y, col = blockIdx.z * 8 + threadIdx.x / 32,
            lane = threadIdx.x % 32;
  const int qhead = head / (vh / kh), channels = (2 * kh + vh) * 128;
  // One warp owns one state column. Four FP32 entries per lane, no state in HBM.
  float state[4] = {0, 0, 0, 0};
  for (uint32_t t = cu[seq]; t < cu[seq + 1]; ++t) {
    float decay = decay_beta[uint64_t(t) * 2 * vh + head],
          beta = decay_beta[uint64_t(t) * 2 * vh + vh + head];
    float q[4], k[4], memory = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q[i] = qk[(uint64_t(t) * kh + qhead) * 256 + lane + i * 32];
      k[i] = qk[(uint64_t(t) * kh + qhead) * 256 + 128 + lane + i * 32];
      state[i] *= decay;
      memory += state[i] * k[i];
    }
    memory = gdn_sum(memory);
    float v = __bfloat162float(qkv[uint64_t(t) * channels + 2 * kh * 128 + head * 128 + col]);
    float delta = (v - memory) * beta, sum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      state[i] += k[i] * delta;
      sum += state[i] * q[i];
    }
    sum = gdn_sum(sum);
    if (lane == 0)
      out[(uint64_t(t) * vh + head) * 128 + col] = __float2bfloat16_rn(sum);
  }
}

extern "C" __global__ void qwen35_gdn_norm(const __nv_bfloat16* x,
                                           const __nv_bfloat16* z,
                                           const __nv_bfloat16* weight,
                                           __nv_bfloat16* out,
                                           float eps) {
  const uint64_t row = blockIdx.x;
  const int j = threadIdx.x;
  float v = __bfloat162float(x[row * 128 + j]);
  float sum = gdn_sum(v * v);
  __shared__ float sums[4];
  if (j % 32 == 0)
    sums[j / 32] = sum;
  __syncthreads();
  sum = (sums[0] + sums[1] + sums[2] + sums[3]) / 128.f;
  float normalized = __bfloat162float(__float2bfloat16_rn(v * rsqrtf(sum + eps)));
  normalized = __bfloat162float(__float2bfloat16_rn(normalized * __bfloat162float(weight[j])));
  float gate = __bfloat162float(z[row * 128 + j]);
  out[row * 128 + j] = __float2bfloat16_rn(normalized * (gate / (1.f + expf(-gate))));
}

// ── Extracted packed convolution and cached recurrence ─────────────────────
// Arithmetic, FMA/rounding order, and the 32-lane reduction reconstruction
// are preserved from kernels/qwen35_kernels.cu. The template bodies are
// device-inline so both explicit raw exports use exactly the same code.
namespace infinity_gdn_native {

/// Aligned four-value BF16 pack used by the native convolution path.
struct alignas(8) ConvValues {
  __nv_bfloat16 values[4];
};

// Cached suffixes read their immutable prefix tail; short prefixes need no padding.
template <bool Cached>
__device__ __forceinline__ void packed_conv_vector(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight,
                                                   const uint32_t* cumulative,
                                                   __nv_bfloat16* output,
                                                   int channels,
                                                   int kernel,
                                                   int sequences,
                                                   int64_t row_stride,
                                                   const int64_t* sources,
                                                   const __nv_bfloat16* history,
                                                   const int32_t* history_cu) {
  const int token = blockIdx.x, column = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (column >= channels)
    return;
  int lo = 0, hi = sequences;
  while (lo + 1 < hi) {
    const int middle = (lo + hi) / 2;
    if (cumulative[middle] <= token)
      lo = middle;
    else
      hi = middle;
  }
  int history_end = 0, previous = 0;
  if constexpr (Cached) {
    history_end = history_cu[sources[lo] + 1];
    previous = history_end - history_cu[sources[lo]];
  }
  float sums[4] = {};
  for (int j = 0; j < kernel; ++j) {
    const int source = token + j - kernel + 1;
    const __nv_bfloat16* row = nullptr;
    if (source >= int(cumulative[lo]))
      row = x + int64_t(source) * row_stride;
    else if constexpr (Cached) {
      if (source >= int(cumulative[lo]) - previous)
        row = history + int64_t(history_end + source - int(cumulative[lo])) * channels;
    }
    if (row) {
      const auto values = *reinterpret_cast<const ConvValues*>(row + column);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        sums[i] += __bfloat162float(values.values[i]) *
                   __bfloat162float(weight[(column + i) * kernel + j]);
    }
  }
  ConvValues result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // Preserve convolution -> BF16 -> SiLU rounding and accumulation order.
    const float rounded = __bfloat162float(__float2bfloat16_rn(sums[i]));
    result.values[i] = __float2bfloat16_rn(rounded / (1.f + expf(-rounded)));
  }
  *reinterpret_cast<ConvValues*>(output + int64_t(token) * channels + column) = result;
}

// Adapted from Candle qwen35_gdn_recurrent above; snapshot the complete prefix state.
template <int Lanes>
__device__ __forceinline__ float gdn_sum(float (&partial)[32 / Lanes]) {
  // Reconstruct the original 32-lane reduction before subgroup shuffles.
#pragma unroll
  for (int d = 16 / Lanes; d; d >>= 1) {
#pragma unroll
    for (int j = 0; j < d; ++j)
      partial[j] += partial[j + d];
  }
  float sum = partial[0];
  for (int d = Lanes / 2; d; d >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, d, Lanes);
  return __shfl_sync(0xffffffff, sum, 0, Lanes);
}

template <int Lanes>
__device__ __forceinline__ void qwen35_gdn_cached(const __nv_bfloat16* qkv,
                                                  const float* qk,
                                                  const float* decay_beta,
                                                  const uint32_t* cu,
                                                  __nv_bfloat16* out,
                                                  int kh,
                                                  int vh,
                                                  const float* initial,
                                                  float* final_state,
                                                  const int64_t* sources) {
  constexpr int columns = 512 / Lanes, entries = 128 / Lanes, partials = 32 / Lanes;
  const int seq = blockIdx.x, head = blockIdx.y, col = blockIdx.z * columns + threadIdx.x / Lanes,
            lane = threadIdx.x % Lanes;
  const int qhead = head / (vh / kh), channels = (2 * kh + vh) * 128;
  // Split accumulators preserve the original 32-lane tree.
  float state[entries] = {};
  const auto source = initial ? sources[seq] : seq;
  if (initial)
    for (int i = 0; i < entries; ++i)
      state[i] = initial[((source * vh + head) * 128 + lane + i * Lanes) * 128 + col];
  constexpr int tile = 32;
  __shared__ __align__(16) float queries_keys[tile][256];
  __shared__ float gates[tile][2];
  __shared__ __align__(8) __nv_bfloat16 values[tile][columns], outputs[tile][columns];
  const uint32_t begin = cu[seq], end = cu[seq + 1];
  for (uint32_t start = begin; start < end; start += tile) {
    const int count = min(uint32_t(tile), end - start);
    __syncthreads();
    // Cooperatively stage each head once for all state columns.
    for (int i = threadIdx.x; i < count * 64; i += blockDim.x)
      reinterpret_cast<float4*>(queries_keys[i / 64])[i % 64] = reinterpret_cast<const float4*>(
          qk + (uint64_t(start + i / 64) * kh + qhead) * 256)[i % 64];
    for (int i = threadIdx.x; i < count * 2; i += blockDim.x)
      gates[i / 2][i % 2] = decay_beta[uint64_t(start + i / 2) * 2 * vh + head + (i % 2) * vh];
    for (int i = threadIdx.x; i < count * (columns / 4); i += blockDim.x) {
      const int row = i / (columns / 4), pack = i % (columns / 4);
      reinterpret_cast<ConvValues*>(values[row])[pack] = reinterpret_cast<const ConvValues*>(
          qkv + uint64_t(start + row) * channels + 2 * kh * 128 + head * 128 +
          blockIdx.z * columns)[pack];
    }
    __syncthreads();
#pragma unroll 2
    for (int row = 0; row < count; ++row) {
      float decay = gates[row][0], beta = gates[row][1];
      float q[entries], k[entries], memory[partials] = {};
#pragma unroll
      for (int i = 0; i < entries; ++i) {
        q[i] = queries_keys[row][lane + i * Lanes];
        k[i] = queries_keys[row][128 + lane + i * Lanes];
        state[i] *= decay;
        memory[i % partials] += state[i] * k[i];
      }
      const float remembered = gdn_sum<Lanes>(memory);
      float delta = (__bfloat162float(values[row][threadIdx.x / Lanes]) - remembered) * beta,
            sum[partials] = {};
#pragma unroll
      for (int i = 0; i < entries; ++i) {
        state[i] += k[i] * delta;
        sum[i % partials] += state[i] * q[i];
      }
      const float value = gdn_sum<Lanes>(sum);
      if (lane == 0)
        outputs[row][threadIdx.x / Lanes] = __float2bfloat16_rn(value);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < count * (columns / 4); i += blockDim.x) {
      const int row = i / (columns / 4), pack = i % (columns / 4);
      reinterpret_cast<ConvValues*>(out + (uint64_t(start + row) * vh + head) * 128 +
                                    blockIdx.z * columns)[pack] =
          reinterpret_cast<const ConvValues*>(outputs[row])[pack];
    }
  }
  if (final_state)
    for (int i = 0; i < entries; ++i)
      final_state[((int64_t(seq) * vh + head) * 128 + lane + i * Lanes) * 128 + col] = state[i];
}

} // namespace infinity_gdn_native

// ── Additive raw exports ────────────────────────────────────────────────────
// Fresh convolution with the original packed row stride and logical sequence
// boundaries. Physical rows are never padded.
extern "C" __global__ void infinity_qwen35_packed_conv_fresh(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const uint32_t* cumulative,
    __nv_bfloat16* output,
    int channels,
    int kernel,
    int sequences,
    int64_t row_stride) {
  infinity_gdn_native::packed_conv_vector<false>(x, weight, cumulative, output, channels, kernel,
                                                 sequences, row_stride, nullptr, nullptr, nullptr);
}

// Cached convolution with an immutable prefix tail and per-sequence source map.
extern "C" __global__ void infinity_qwen35_packed_conv_cached(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const uint32_t* cumulative,
    __nv_bfloat16* output,
    int channels,
    int kernel,
    int sequences,
    int64_t row_stride,
    const int64_t* sources,
    const __nv_bfloat16* history,
    const int32_t* history_cu) {
  infinity_gdn_native::packed_conv_vector<true>(x, weight, cumulative, output, channels, kernel,
                                                sequences, row_stride, sources, history,
                                                history_cu);
}

// The original launch policy chooses eight lanes at batch*value_heads >= 256
// and sixteen lanes below that boundary. Both reductions reconstruct the same
// 32-lane tree; the host planner records the choice before launch.
extern "C" __global__ void infinity_qwen35_gdn_cached_lanes8(
    const __nv_bfloat16* qkv,
    const float* qk,
    const float* decay_beta,
    const uint32_t* cu,
    __nv_bfloat16* out,
    int kh,
    int vh,
    const float* initial,
    float* final_state,
    const int64_t* sources) {
  infinity_gdn_native::qwen35_gdn_cached<8>(qkv, qk, decay_beta, cu, out, kh, vh, initial,
                                            final_state, sources);
}

extern "C" __global__ void infinity_qwen35_gdn_cached_lanes16(
    const __nv_bfloat16* qkv,
    const float* qk,
    const float* decay_beta,
    const uint32_t* cu,
    __nv_bfloat16* out,
    int kh,
    int vh,
    const float* initial,
    float* final_state,
    const int64_t* sources) {
  infinity_gdn_native::qwen35_gdn_cached<16>(qkv, qk, decay_beta, cu, out, kh, vh, initial,
                                             final_state, sources);
}
