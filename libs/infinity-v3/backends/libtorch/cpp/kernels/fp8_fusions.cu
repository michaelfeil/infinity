// SPDX-License-Identifier: Apache-2.0
// RMS arithmetic adapted from the Apache-2.0 fast_kernels.cu implementation.
// Preserve its reduction order and BF16 rounding before row quantization.
#include "kernels/fp8_fusions.h"
#include "kernels/dependent_launch.cuh"
#include "kernels/encoder_activation_math.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include "kernels/fp8_quant_pair.cuh"
#include <climits>
#include <cstdint>
#include <cmath>
#include <type_traits>

namespace infinity {
namespace {
__device__ float sum256(float value) {
  __shared__ float warps[8];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int d = 16; d; d /= 2)
    value += __shfl_down_sync(0xffffffff, value, d);
  if (!lane)
    warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? warps[lane] : 0.f;
  if (warp == 0)
    for (int d = 16; d; d /= 2)
      value += __shfl_down_sync(0xffffffff, value, d);
  if (!threadIdx.x)
    warps[0] = value;
  __syncthreads();
  float total = warps[0];
  __syncthreads();
  return total;
}

__device__ float row_scale(float value) {
  __shared__ float warps[16];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int d = 16; d; d /= 2)
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
  if (!lane)
    warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? warps[lane] : 0.f;
  if (warp == 0)
    for (int d = 16; d; d /= 2)
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, d));
  if (!threadIdx.x)
    warps[0] = __fmul_rn(fmaxf(value, 1e-12f), 1.f / 448.f);
  __syncthreads();
  return warps[0];
}

template <Fp8GatedActivation Kind, class T>
__device__ __forceinline__ T gated_rounded(T gate, T up) {
  T activated;
  if constexpr (Kind == Fp8GatedActivation::RoundedSilu) {
    T exponential = hexp(T(-gate));
    T denominator = T(T(1) + exponential);
    activated = T(gate / denominator);
  } else if constexpr (Kind == Fp8GatedActivation::PreciseGelu) {
    // Match gemma_precise_activation<true>(c10::BFloat16) exactly. c10::BFloat16 operators use
    // float opmath plus cvt.rn.bf16.f32, while the native __hmul/__hadd helpers may select
    // mul.bf16/add.bf16, which can round or flush differently. Keep every stage as a separate
    // c10::BFloat16 value and bitcast the final BF16 result back to T.
    static_assert(std::is_same_v<T, __nv_bfloat16>, "PreciseGelu requires BF16 rows");
    const c10::BFloat16 x(static_cast<float>(gate));
    const c10::BFloat16 square = x * x;
    const c10::BFloat16 cube = square * x;
    const c10::BFloat16 alpha = x + c10::BFloat16(.044715) * cube;
    const c10::BFloat16 argument = c10::BFloat16(M_2_SQRTPI * M_SQRT1_2) * alpha;
    const c10::BFloat16 tangent = c10::BFloat16(tanhf(static_cast<float>(argument)));
    const c10::BFloat16 gated = c10::BFloat16(.5) * x * (c10::BFloat16(1.) + tangent);
    const c10::BFloat16 result = gated * c10::BFloat16(static_cast<float>(up));
    return static_cast<T>(result);
  } else {
    // Match fast_kernels gated_value activation 4: each GELU intermediate is a separate
    // model-dtype operation, including the tanh input/value.
    T x = gate, square = T(x * x), cube = T(square * x);
    T alpha = T(x + T(.044715) * cube);
    T argument = T(T(.7978845608028654) * alpha);
    T tangent = T(tanhf(float(argument)));
    activated = T(T(T(.5) * x) * T(T(1) + tangent));
  }
  return T(float(activated) * float(up));
}

template <Fp8GatedActivation Kind, class T, int Columns>
__global__ void
gated_quant_kernel(const T* input, __nv_fp8_e4m3* out, float* scales, int width, bool gate_first) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  float values[Columns], amax = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    float value = 0;
    if (col < width) {
      const T first = input[row * 2 * width + col], second = input[row * 2 * width + width + col];
      value = float(gated_rounded<Kind>(gate_first ? first : second, gate_first ? second : first));
    }
    values[j] = value;
    amax = fmaxf(amax, fabsf(value));
  }
  float scale = row_scale(amax);

  if (!threadIdx.x)
    scales[row] = scale;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    if (col < width)
      out[row * width + col] = __nv_fp8_e4m3(__fdiv_rn(values[j], scale));
  }
}

/// Aligned scalar pack used by fused normalization and FP8 conversion kernels.
template <class T, int N> struct alignas(sizeof(T) * N) Packed {
  T value[N];
};

template <class T> struct GatedPair;

template <> struct GatedPair<__half> {
  using Type = __half2;

  __device__ static Type model(float value) {
    return __float2half2_rn(value);
  }

  __device__ static Type tanh(Type value) {
    return __floats2half2_rn(tanhf(__low2float(value)), tanhf(__high2float(value)));
  }
};

template <> struct GatedPair<__nv_bfloat16> {
  using Type = __nv_bfloat162;

  __device__ static Type model(float value) {
    return __float2bfloat162_rn(value);
  }

  __device__ static Type tanh(Type value) {
    return __floats2bfloat162_rn(tanhf(__low2float(value)), tanhf(__high2float(value)));
  }
};

/// Match the native aligned-pair GELU and BF16 gate-first SwiGLU staging used by fast_kernels.
/// These explicit pair helpers are selected only for the layouts where the original wrapper does.
template <Fp8GatedActivation Kind, class T>
__device__ __forceinline__ typename GatedPair<T>::Type
gated_pair_rounded(typename GatedPair<T>::Type gate, typename GatedPair<T>::Type up) {
  using Pair = typename GatedPair<T>::Type;
  static_assert(Kind != Fp8GatedActivation::PreciseGelu,
                "PreciseGelu requires the scalar model-dtype staging");
  if constexpr (Kind == Fp8GatedActivation::RoundedGelu) {
    const auto square = __hmul2(gate, gate);
    const auto cube = __hmul2(square, gate);
    const auto alpha = __hfma2(GatedPair<T>::model(.044715f), cube, gate);
    const auto tangent =
        GatedPair<T>::tanh(__hmul2(GatedPair<T>::model(.7978845608028654f), alpha));
    const auto activated = __hmul2(__hmul2(GatedPair<T>::model(.5f), gate),
                                   __hadd2(GatedPair<T>::model(1.f), tangent));
    return __hmul2(activated, up);
  } else {
    const auto exponential = h2exp(__hneg2(gate));
    const auto denominator = __hadd2(GatedPair<T>::model(1.f), exponential);
    return __hmul2(__h2div(gate, denominator), up);
  }
}

template <Fp8GatedActivation Kind, bool PairMath, class T>
__device__ __forceinline__ Packed<T, 8> gated_pack(const Packed<T, 8>& gate,
                                                   const Packed<T, 8>& up) {
  Packed<T, 8> rounded;
  if constexpr (PairMath) {
    using Pair = typename GatedPair<T>::Type;
    const auto* gate_pairs = reinterpret_cast<const Pair*>(gate.value);
    const auto* up_pairs = reinterpret_cast<const Pair*>(up.value);
    auto* rounded_pairs = reinterpret_cast<Pair*>(rounded.value);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
      rounded_pairs[pair] = gated_pair_rounded<Kind, T>(gate_pairs[pair], up_pairs[pair]);
  } else {
#pragma unroll
    for (int lane = 0; lane < 8; ++lane)
      rounded.value[lane] = gated_rounded<Kind>(gate.value[lane], up.value[lane]);
  }
  return rounded;
}

__device__ __forceinline__ float scale_output_value(float value, float row, float column) {
  return __fmul_rn(__fmul_rn(value, row), column);
}

template <typename T> __device__ __forceinline__ T convert_scaled_output(float value) {
  if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(value);
  else
    return __float2bfloat16_rn(value);
}

template <class T> __device__ __forceinline__ T activation_rounded(T value, int kind) {
  switch (kind) {
  case static_cast<int>(Fp8Activation::Relu):
    return activation_math::rounded_relu(value);
  case static_cast<int>(Fp8Activation::RoundedSilu):
    return activation_math::rounded_silu(value);
  default:
    return activation_math::rounded_gelu(value);
  }
}

template <class T> __device__ __forceinline__ T activation_bias(T value, T bias) {
  return T(float(value) + float(bias));
}

template <class T, bool Scaled>
__device__ __forceinline__ void load_gated_pair(const T* input,
                                                const float* scaled_input,
                                                const float* row_scales,
                                                const float* column_scales,
                                                int64_t row,
                                                int stride,
                                                int col,
                                                bool gate_first,
                                                Packed<T, 8>& gate,
                                                Packed<T, 8>& up) {
  const int gate_col = gate_first ? col : stride + col;
  const int up_col = gate_first ? stride + col : col;
  if constexpr (Scaled) {
    const float row_scale_value = row_scales[row];
    const auto gate_raw =
        reinterpret_cast<const Packed<float, 8>*>(scaled_input)[row * 2 * stride + gate_col];
    const auto up_raw =
        reinterpret_cast<const Packed<float, 8>*>(scaled_input)[row * 2 * stride + up_col];
    const auto gate_column = reinterpret_cast<const Packed<float, 8>*>(column_scales)[gate_col];
    const auto up_column = reinterpret_cast<const Packed<float, 8>*>(column_scales)[up_col];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      gate.value[lane] = convert_scaled_output<T>(
          scale_output_value(gate_raw.value[lane], row_scale_value, gate_column.value[lane]));
      up.value[lane] = convert_scaled_output<T>(
          scale_output_value(up_raw.value[lane], row_scale_value, up_column.value[lane]));
    }
  } else {
    gate = reinterpret_cast<const Packed<T, 8>*>(input)[row * 2 * stride + gate_col];
    up = reinterpret_cast<const Packed<T, 8>*>(input)[row * 2 * stride + up_col];
  }
}

template <class T,
          int Groups,
          bool Scaled = false,
          Fp8GatedActivation Kind = Fp8GatedActivation::RoundedSilu,
          bool PairMath = false>
__global__ __launch_bounds__(512) void gated_quant_vector_kernel(const T* input,
                                                                 const float* scaled_input,
                                                                 const float* row_scales,
                                                                 const float* column_scales,
                                                                 __nv_fp8_e4m3* out,
                                                                 float* scales,
                                                                 int width,
                                                                 bool gate_first) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  // Scaled rows fit the common 48 KiB block limit (at most five groups). Ordinary gated rows keep
  // their original register storage; wider scaled rows use the composed fallback.
  __shared__ Packed<T, 8> staged[Scaled ? Groups * 512 : 1];
  float values[Scaled ? 1 : Groups][8];
  float amax = 0;
  int stride = width / 8;
#pragma unroll
  for (int j = 0; j < Groups; ++j) {
    int col = threadIdx.x + j * blockDim.x;
    if (col >= stride)
      continue;
    Packed<T, 8> gate, up;
    load_gated_pair<T, Scaled>(
        input, scaled_input, row_scales, column_scales, row, stride, col, gate_first, gate, up);
    const auto rounded = gated_pack<Kind, PairMath>(gate, up);
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      if constexpr (!Scaled)
        values[j][lane] = float(rounded.value[lane]);
      const float magnitude = fabsf(float(rounded.value[lane]));
      amax = fmaxf(amax, magnitude);
    }
    if constexpr (Scaled)
      staged[col] = rounded;
  }
  // row_scale also synchronizes the completed shared row before reuse below.
  float scale = row_scale(amax);
  const float reciprocal = __frcp_rn(scale);
  // A non-finite maximum needs the original IEEE division, including finite / infinity.
  const bool fast = isfinite(scale);

  if (!threadIdx.x)
    scales[row] = scale;
  if (fast) {
#pragma unroll
    for (int j = 0; j < Groups; ++j) {
      int col = threadIdx.x + j * blockDim.x;
      if (col < stride) {
        Packed<__nv_fp8x2_e4m3, 4> result;
        Packed<T, 8> rounded;
        if constexpr (Scaled)
          rounded = staged[col];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          if constexpr (Scaled)
            result.value[pair] = fp8_quantize_pair_recip(float(rounded.value[2 * pair]),
                                                         float(rounded.value[2 * pair + 1]),
                                                         scale,
                                                         reciprocal);
          else
            result.value[pair] = fp8_quantize_pair_recip(
                values[j][2 * pair], values[j][2 * pair + 1], scale, reciprocal);
        }
        reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
      }
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < Groups; ++j) {
    int col = threadIdx.x + j * blockDim.x;
    if (col < stride) {
      Packed<__nv_fp8x2_e4m3, 4> result;
      Packed<T, 8> rounded;
      if constexpr (Scaled)
        rounded = staged[col];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        if constexpr (Scaled)
          result.value[pair] = fp8_quantize_pair(
              float(rounded.value[2 * pair]), float(rounded.value[2 * pair + 1]), scale);
        else
          result.value[pair] =
              fp8_quantize_pair(values[j][2 * pair], values[j][2 * pair + 1], scale);
      }
      reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
    }
  }
}

template <class T, int Groups, bool Scaled>
__global__ __launch_bounds__(512) void activation_quant_vector_kernel(const T* input,
                                                                      const float* scaled_input,
                                                                      const float* row_scales,
                                                                      const float* column_scales,
                                                                      const T* bias,
                                                                      __nv_fp8_e4m3* out,
                                                                      float* scales,
                                                                      int width,
                                                                      int kind) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  // Aligned rows use at most five groups (about 40 KiB shared); other rows use the scalar kernel.
  __shared__ Packed<T, 8> staged[Groups * 512];
  float amax = 0;
  int stride = width / 8;
#pragma unroll
  for (int j = 0; j < Groups; ++j) {
    int col = threadIdx.x + j * blockDim.x;
    if (col >= stride)
      continue;
    Packed<T, 8> rounded;
    if constexpr (Scaled) {
      const auto raw = reinterpret_cast<const Packed<float, 8>*>(scaled_input)[row * stride + col];
      const auto columns = reinterpret_cast<const Packed<float, 8>*>(column_scales)[col];
      const float row_scale_value = row_scales[row];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        T value = convert_scaled_output<T>(
            scale_output_value(raw.value[lane], row_scale_value, columns.value[lane]));
        if (bias)
          value = activation_bias(value, bias[col * 8 + lane]);
        rounded.value[lane] = activation_rounded(value, kind);
      }
    } else {
      const auto values = reinterpret_cast<const Packed<T, 8>*>(input)[row * stride + col];
#pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        T value = values.value[lane];
        if (bias)
          value = activation_bias(value, bias[col * 8 + lane]);
        rounded.value[lane] = activation_rounded(value, kind);
      }
    }
#pragma unroll
    for (int lane = 0; lane < 8; ++lane)
      amax = fmaxf(amax, fabsf(float(rounded.value[lane])));
    staged[col] = rounded;
  }
  // row_scale also synchronizes the completed shared row before reuse below.
  float scale = row_scale(amax);
  const float reciprocal = __frcp_rn(scale);
  // A non-finite maximum needs the original IEEE division, including finite / infinity.
  const bool fast = isfinite(scale);

  if (!threadIdx.x)
    scales[row] = scale;
  if (fast) {
#pragma unroll
    for (int j = 0; j < Groups; ++j) {
      int col = threadIdx.x + j * blockDim.x;
      if (col < stride) {
        Packed<__nv_fp8x2_e4m3, 4> result;
        const auto rounded = staged[col];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair)
          result.value[pair] = fp8_quantize_pair_recip(float(rounded.value[2 * pair]),
                                                       float(rounded.value[2 * pair + 1]),
                                                       scale,
                                                       reciprocal);
        reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
      }
    }
    return;
  }
#pragma unroll
  for (int j = 0; j < Groups; ++j) {
    int col = threadIdx.x + j * blockDim.x;
    if (col < stride) {
      Packed<__nv_fp8x2_e4m3, 4> result;
      const auto rounded = staged[col];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair)
        result.value[pair] = fp8_quantize_pair(
            float(rounded.value[2 * pair]), float(rounded.value[2 * pair + 1]), scale);
      reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
    }
  }
}

template <class T, int Columns>
__global__ void activation_quant_kernel(
    const T* input, const T* bias, __nv_fp8_e4m3* out, float* scales, int width, int kind) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  float amax = 0;
#pragma unroll 1
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    float value = 0;
    if (col < width) {
      T input_value = input[row * width + col];
      if (bias)
        input_value = activation_bias(input_value, bias[col]);
      value = float(activation_rounded(input_value, kind));
    }
    amax = fmaxf(amax, fabsf(value));
  }
  float scale = row_scale(amax);

  if (!threadIdx.x)
    scales[row] = scale;
#pragma unroll 1
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    if (col < width) {
      T input_value = input[row * width + col];
      if (bias)
        input_value = activation_bias(input_value, bias[col]);
      out[row * width + col] =
          __nv_fp8_e4m3(__fdiv_rn(float(activation_rounded(input_value, kind)), scale));
    }
  }
}

template <int Columns>
__global__ void rms_quant_kernel(const __nv_bfloat16* x,
                                 const __nv_bfloat16* residual,
                                 const __nv_bfloat16* weight,
                                 __nv_fp8_e4m3* out,
                                 __nv_bfloat16* summed,
                                 float* scales,
                                 int width,
                                 float epsilon) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  float variance = 0, values[Columns];
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    float value = col < width ? float(x[row * width + col]) : 0.f;
    if (col < width && residual)
      value += float(residual[row * width + col]);
    values[j] = value;
    variance += value * value;
    if (col < width && residual)
      summed[row * width + col] = __nv_bfloat16(value);
  }
  float inverse = rsqrtf(sum256(variance) / width + epsilon), amax = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    float value =
        col < width ? float(__nv_bfloat16((values[j] * inverse) * float(weight[col]))) : 0.f;
    values[j] = value;
    amax = fmaxf(amax, fabsf(value));
  }
  float scale = row_scale(amax);

  if (!threadIdx.x)
    scales[row] = scale;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    if (col < width)
      out[row * width + col] = __nv_fp8_e4m3(__fdiv_rn(values[j], scale));
  }
}

// Transpose through shared memory to keep the old scalar reduction order while
// issuing vector global loads/stores. The reduction contract remains unchanged.
template <int Groups>
__global__ void rms_quant_vector(const __nv_bfloat16* x,
                                 const __nv_bfloat16* residual,
                                 const __nv_bfloat16* weight,
                                 __nv_fp8_e4m3* out,
                                 __nv_bfloat16* summed,
                                 float* scales,
                                 int width,
                                 float epsilon) {
  wait_for_producer();
  __shared__ float original[Groups * 2048];
  float values[Groups][8];
  const int64_t row = blockIdx.x;
  const int stride = width / 8;
#pragma unroll
  for (int group = 0; group < Groups; ++group) {
    int col = threadIdx.x + group * 256;
    if (col >= stride)
      continue;
    auto input = reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(x)[row * stride + col];
    Packed<__nv_bfloat16, 8> previous{}, result;
    if (residual)
      previous = reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(residual)[row * stride + col];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      float value = float(input.value[lane]);
      if (residual)
        value += float(previous.value[lane]);
      values[group][lane] = value;
      original[col * 8 + lane] = value;
      result.value[lane] = __nv_bfloat16(value);
    }
    if (residual)
      reinterpret_cast<Packed<__nv_bfloat16, 8>*>(summed)[row * stride + col] = result;
  }
  __syncthreads();
  float variance = 0;
  // Preserve the strided reduction order while keeping other values in registers.
  for (int j = 0; j < width / 256; ++j) {
    float value = original[threadIdx.x + j * 256];
    variance += value * value;
  }
  float inverse = rsqrtf(sum256(variance) / width + epsilon), amax = 0;
#pragma unroll
  for (int group = 0; group < Groups; ++group) {
    int col = threadIdx.x + group * 256;
    if (col >= stride)
      continue;
    auto weights = reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(weight)[col];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      float value =
          float(__nv_bfloat16((values[group][lane] * inverse) * float(weights.value[lane])));
      values[group][lane] = value;
      amax = fmaxf(amax, fabsf(value));
    }
  }
  float scale = row_scale(amax);
  const float reciprocal = __frcp_rn(scale);
  // A non-finite maximum needs the original IEEE division, including finite / infinity.
  const bool fast = isfinite(scale);
  if (!threadIdx.x)
    scales[row] = scale;
  if (fast) {
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
      int col = threadIdx.x + group * 256;
      if (col >= stride)
        continue;
      Packed<__nv_fp8x2_e4m3, 4> result;
#pragma unroll
      for (int lane = 0; lane < 4; ++lane)
        result.value[lane] = fp8_quantize_pair_recip(
            values[group][2 * lane], values[group][2 * lane + 1], scale, reciprocal);
      reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
    }
    return;
  }
#pragma unroll
  for (int group = 0; group < Groups; ++group) {
    int col = threadIdx.x + group * 256;
    if (col >= stride)
      continue;
    Packed<__nv_fp8x2_e4m3, 4> result;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane)
      result.value[lane] =
          fp8_quantize_pair(values[group][2 * lane], values[group][2 * lane + 1], scale);
    reinterpret_cast<Packed<__nv_fp8x2_e4m3, 4>*>(out)[row * stride + col] = result;
  }
}

template <typename T> struct alignas(sizeof(T) * 4) PackedScaleOutput {
  T value[4];
};

template <typename T, int Width>
__global__ void scale_output_kernel(const float* values,
                                    const float* row_scales,
                                    const float* column_scales,
                                    T* output,
                                    int64_t rows,
                                    int64_t columns) {
  static_assert(Width == 1 || Width == 4, "Unsupported scale output vector width");
  wait_for_producer();
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const float row_scale = row_scales[row];
    const int64_t offset = row * columns;
    const int64_t stride = int64_t(gridDim.x) * blockDim.x * Width;
    for (int64_t column = int64_t(blockIdx.x) * blockDim.x * Width + int64_t(threadIdx.x) * Width;
         column < columns;
         column += stride) {
      if constexpr (Width == 4) {
        const float4 value = *reinterpret_cast<const float4*>(values + offset + column);
        const float4 scale = *reinterpret_cast<const float4*>(column_scales + column);
        PackedScaleOutput<T> packed;
        packed.value[0] = convert_scaled_output<T>(scale_output_value(value.x, row_scale, scale.x));
        packed.value[1] = convert_scaled_output<T>(scale_output_value(value.y, row_scale, scale.y));
        packed.value[2] = convert_scaled_output<T>(scale_output_value(value.z, row_scale, scale.z));
        packed.value[3] = convert_scaled_output<T>(scale_output_value(value.w, row_scale, scale.w));
        *reinterpret_cast<PackedScaleOutput<T>*>(output + offset + column) = packed;
      } else {
        output[offset + column] = convert_scaled_output<T>(
            scale_output_value(values[offset + column], row_scale, column_scales[column]));
      }
    }
  }
}

template <typename T>
void launch_scale_output(const at::Tensor& values,
                         const at::Tensor& row_scales,
                         const at::Tensor& column_scales,
                         at::Tensor& output) {
  constexpr int threads = 256, vector_width = 4;
  const int64_t rows = values.size(0), columns = values.size(1);
  const auto* value_ptr = values.const_data_ptr<float>();
  const auto* column_ptr = column_scales.const_data_ptr<float>();
  void* output_ptr = output.mutable_data_ptr();
  // One alignment policy for both the FP32 loads and the packed model-dtype store.
  const bool vectorized =
      columns % vector_width == 0 &&
      reinterpret_cast<uintptr_t>(value_ptr) % alignof(float4) == 0 &&
      reinterpret_cast<uintptr_t>(column_ptr) % alignof(float4) == 0 &&
      reinterpret_cast<uintptr_t>(output_ptr) % alignof(PackedScaleOutput<T>) == 0;
  const int64_t width = vectorized ? vector_width : 1;
  int64_t blocks_x = (columns + width * threads - 1) / (width * threads);
  if (blocks_x > 65535)
    blocks_x = 65535;
  const int64_t blocks_y = rows > 65535 ? 65535 : rows;
  const auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  if (vectorized)
    launch_dependent<scale_output_kernel<T, vector_width>>(
        dim3(static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y)),
        dim3(threads),
        stream,
        value_ptr,
        row_scales.const_data_ptr<float>(),
        column_ptr,
        reinterpret_cast<T*>(output_ptr),
        rows,
        columns);
  else
    launch_dependent<scale_output_kernel<T, 1>>(
        dim3(static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y)),
        dim3(threads),
        stream,
        value_ptr,
        row_scales.const_data_ptr<float>(),
        column_ptr,
        reinterpret_cast<T*>(output_ptr),
        rows,
        columns);
}

void validate_matrix(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.size(0) <= INT_MAX &&
                  x.size(1) > 0 && x.size(1) <= 65536,
              "Expected contiguous CUDA matrix with bounded positive width");
}

template <class T, bool Scaled, Fp8GatedActivation Kind, bool PairMath>
void launch_gated_vector(const T* input,
                         const float* scaled_input,
                         const float* row_scales,
                         const float* column_scales,
                         __nv_fp8_e4m3* output,
                         float* scales,
                         int64_t rows,
                         int width,
                         cudaStream_t stream,
                         bool gate_first) {
  const int threads = width <= 4096 ? 256 : 512;
#define VECTOR(C)                                                                                  \
  launch_dependent<gated_quant_vector_kernel<T, C, Scaled, Kind, PairMath>>(rows,                  \
                                                                            threads,               \
                                                                            stream,                \
                                                                            input,                 \
                                                                            scaled_input,          \
                                                                            row_scales,            \
                                                                            column_scales,         \
                                                                            output,                \
                                                                            scales,                \
                                                                            width,                 \
                                                                            gate_first)
  if (width <= 2048) {
    VECTOR(1);
  } else if (width <= 8192) {
    VECTOR(2);
  } else if (width <= 12288) {
    VECTOR(3);
  } else if (width <= 16384) {
    VECTOR(4);
  } else if (width <= 20480) {
    VECTOR(5);
  } else if constexpr (!Scaled) {
    if (width <= 24576) {
      VECTOR(6);
    } else if (width <= 28672) {
      VECTOR(7);
    } else {
      VECTOR(8);
    }
  }
#undef VECTOR
}

template <class T, bool Scaled, Fp8GatedActivation Kind>
void launch_gated_vector_math(const T* input,
                              const float* scaled_input,
                              const float* row_scales,
                              const float* column_scales,
                              __nv_fp8_e4m3* output,
                              float* scales,
                              int64_t rows,
                              int width,
                              cudaStream_t stream,
                              bool pair_math,
                              bool gate_first) {
  if constexpr (Kind == Fp8GatedActivation::PreciseGelu)
    launch_gated_vector<T, Scaled, Kind, false>(input,
                                                scaled_input,
                                                row_scales,
                                                column_scales,
                                                output,
                                                scales,
                                                rows,
                                                width,
                                                stream,
                                                gate_first);
  else if (pair_math)
    launch_gated_vector<T, Scaled, Kind, true>(input,
                                               scaled_input,
                                               row_scales,
                                               column_scales,
                                               output,
                                               scales,
                                               rows,
                                               width,
                                               stream,
                                               gate_first);
  else
    launch_gated_vector<T, Scaled, Kind, false>(input,
                                                scaled_input,
                                                row_scales,
                                                column_scales,
                                                output,
                                                scales,
                                                rows,
                                                width,
                                                stream,
                                                gate_first);
}

template <class T, Fp8GatedActivation Kind>
void launch_gated(const at::Tensor& input,
                  at::Tensor& output,
                  at::Tensor& scales,
                  bool pair_math,
                  bool gate_first) {
  int width = input.size(1) / 2;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto x = reinterpret_cast<const T*>(input.const_data_ptr());
  auto y = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  if (width % 8 == 0 && reinterpret_cast<uintptr_t>(x) % alignof(Packed<T, 8>) == 0) {
    launch_gated_vector_math<T, false, Kind>(x,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             y,
                                             scales.data_ptr<float>(),
                                             input.size(0),
                                             width,
                                             stream,
                                             pair_math,
                                             gate_first);
    return;
  }
#define RUN(C)                                                                                     \
  launch_dependent<gated_quant_kernel<Kind, T, C>>(                                                \
      input.size(0), 256, stream, x, y, scales.data_ptr<float>(), width, gate_first)
  if (width <= 256) {
    RUN(1);
  } else if (width <= 1024) {
    RUN(4);
  } else if (width <= 4096) {
    RUN(16);
  } else if (width <= 12288) {
    RUN(48);
  } else if (width <= 16384) {
    RUN(64);
  } else {
    RUN(128);
  }
#undef RUN
}

template <class T>
void dispatch_gated(const at::Tensor& input,
                    at::Tensor& output,
                    at::Tensor& scales,
                    Fp8GatedActivation kind,
                    bool pair_math,
                    bool gate_first) {
  if (kind == Fp8GatedActivation::RoundedSilu)
    launch_gated<T, Fp8GatedActivation::RoundedSilu>(input, output, scales, pair_math, gate_first);
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (kind == Fp8GatedActivation::PreciseGelu)
      launch_gated<T, Fp8GatedActivation::PreciseGelu>(
          input, output, scales, pair_math, gate_first);
    else
      launch_gated<T, Fp8GatedActivation::RoundedGelu>(
          input, output, scales, pair_math, gate_first);
  } else
    launch_gated<T, Fp8GatedActivation::RoundedGelu>(input, output, scales, pair_math, gate_first);
}

template <class T>
void dispatch_scaled_gated(const float* input,
                           const float* row_scales,
                           const float* columns,
                           __nv_fp8_e4m3* output,
                           float* scales,
                           int64_t rows,
                           int width,
                           cudaStream_t stream,
                           Fp8GatedActivation kind,
                           bool pair_math,
                           bool gate_first) {
  if (kind == Fp8GatedActivation::RoundedSilu)
    launch_gated_vector_math<T, true, Fp8GatedActivation::RoundedSilu>(nullptr,
                                                                       input,
                                                                       row_scales,
                                                                       columns,
                                                                       output,
                                                                       scales,
                                                                       rows,
                                                                       width,
                                                                       stream,
                                                                       pair_math,
                                                                       gate_first);
  else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (kind == Fp8GatedActivation::PreciseGelu)
      launch_gated_vector_math<T, true, Fp8GatedActivation::PreciseGelu>(nullptr,
                                                                         input,
                                                                         row_scales,
                                                                         columns,
                                                                         output,
                                                                         scales,
                                                                         rows,
                                                                         width,
                                                                         stream,
                                                                         pair_math,
                                                                         gate_first);
    else
      launch_gated_vector_math<T, true, Fp8GatedActivation::RoundedGelu>(nullptr,
                                                                         input,
                                                                         row_scales,
                                                                         columns,
                                                                         output,
                                                                         scales,
                                                                         rows,
                                                                         width,
                                                                         stream,
                                                                         pair_math,
                                                                         gate_first);
  } else
    launch_gated_vector_math<T, true, Fp8GatedActivation::RoundedGelu>(nullptr,
                                                                       input,
                                                                       row_scales,
                                                                       columns,
                                                                       output,
                                                                       scales,
                                                                       rows,
                                                                       width,
                                                                       stream,
                                                                       pair_math,
                                                                       gate_first);
}

void validate_gated_activation(Fp8GatedActivation kind) {
  switch (kind) {
  case Fp8GatedActivation::RoundedSilu:
  case Fp8GatedActivation::RoundedGelu:
  case Fp8GatedActivation::PreciseGelu:
    return;
  }
  TORCH_CHECK(false, "Unsupported FP8 gated activation");
}

void validate_activation(Fp8Activation kind) {
  switch (kind) {
  case Fp8Activation::Relu:
  case Fp8Activation::RoundedSilu:
  case Fp8Activation::RoundedGelu:
    return;
  }
  TORCH_CHECK(false, "Unsupported FP8 activation");
}

template <class T, bool Scaled>
void launch_activation_vector(const T* input,
                              const float* scaled_input,
                              const float* row_scales,
                              const float* column_scales,
                              const T* bias,
                              __nv_fp8_e4m3* output,
                              float* scales,
                              int64_t rows,
                              int width,
                              cudaStream_t stream,
                              int kind) {
  const int threads = width <= 4096 ? 256 : 512;
#define VECTOR(C)                                                                                  \
  launch_dependent<activation_quant_vector_kernel<T, C, Scaled>>(rows,                             \
                                                                 threads,                          \
                                                                 stream,                           \
                                                                 input,                            \
                                                                 scaled_input,                     \
                                                                 row_scales,                       \
                                                                 column_scales,                    \
                                                                 bias,                             \
                                                                 output,                           \
                                                                 scales,                           \
                                                                 width,                            \
                                                                 kind)
  if (width <= 2048) {
    VECTOR(1);
  } else if (width <= 8192) {
    VECTOR(2);
  } else if (width <= 12288) {
    VECTOR(3);
  } else if (width <= 16384) {
    VECTOR(4);
  } else {
    VECTOR(5);
  }
#undef VECTOR
}

template <class T>
void launch_activation(const at::Tensor& input,
                       const at::Tensor& bias,
                       at::Tensor& output,
                       at::Tensor& scales,
                       Fp8Activation kind) {
  int width = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto x = reinterpret_cast<const T*>(input.const_data_ptr());
  auto b = bias.defined() ? reinterpret_cast<const T*>(bias.const_data_ptr()) : nullptr;
  auto y = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  int code = static_cast<int>(kind);
  if (width <= fp8_scaled_swiglu_max_width && width % 8 == 0 &&
      reinterpret_cast<uintptr_t>(x) % alignof(Packed<T, 8>) == 0) {
    launch_activation_vector<T, false>(x,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       b,
                                       y,
                                       scales.data_ptr<float>(),
                                       input.size(0),
                                       width,
                                       stream,
                                       code);
    return;
  }
#define RUN(C)                                                                                     \
  launch_dependent<activation_quant_kernel<T, C>>(                                                 \
      input.size(0), 256, stream, x, b, y, scales.data_ptr<float>(), width, code)
  if (width <= 256) {
    RUN(1);
  } else if (width <= 1024) {
    RUN(4);
  } else if (width <= 4096) {
    RUN(16);
  } else if (width <= 12288) {
    RUN(48);
  } else if (width <= 16384) {
    RUN(64);
  } else if (width <= 32768) {
    RUN(128);
  } else {
    RUN(256);
  }
#undef RUN
}
}

at::Tensor fp8_scale_output(const at::Tensor& values,
                            const at::Tensor& row_scales,
                            const at::Tensor& column_scales,
                            at::ScalarType dtype) {
  TORCH_CHECK(values.is_cuda() && values.dim() == 2 && values.is_contiguous() &&
                  values.scalar_type() == at::kFloat,
              "FP8 output scaling requires contiguous FP32 CUDA values");
  TORCH_CHECK(row_scales.is_cuda() && row_scales.dim() == 2 &&
                  row_scales.size(0) == values.size(0) && row_scales.size(1) == 1 &&
                  row_scales.is_contiguous() && row_scales.scalar_type() == at::kFloat &&
                  row_scales.device() == values.device(),
              "Invalid FP8 output row scales");
  TORCH_CHECK(column_scales.is_cuda() && column_scales.dim() == 2 &&
                  column_scales.size(0) == values.size(1) && column_scales.size(1) == 1 &&
                  column_scales.is_contiguous() && column_scales.scalar_type() == at::kFloat &&
                  column_scales.device() == values.device(),
              "Invalid FP8 output column scales");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "FP8 output scaling requires FP16 or BF16 output");
  c10::cuda::CUDAGuard guard(values.device());
  auto output = at::empty(values.sizes(), values.options().dtype(dtype));
  if (values.size(0) && values.size(1)) {
    if (dtype == at::kHalf)
      launch_scale_output<__half>(values, row_scales, column_scales, output);
    else
      launch_scale_output<__nv_bfloat16>(values, row_scales, column_scales, output);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

namespace detail {
std::pair<at::Tensor, at::Tensor> fp8_gated_quantize_impl(const at::Tensor& input,
                                                          Fp8GatedActivation kind,
                                                          bool gate_first,
                                                          bool preserve_legacy_scalar) {
  validate_matrix(input);
  validate_gated_activation(kind);
  TORCH_CHECK(input.size(1) % 2 == 0 &&
                  (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16),
              "Gated FP8 quantization requires even gate/up width and FP16/BF16 input");
  TORCH_CHECK(kind != Fp8GatedActivation::PreciseGelu || input.scalar_type() == at::kBFloat16,
              "PreciseGelu requires BF16 gate/up rows");
  c10::cuda::CUDAGuard guard(input.device());
  auto output =
      at::empty({input.size(0), input.size(1) / 2}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({input.size(0), 1}, input.options().dtype(at::kFloat));
  if (input.size(0)) {
    const auto dtype = input.scalar_type();
    const auto width = input.size(1) / 2;
    // The native aligned pair recipe requires a 16-byte aligned input pointer and an activation
    // width divisible by eight. Scalar and unaligned rows retain the separate scalar staging;
    // callers that pin the frozen staging pass preserve_legacy_scalar. PreciseGelu always keeps
    // the Gemma scalar staging so the separate model-dtype multiply/add rounding is preserved.
    const bool pair_math =
        !preserve_legacy_scalar && kind != Fp8GatedActivation::PreciseGelu && width % 8 == 0 &&
        reinterpret_cast<uintptr_t>(input.const_data_ptr()) % alignof(Packed<__half, 8>) == 0 &&
        (kind == Fp8GatedActivation::RoundedGelu || (dtype == at::kBFloat16 && gate_first));
    if (dtype == at::kHalf)
      dispatch_gated<__half>(input, output, scales, kind, pair_math, gate_first);
    else
      dispatch_gated<__nv_bfloat16>(input, output, scales, kind, pair_math, gate_first);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, scales};
}

std::pair<at::Tensor, at::Tensor> fp8_scaled_gated_quantize_impl(const at::Tensor& values,
                                                                 const at::Tensor& row_scales,
                                                                 const at::Tensor& column_scales,
                                                                 at::ScalarType dtype,
                                                                 Fp8GatedActivation kind,
                                                                 bool gate_first,
                                                                 bool preserve_legacy_scalar) {
  validate_matrix(values);
  validate_gated_activation(kind);
  TORCH_CHECK(values.scalar_type() == at::kFloat && values.size(1) % 2 == 0,
              "Scaled gated FP8 quantization requires contiguous FP32 gate/up rows");
  TORCH_CHECK(kind != Fp8GatedActivation::PreciseGelu || dtype == at::kBFloat16,
              "PreciseGelu requires BF16 output rows");
  const auto rows = values.size(0), width = values.size(1) / 2;
  TORCH_CHECK(row_scales.is_cuda() && row_scales.dim() == 2 && row_scales.size(0) == rows &&
                  row_scales.size(1) == 1 && row_scales.is_contiguous() &&
                  row_scales.scalar_type() == at::kFloat && row_scales.device() == values.device(),
              "Invalid scaled gated FP8 row scales");
  TORCH_CHECK(column_scales.is_cuda() && column_scales.dim() == 2 &&
                  column_scales.size(0) == values.size(1) && column_scales.size(1) == 1 &&
                  column_scales.is_contiguous() && column_scales.scalar_type() == at::kFloat &&
                  column_scales.device() == values.device(),
              "Invalid scaled gated FP8 column scales");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Scaled gated FP8 quantization requires FP16 or BF16 output");
  c10::cuda::CUDAGuard guard(values.device());
  const auto* input = values.const_data_ptr<float>();
  const auto* columns = column_scales.const_data_ptr<float>();
  const bool vectorized = width <= fp8_scaled_swiglu_max_width && width % 8 == 0 &&
                          reinterpret_cast<uintptr_t>(input) % alignof(Packed<float, 8>) == 0 &&
                          reinterpret_cast<uintptr_t>(columns) % alignof(Packed<float, 8>) == 0;
  // fp8_scale_output always allocates an aligned model-dtype destination, so the composed fallback
  // selects the native pair recipe from width alone even when the FP32 sources are unaligned.
  if (rows && !vectorized)
    return fp8_gated_quantize_impl(fp8_scale_output(values, row_scales, column_scales, dtype),
                                   kind,
                                   gate_first,
                                   preserve_legacy_scalar);
  auto output = at::empty({rows, width}, values.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({rows, 1}, values.options().dtype(at::kFloat));
  if (!rows)
    return {output, scales};
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  auto* quantized = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  const bool pair_math =
      !preserve_legacy_scalar && kind != Fp8GatedActivation::PreciseGelu &&
      (kind == Fp8GatedActivation::RoundedGelu || (dtype == at::kBFloat16 && gate_first));
  if (dtype == at::kHalf)
    dispatch_scaled_gated<__half>(input,
                                  row_scales.const_data_ptr<float>(),
                                  columns,
                                  quantized,
                                  scales.data_ptr<float>(),
                                  rows,
                                  width,
                                  stream,
                                  kind,
                                  pair_math,
                                  gate_first);
  else
    dispatch_scaled_gated<__nv_bfloat16>(input,
                                         row_scales.const_data_ptr<float>(),
                                         columns,
                                         quantized,
                                         scales.data_ptr<float>(),
                                         rows,
                                         width,
                                         stream,
                                         kind,
                                         pair_math,
                                         gate_first);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}
}

std::pair<at::Tensor, at::Tensor>
fp8_gated_quantize(const at::Tensor& input, Fp8GatedActivation kind, bool gate_first) {
  return detail::fp8_gated_quantize_impl(input, kind, gate_first, false);
}

std::pair<at::Tensor, at::Tensor> fp8_scaled_gated_quantize(const at::Tensor& values,
                                                            const at::Tensor& row_scales,
                                                            const at::Tensor& column_scales,
                                                            at::ScalarType dtype,
                                                            Fp8GatedActivation kind,
                                                            bool gate_first) {
  return detail::fp8_scaled_gated_quantize_impl(
      values, row_scales, column_scales, dtype, kind, gate_first, false);
}

std::pair<at::Tensor, at::Tensor> fp8_swiglu_quantize(const at::Tensor& gate_up) {
  return detail::fp8_gated_quantize_impl(gate_up, Fp8GatedActivation::RoundedSilu, true, false);
}

std::pair<at::Tensor, at::Tensor> fp8_scaled_swiglu_quantize(const at::Tensor& values,
                                                             const at::Tensor& row_scales,
                                                             const at::Tensor& column_scales,
                                                             at::ScalarType dtype) {
  return detail::fp8_scaled_gated_quantize_impl(
      values, row_scales, column_scales, dtype, Fp8GatedActivation::RoundedSilu, true, false);
}

std::pair<at::Tensor, at::Tensor>
fp8_activation_quantize(const at::Tensor& input, const at::Tensor& bias, Fp8Activation kind) {
  validate_matrix(input);
  validate_activation(kind);
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "FP8 activation quantization requires FP16 or BF16 input");
  TORCH_CHECK(!bias.defined() ||
                  (bias.is_cuda() && bias.dim() == 1 && bias.size(0) == input.size(1) &&
                   bias.device() == input.device() && bias.scalar_type() == input.scalar_type()),
              "Invalid FP8 activation bias");
  c10::cuda::CUDAGuard guard(input.device());
  auto output = at::empty(input.sizes(), input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({input.size(0), 1}, input.options().dtype(at::kFloat));
  if (input.size(0)) {
    const auto bias_values = bias.defined() ? bias.contiguous() : at::Tensor();
    if (input.scalar_type() == at::kHalf)
      launch_activation<__half>(input, bias_values, output, scales, kind);
    else
      launch_activation<__nv_bfloat16>(input, bias_values, output, scales, kind);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, scales};
}

std::pair<at::Tensor, at::Tensor> fp8_scaled_activation_quantize(const at::Tensor& values,
                                                                 const at::Tensor& row_scales,
                                                                 const at::Tensor& column_scales,
                                                                 at::ScalarType dtype,
                                                                 const at::Tensor& bias,
                                                                 Fp8Activation kind) {
  validate_matrix(values);
  validate_activation(kind);
  TORCH_CHECK(values.scalar_type() == at::kFloat,
              "Scaled FP8 activation quantization requires contiguous FP32 rows");
  const auto rows = values.size(0), width = values.size(1);
  TORCH_CHECK(row_scales.is_cuda() && row_scales.dim() == 2 && row_scales.size(0) == rows &&
                  row_scales.size(1) == 1 && row_scales.is_contiguous() &&
                  row_scales.scalar_type() == at::kFloat && row_scales.device() == values.device(),
              "Invalid Scaled FP8 activation row scales");
  TORCH_CHECK(column_scales.is_cuda() && column_scales.dim() == 2 &&
                  column_scales.size(0) == width && column_scales.size(1) == 1 &&
                  column_scales.is_contiguous() && column_scales.scalar_type() == at::kFloat &&
                  column_scales.device() == values.device(),
              "Invalid Scaled FP8 activation column scales");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Scaled FP8 activation requires FP16 or BF16 output");
  TORCH_CHECK(!bias.defined() || (bias.is_cuda() && bias.dim() == 1 && bias.size(0) == width &&
                                  bias.device() == values.device() && bias.scalar_type() == dtype),
              "Invalid Scaled FP8 activation bias");
  c10::cuda::CUDAGuard guard(values.device());
  const auto* input = values.const_data_ptr<float>();
  const auto* columns = column_scales.const_data_ptr<float>();
  const bool vectorized = width <= fp8_scaled_swiglu_max_width && width % 8 == 0 &&
                          reinterpret_cast<uintptr_t>(input) % alignof(Packed<float, 8>) == 0 &&
                          reinterpret_cast<uintptr_t>(columns) % alignof(Packed<float, 8>) == 0;
  if (rows && !vectorized)
    return fp8_activation_quantize(
        fp8_scale_output(values, row_scales, column_scales, dtype), bias, kind);
  auto output = at::empty({rows, width}, values.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({rows, 1}, values.options().dtype(at::kFloat));
  if (!rows)
    return {output, scales};
  const auto bias_values = bias.defined() ? bias.contiguous() : at::Tensor();
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  auto* quantized = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  const int code = static_cast<int>(kind);
  if (dtype == at::kHalf)
    launch_activation_vector<__half, true>(
        nullptr,
        input,
        row_scales.const_data_ptr<float>(),
        columns,
        bias.defined() ? reinterpret_cast<const __half*>(bias_values.const_data_ptr()) : nullptr,
        quantized,
        scales.data_ptr<float>(),
        rows,
        width,
        stream,
        code);
  else
    launch_activation_vector<__nv_bfloat16, true>(
        nullptr,
        input,
        row_scales.const_data_ptr<float>(),
        columns,
        bias.defined() ? reinterpret_cast<const __nv_bfloat16*>(bias_values.const_data_ptr())
                       : nullptr,
        quantized,
        scales.data_ptr<float>(),
        rows,
        width,
        stream,
        code);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, scales};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_add_rms_quantize(const at::Tensor& x,
                                                                    const at::Tensor& residual,
                                                                    const at::Tensor& weight,
                                                                    double epsilon) {
  validate_matrix(x);
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.size(1) <= 8192,
              "Fused FP8 RMS requires BF16 width <=8192");
  TORCH_CHECK(weight.is_contiguous() && weight.dim() == 1 && weight.size(0) == x.size(1) &&
                  weight.device() == x.device() && weight.scalar_type() == x.scalar_type(),
              "Invalid RMS weight");
  TORCH_CHECK(!residual.defined() ||
                  (residual.sizes() == x.sizes() && residual.is_contiguous() &&
                   residual.device() == x.device() && residual.scalar_type() == x.scalar_type()),
              "Invalid RMS residual");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon >= 0, "Invalid RMS epsilon");
  c10::cuda::CUDAGuard guard(x.device());
  auto output = at::empty_like(x, x.options().dtype(at::kFloat8_e4m3fn));
  auto summed = residual.defined() ? at::empty_like(x) : x;
  auto scales = at::empty({x.size(0), 1}, x.options().dtype(at::kFloat));
  if (x.size(0)) {
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto ptr = [](const at::Tensor& t) {
      return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());
    };
    if (x.size(1) % 256 == 0 && reinterpret_cast<uintptr_t>(ptr(x)) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(ptr(weight)) % 16 == 0 &&
        (!residual.defined() || reinterpret_cast<uintptr_t>(ptr(residual)) % 16 == 0)) {
#define RUN_VECTOR(G)                                                                              \
  launch_dependent<rms_quant_vector<G>>(                                                           \
      x.size(0),                                                                                   \
      256,                                                                                         \
      stream,                                                                                      \
      ptr(x),                                                                                      \
      residual.defined() ? ptr(residual) : nullptr,                                                \
      ptr(weight),                                                                                 \
      reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr()),                                 \
      reinterpret_cast<__nv_bfloat16*>(summed.mutable_data_ptr()),                                 \
      scales.data_ptr<float>(),                                                                    \
      x.size(1),                                                                                   \
      epsilon)
      if (x.size(1) <= 2048) {
        RUN_VECTOR(1);
      } else if (x.size(1) <= 4096) {
        RUN_VECTOR(2);
      } else if (x.size(1) <= 6144) {
        RUN_VECTOR(3);
      } else {
        RUN_VECTOR(4);
      }
#undef RUN_VECTOR
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {output, summed, scales};
    }
#define RUN(C)                                                                                     \
  launch_dependent<rms_quant_kernel<C>>(                                                           \
      x.size(0),                                                                                   \
      256,                                                                                         \
      stream,                                                                                      \
      ptr(x),                                                                                      \
      residual.defined() ? ptr(residual) : nullptr,                                                \
      ptr(weight),                                                                                 \
      reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr()),                                 \
      reinterpret_cast<__nv_bfloat16*>(summed.mutable_data_ptr()),                                 \
      scales.data_ptr<float>(),                                                                    \
      x.size(1),                                                                                   \
      epsilon)
    auto width = x.size(1);
    if (width <= 256) {
      RUN(1);
    } else if (width <= 512) {
      RUN(2);
    } else if (width <= 768) {
      RUN(3);
    } else if (width <= 1024) {
      RUN(4);
    } else if (width <= 2048) {
      RUN(8);
    } else if (width <= 4096) {
      RUN(16);
    } else {
      RUN(32);
    }
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, summed, scales};
}
}
