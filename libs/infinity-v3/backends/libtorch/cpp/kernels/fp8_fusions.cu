// SPDX-License-Identifier: Apache-2.0
// RMS arithmetic adapted from the Apache-2.0 fast_kernels.cu implementation.
// Preserve its reduction order and BF16 rounding before row quantization.
#include "kernels/fp8_fusions.h"
#include "kernels/dependent_launch.cuh"
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

template <class T, int Columns>
__global__ void swiglu_quant_kernel(const T* input, __nv_fp8_e4m3* out, float* scales, int width) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  float values[Columns], amax = 0;
#pragma unroll
  for (int j = 0; j < Columns; ++j) {
    int col = threadIdx.x + j * 256;
    float value = 0;
    if (col < width) {
      T gate = input[row * 2 * width + col], up = input[row * 2 * width + width + col];
      T exponential = hexp(T(-gate));
      T denominator = T(T(1) + exponential);
      T activated = T(gate / denominator);
      value = float(T(float(activated) * float(up)));
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

__device__ __forceinline__ float scale_output_value(float value, float row, float column) {
  return __fmul_rn(__fmul_rn(value, row), column);
}

template <typename T> __device__ __forceinline__ T convert_scaled_output(float value) {
  if constexpr (std::is_same_v<T, __half>)
    return __float2half_rn(value);
  else
    return __float2bfloat16_rn(value);
}

template <class T> __device__ __forceinline__ T swiglu_rounded(T gate, T up) {
  T exponential = hexp(T(-gate));
  T denominator = T(T(1) + exponential);
  T activated = T(gate / denominator);
  return T(float(activated) * float(up));
}

template <class T, bool Scaled>
__device__ __forceinline__ void load_swiglu_pair(const T* input,
                                                 const float* scaled_input,
                                                 const float* row_scales,
                                                 const float* column_scales,
                                                 int64_t row,
                                                 int stride,
                                                 int col,
                                                 Packed<T, 8>& gate,
                                                 Packed<T, 8>& up) {
  if constexpr (Scaled) {
    const float row_scale_value = row_scales[row];
    const auto gate_raw =
        reinterpret_cast<const Packed<float, 8>*>(scaled_input)[row * 2 * stride + col];
    const auto up_raw =
        reinterpret_cast<const Packed<float, 8>*>(scaled_input)[row * 2 * stride + stride + col];
    const auto gate_column = reinterpret_cast<const Packed<float, 8>*>(column_scales)[col];
    const auto up_column = reinterpret_cast<const Packed<float, 8>*>(column_scales)[stride + col];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      gate.value[lane] = convert_scaled_output<T>(
          scale_output_value(gate_raw.value[lane], row_scale_value, gate_column.value[lane]));
      up.value[lane] = convert_scaled_output<T>(
          scale_output_value(up_raw.value[lane], row_scale_value, up_column.value[lane]));
    }
  } else {
    gate = reinterpret_cast<const Packed<T, 8>*>(input)[row * 2 * stride + col];
    up = reinterpret_cast<const Packed<T, 8>*>(input)[row * 2 * stride + stride + col];
  }
}

template <class T, int Groups, bool Scaled = false>
__global__ __launch_bounds__(512) void swiglu_quant_vector_kernel(const T* input,
                                                                  const float* scaled_input,
                                                                  const float* row_scales,
                                                                  const float* column_scales,
                                                                  __nv_fp8_e4m3* out,
                                                                  float* scales,
                                                                  int width) {
  wait_for_producer();
  const int64_t row = blockIdx.x;
  // Scaled rows fit the common 48 KiB block limit (at most five groups). Ordinary SwiGLU keeps
  // its original register storage; wider scaled rows use the composed fallback.
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
    load_swiglu_pair<T, Scaled>(
        input, scaled_input, row_scales, column_scales, row, stride, col, gate, up);
    Packed<T, 8> rounded;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      T value = swiglu_rounded(gate.value[lane], up.value[lane]);
      if constexpr (Scaled)
        rounded.value[lane] = value;
      else
        values[j][lane] = float(value);
      amax = fmaxf(amax, fabsf(float(value)));
    }
    if constexpr (Scaled)
      staged[col] = rounded;
  }
  // row_scale also synchronizes the completed shared row before reuse below.
  float scale = row_scale(amax);

  if (!threadIdx.x)
    scales[row] = scale;
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
  if (!threadIdx.x)
    scales[row] = scale;
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

template <class T, bool Scaled>
void launch_swiglu_vector(const T* input,
                          const float* scaled_input,
                          const float* row_scales,
                          const float* column_scales,
                          __nv_fp8_e4m3* output,
                          float* scales,
                          int64_t rows,
                          int width,
                          cudaStream_t stream) {
  const int threads = width <= 4096 ? 256 : 512;
#define VECTOR(C)                                                                                  \
  launch_dependent<swiglu_quant_vector_kernel<T, C, Scaled>>(rows,                                 \
                                                             threads,                              \
                                                             stream,                               \
                                                             input,                                \
                                                             scaled_input,                         \
                                                             row_scales,                           \
                                                             column_scales,                        \
                                                             output,                               \
                                                             scales,                               \
                                                             width)
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

template <class T>
void launch_swiglu(const at::Tensor& input, at::Tensor& output, at::Tensor& scales) {
  int width = input.size(1) / 2;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto x = reinterpret_cast<const T*>(input.const_data_ptr());
  auto y = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  if (width % 8 == 0 && reinterpret_cast<uintptr_t>(x) % alignof(Packed<T, 8>) == 0) {
    launch_swiglu_vector<T, false>(
        x, nullptr, nullptr, nullptr, y, scales.data_ptr<float>(), input.size(0), width, stream);
    return;
  }
#define RUN(C)                                                                                     \
  launch_dependent<swiglu_quant_kernel<T, C>>(                                                     \
      input.size(0), 256, stream, x, y, scales.data_ptr<float>(), width)
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

/// Produce FP8 gated-activation rows and F32 row scales without
/// materializing a separate activation tensor.
std::pair<at::Tensor, at::Tensor> fp8_swiglu_quantize(const at::Tensor& input) {
  validate_matrix(input);
  TORCH_CHECK(input.size(1) % 2 == 0 &&
                  (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16),
              "SwiGLU requires even gate/up width and FP16/BF16 input");
  c10::cuda::CUDAGuard guard(input.device());
  auto output =
      at::empty({input.size(0), input.size(1) / 2}, input.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({input.size(0), 1}, input.options().dtype(at::kFloat));
  if (input.size(0)) {
    if (input.scalar_type() == at::kHalf)
      launch_swiglu<__half>(input, output, scales);
    else
      launch_swiglu<__nv_bfloat16>(input, output, scales);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, scales};
}

std::pair<at::Tensor, at::Tensor> fp8_scaled_swiglu_quantize(const at::Tensor& values,
                                                             const at::Tensor& row_scales,
                                                             const at::Tensor& column_scales,
                                                             at::ScalarType dtype) {
  validate_matrix(values);
  TORCH_CHECK(values.scalar_type() == at::kFloat && values.size(1) % 2 == 0,
              "Scaled SwiGLU requires contiguous FP32 gate/up rows");
  const auto rows = values.size(0), width = values.size(1) / 2;
  TORCH_CHECK(row_scales.is_cuda() && row_scales.dim() == 2 && row_scales.size(0) == rows &&
                  row_scales.size(1) == 1 && row_scales.is_contiguous() &&
                  row_scales.scalar_type() == at::kFloat && row_scales.device() == values.device(),
              "Invalid Scaled SwiGLU row scales");
  TORCH_CHECK(column_scales.is_cuda() && column_scales.dim() == 2 &&
                  column_scales.size(0) == values.size(1) && column_scales.size(1) == 1 &&
                  column_scales.is_contiguous() && column_scales.scalar_type() == at::kFloat &&
                  column_scales.device() == values.device(),
              "Invalid Scaled SwiGLU column scales");
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Scaled SwiGLU requires FP16 or BF16 output");
  c10::cuda::CUDAGuard guard(values.device());
  const auto* input = values.const_data_ptr<float>();
  const auto* columns = column_scales.const_data_ptr<float>();
  const bool vectorized = width <= fp8_scaled_swiglu_max_width && width % 8 == 0 &&
                          reinterpret_cast<uintptr_t>(input) % alignof(Packed<float, 8>) == 0 &&
                          reinterpret_cast<uintptr_t>(columns) % alignof(Packed<float, 8>) == 0;
  if (rows && !vectorized)
    return fp8_swiglu_quantize(fp8_scale_output(values, row_scales, column_scales, dtype));
  auto output = at::empty({rows, width}, values.options().dtype(at::kFloat8_e4m3fn));
  auto scales = at::empty({rows, 1}, values.options().dtype(at::kFloat));
  if (!rows)
    return {output, scales};
  auto stream = at::cuda::getCurrentCUDAStream(values.get_device());
  auto* quantized = reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  if (dtype == at::kHalf)
    launch_swiglu_vector<__half, true>(nullptr,
                                       input,
                                       row_scales.const_data_ptr<float>(),
                                       columns,
                                       quantized,
                                       scales.data_ptr<float>(),
                                       rows,
                                       width,
                                       stream);
  else
    launch_swiglu_vector<__nv_bfloat16, true>(nullptr,
                                              input,
                                              row_scales.const_data_ptr<float>(),
                                              columns,
                                              quantized,
                                              scales.data_ptr<float>(),
                                              rows,
                                              width,
                                              stream);
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
