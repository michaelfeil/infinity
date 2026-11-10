// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// MXFP8 block-scale quantization and rounded gated staging.
// The gated pair arithmetic is adapted from Infinity fp8_fusions.cu (Apache-2.0,
// Copyright Infinity Core Contributors) and Candle's MIT-licensed unary math.
// The routed MoE GELU staging matches gemma4_moe_gelu_mul in
// vendor/candle_kernels/kernels/gemma4_moe.cu (MIT, Candle).
// The scale tensor layout follows NVIDIA cuBLAS 13.5 section 3.1.4.4.2.

#include "kernels/mxfp8_fusions.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace infinity {
namespace {
constexpr int64_t kScaleTile = 128;
constexpr int64_t kScaleK = 4;
constexpr int64_t kThreads = 256;
constexpr float kE4M3Max = 448.0f;

__host__ __device__ inline int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

__host__ __device__ inline int64_t scale_inner_dim(int64_t cols) {
  return round_up(cols / 32, kScaleK);
}

__host__ __device__ inline int64_t scale_offset(int64_t outer, int64_t block, int64_t inner_dim) {
  const int64_t sf_outer = outer / kScaleTile;
  const int64_t inner_major = block & ~(kScaleK - 1);
  const int64_t tile_base = (inner_major + sf_outer * inner_dim) * kScaleTile;
  const int64_t within = (outer % 32) * 16 + ((outer % kScaleTile) / 32) * 4 + (block % kScaleK);
  return tile_base + within;
}

__host__ __device__ inline uint8_t block_scale_byte(float amax, float& reciprocal) {
  if (amax == 0.0f) {
    reciprocal = ldexpf(1.0f, 127);
    return 0;
  }
  if (!isfinite(amax)) {
    reciprocal = 1.0f;
    return 255;
  }
  const float target = amax * (1.0f / kE4M3Max);
  int exponent = 0;
  const float mantissa = frexpf(target, &exponent);
  int e = (mantissa == 0.5f) ? exponent - 1 : exponent;
  e = max(-127, min(127, e));
  reciprocal = ldexpf(1.0f, -e);
  return static_cast<uint8_t>(e + 127);
}

template <class T, int N> struct alignas(sizeof(T) * N) Packed {
  T value[N];
};

__global__ void mxfp8_quantize_pack_kernel(const __nv_bfloat16* __restrict__ x,
                                           uint8_t* __restrict__ q,
                                           uint8_t* __restrict__ scales,
                                           int64_t cols,
                                           int64_t inner_dim) {
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int64_t packs = cols / 8;
  for (int64_t pack = threadIdx.x; pack < packs; pack += blockDim.x) {
    const int64_t block = pack / 4;
    const int64_t base = row * cols + pack * 8;
    const auto values = *reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(x + base);
    float local[8];
    float local_max = 0.0f;
    int nonfinite = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      local[i] = __bfloat162float(values.value[i]);
      local_max = fmaxf(local_max, fabsf(local[i]));
      nonfinite |= !isfinite(local[i]);
    }
    const unsigned mask = __activemask();
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 2));
    int any_nonfinite = nonfinite;
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 1);
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 2);
    float reciprocal = 0.0f;
    const uint8_t scale = any_nonfinite ? 255 : block_scale_byte(local_max, reciprocal);
    if ((lane & 3) == 0)
      scales[scale_offset(row, block, inner_dim)] = scale;
    Packed<uint8_t, 8> quantized;
    if (any_nonfinite) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = 0x7f;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(local[i] * reciprocal, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint64_t*>(q + base) = *reinterpret_cast<const uint64_t*>(&quantized);
  }
}

__device__ __forceinline__ __nv_bfloat162 gated_silu_pair(__nv_bfloat162 gate, __nv_bfloat162 up) {
  const auto exponential = h2exp(__hneg2(gate));
  const auto denominator = __hadd2(__float2bfloat162_rn(1.0f), exponential);
  return __hmul2(__h2div(gate, denominator), up);
}

// gemma4_moe_gelu_mul evaluates `gate + 0.044715f * gate * gate * gate` in FP32,
// which nvcc contracts into fmaf((0.044715f * gate) * gate, gate, gate). Spell
// that contraction out so this fused copy is independent of contraction
// heuristics. Every remaining stage stays in FP32 exactly as in the MoE kernel:
// multiply by 0.7978845608028654f, tanhf, (1 + tanh), then scale by 0.5 * gate.
__device__ __forceinline__ float moe_gelu_tanh(float gate) {
  const float cubic = (0.044715f * gate) * gate;
  const float alpha = __fmaf_rn(cubic, gate, gate);
  return (0.5f * gate) * (1.0f + tanhf(0.7978845608028654f * alpha));
}

// One CTA per logical routed slot. The slot's physical row is read from row_map
// and reused for the value and scale stores, so no physical padding row is read.
__global__ void moe_gelu_mxfp8_quantize_pack_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                                    const int32_t* __restrict__ row_map,
                                                    uint8_t* __restrict__ q,
                                                    uint8_t* __restrict__ scales,
                                                    int64_t width,
                                                    int64_t inner_dim) {
  const int64_t row = row_map[blockIdx.x];
  const int lane = threadIdx.x & 31;
  const int64_t packs = width / 8;
  for (int64_t pack = threadIdx.x; pack < packs; pack += blockDim.x) {
    const int64_t block = pack / 4;
    const int64_t gate_base = row * 2 * width + pack * 8;
    const auto gate = *reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(gate_up + gate_base);
    const auto up = *reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(gate_up + gate_base + width);
    float local[8];
    float local_max = 0.0f;
    int nonfinite = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gelu = moe_gelu_tanh(__bfloat162float(gate.value[i]));
      const float rounded_gelu = __bfloat162float(__float2bfloat16_rn(gelu));
      const float activated =
          __bfloat162float(__float2bfloat16_rn(rounded_gelu * __bfloat162float(up.value[i])));
      local[i] = activated;
      local_max = fmaxf(local_max, fabsf(local[i]));
      nonfinite |= !isfinite(local[i]);
    }
    const unsigned mask = __activemask();
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 2));
    int any_nonfinite = nonfinite;
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 1);
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 2);
    float reciprocal = 0.0f;
    const uint8_t scale = any_nonfinite ? 255 : block_scale_byte(local_max, reciprocal);
    if ((lane & 3) == 0)
      scales[scale_offset(row, block, inner_dim)] = scale;
    Packed<uint8_t, 8> quantized;
    if (any_nonfinite) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = 0x7f;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(local[i] * reciprocal, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint64_t*>(q + row * width + pack * 8) =
        *reinterpret_cast<const uint64_t*>(&quantized);
  }
}

__global__ void gated_silu_mxfp8_quantize_pack_kernel(const __nv_bfloat16* __restrict__ gate_up,
                                                      uint8_t* __restrict__ q,
                                                      uint8_t* __restrict__ scales,
                                                      int64_t width,
                                                      int64_t inner_dim) {
  const int64_t row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int64_t packs = width / 8;
  for (int64_t pack = threadIdx.x; pack < packs; pack += blockDim.x) {
    const int64_t block = pack / 4;
    const int64_t base = row * 2 * width + pack * 8;
    const auto gate = *reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(gate_up + base);
    const auto up = *reinterpret_cast<const Packed<__nv_bfloat16, 8>*>(gate_up + base + width);
    const auto* gate_pairs = reinterpret_cast<const __nv_bfloat162*>(gate.value);
    const auto* up_pairs = reinterpret_cast<const __nv_bfloat162*>(up.value);
    float local[8];
    float local_max = 0.0f;
    int nonfinite = 0;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const auto activated = gated_silu_pair(gate_pairs[pair], up_pairs[pair]);
      local[2 * pair] = __bfloat162float(__low2bfloat16(activated));
      local[2 * pair + 1] = __bfloat162float(__high2bfloat16(activated));
      local_max = fmaxf(local_max, fmaxf(fabsf(local[2 * pair]), fabsf(local[2 * pair + 1])));
      nonfinite |= !isfinite(local[2 * pair]) || !isfinite(local[2 * pair + 1]);
    }
    const unsigned mask = __activemask();
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 1));
    local_max = fmaxf(local_max, __shfl_xor_sync(mask, local_max, 2));
    int any_nonfinite = nonfinite;
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 1);
    any_nonfinite |= __shfl_xor_sync(mask, any_nonfinite, 2);
    float reciprocal = 0.0f;
    const uint8_t scale = any_nonfinite ? 255 : block_scale_byte(local_max, reciprocal);
    if ((lane & 3) == 0)
      scales[scale_offset(row, block, inner_dim)] = scale;
    Packed<uint8_t, 8> quantized;
    if (any_nonfinite) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = 0x7f;
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        quantized.value[i] = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(local[i] * reciprocal, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint64_t*>(q + row * width + pack * 8) =
        *reinterpret_cast<const uint64_t*>(&quantized);
  }
}
} // namespace

at::Tensor mxfp8_aligned_tensor(const at::Tensor& input,
                                at::ScalarType dtype,
                                int64_t dims,
                                const char* label) {
  TORCH_CHECK(input.is_cuda() && input.dim() == dims && input.scalar_type() == dtype,
              label,
              " requires a contiguous CUDA tensor of the expected rank and dtype");
  if (input.numel() == 0)
    return input.contiguous();
  auto contiguous = input.contiguous();
  if (reinterpret_cast<uintptr_t>(contiguous.const_data_ptr()) % 16 == 0)
    return contiguous;
  auto aligned = at::empty(contiguous.sizes(), contiguous.options());
  aligned.copy_(contiguous);
  return aligned;
}

int64_t mxfp8_scale_elements(int64_t rows, int64_t cols) {
  TORCH_CHECK(rows >= 0 && cols > 0 && cols % 32 == 0,
              "MXFP8 scale geometry requires rows>=0 and K divisible by 32");
  return round_up(rows, kScaleTile) * scale_inner_dim(cols);
}

Mxfp8Quantized mxfp8_quantize_rows(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "MXFP8 quantize requires 2D CUDA input");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "MXFP8 quantize requires BF16 input");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % 32 == 0, "MXFP8 quantize requires K divisible by 32");
  c10::cuda::CUDAGuard guard(input.device());
  Mxfp8Quantized out;
  out.rows = rows;
  out.cols = cols;
  if (rows == 0) {
    out.values = at::empty({0, cols}, input.options().dtype(at::kFloat8_e4m3fn));
    out.scales = at::zeros({0}, input.options().dtype(at::kByte));
    return out;
  }
  auto x = mxfp8_aligned_tensor(input, at::kBFloat16, 2, "MXFP8 quantize input");
  out.values = at::empty({rows, cols}, input.options().dtype(at::kFloat8_e4m3fn));
  const int64_t inner = scale_inner_dim(cols);
  out.scales = at::zeros({mxfp8_scale_elements(rows, cols)}, input.options().dtype(at::kByte));
  mxfp8_quantize_pack_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      reinterpret_cast<uint8_t*>(out.values.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(out.scales.mutable_data_ptr()),
      cols,
      inner);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Mxfp8Quantized mxfp8_gated_silu_quantize(const at::Tensor& gate_up) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 2 && gate_up.scalar_type() == at::kBFloat16 &&
                  gate_up.size(1) % 64 == 0,
              "MXFP8 gated quantization requires 2D BF16 [M,2I]");
  const int64_t rows = gate_up.size(0), width = gate_up.size(1) / 2;
  TORCH_CHECK(width % 32 == 0, "MXFP8 gated width must be divisible by 32");
  c10::cuda::CUDAGuard guard(gate_up.device());
  Mxfp8Quantized out;
  out.rows = rows;
  out.cols = width;
  if (rows == 0 || width == 0) {
    out.values = at::empty({rows, width}, gate_up.options().dtype(at::kFloat8_e4m3fn));
    out.scales = at::zeros({0}, gate_up.options().dtype(at::kByte));
    return out;
  }
  auto gate = mxfp8_aligned_tensor(gate_up, at::kBFloat16, 2, "MXFP8 gated input");
  out.values = at::empty({rows, width}, gate_up.options().dtype(at::kFloat8_e4m3fn));
  const int64_t inner = scale_inner_dim(width);
  out.scales = at::zeros({mxfp8_scale_elements(rows, width)}, gate_up.options().dtype(at::kByte));
  gated_silu_mxfp8_quantize_pack_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.const_data_ptr()),
      reinterpret_cast<uint8_t*>(out.values.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(out.scales.mutable_data_ptr()),
      width,
      inner);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

Mxfp8Quantized mxfp8_moe_gelu_quantize(const at::Tensor& gate_up, const at::Tensor& row_map) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 2 && gate_up.is_contiguous() &&
                  gate_up.scalar_type() == at::kBFloat16,
              "MXFP8 MoE GELU quantization requires contiguous CUDA BF16 [rows,2I]");
  TORCH_CHECK(row_map.is_cuda() && row_map.dim() == 1 && row_map.is_contiguous() &&
                  row_map.scalar_type() == at::kInt && row_map.device() == gate_up.device(),
              "MXFP8 MoE GELU quantization requires a contiguous int32 CUDA row map");
  const int64_t rows = gate_up.size(0);
  TORCH_CHECK(gate_up.size(1) % 64 == 0,
              "MXFP8 MoE GELU gate/up width must be twice a multiple of 32");
  const int64_t width = gate_up.size(1) / 2;
  const int64_t slots = row_map.numel();
  c10::cuda::CUDAGuard guard(gate_up.device());
  Mxfp8Quantized out;
  out.rows = rows;
  out.cols = width;
  if (rows == 0 || width == 0) {
    out.values = at::empty({rows, width}, gate_up.options().dtype(at::kFloat8_e4m3fn));
    out.scales = at::zeros({0}, gate_up.options().dtype(at::kByte));
    return out;
  }
  auto gate = mxfp8_aligned_tensor(gate_up, at::kBFloat16, 2, "MXFP8 MoE GELU input");
  out.values = at::empty({rows, width}, gate_up.options().dtype(at::kFloat8_e4m3fn));
  const int64_t inner = scale_inner_dim(width);
  out.scales = at::zeros({mxfp8_scale_elements(rows, width)}, gate_up.options().dtype(at::kByte));
  if (slots == 0)
    return out;
  moe_gelu_mxfp8_quantize_pack_kernel<<<static_cast<unsigned>(slots),
                                        kThreads,
                                        0,
                                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(gate.const_data_ptr()),
      row_map.const_data_ptr<int32_t>(),
      reinterpret_cast<uint8_t*>(out.values.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(out.scales.mutable_data_ptr()),
      width,
      inner);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
} // namespace infinity
