// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// NVFP4 two-level quantization: packed E2M1 values with E4M3 block scales.
// Block scales use the 128x4 SWIZZLE_32_4_4 layout from the cuBLAS block-scale
// documentation (section 3.1.4.3.2). No host synchronization is performed.
#include "helpers/nvfp4_linear.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace infinity {
namespace {
constexpr int64_t kScaleTile = 128;
constexpr int64_t kScaleK = 4;
constexpr int64_t kBlock = 16;
constexpr int kThreads = 256;
constexpr float kE4M3Max = 448.0f;
constexpr float kFp4Max = 6.0f;

__host__ __device__ inline int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

__host__ __device__ inline int64_t scale_inner_dim(int64_t cols) {
  return round_up(cols / kBlock, kScaleK);
}

__host__ __device__ inline int64_t scale_offset(int64_t outer, int64_t block, int64_t inner_dim) {
  const int64_t sf_outer = outer / kScaleTile;
  const int64_t inner_major = block & ~(kScaleK - 1);
  const int64_t tile_base = (inner_major + sf_outer * inner_dim) * kScaleTile;
  const int64_t within = (outer % 32) * 16 + ((outer % kScaleTile) / 32) * 4 + (block % kScaleK);
  return tile_base + within;
}

__device__ inline int warp_max(int value) {
#pragma unroll
  for (int delta = 16; delta; delta /= 2)
    value = max(value, __shfl_down_sync(0xffffffff, value, delta));
  return value;
}

template <class T, int N> struct alignas(sizeof(T) * N) VectorPack {
  T value[N];
};

__device__ inline void block_atomic_max(int local, int* out) {
  local = warp_max(local);
  __shared__ int partial[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0)
    partial[warp] = local;
  __syncthreads();
  if (warp == 0) {
    int block_max = lane < (blockDim.x >> 5) ? partial[lane] : 0;
    block_max = warp_max(block_max);
    if (lane == 0)
      atomicMax(out, block_max);
  }
}

__device__ inline float global_scale_from_bits(const int* amax_bits, bool power2) {
  const float amax = __int_as_float(__ldg(amax_bits));
  const float scale = __fdiv_rn(amax, kFp4Max * kE4M3Max);
  if (!(scale > 0.0f))
    return 1.0f;
  if (!power2)
    return scale;
  int exponent = 0;
  const float mantissa = frexpf(scale, &exponent);
  if (mantissa == 0.5f)
    --exponent;
  return ldexpf(1.0f, exponent);
}

// Exact arithmetic used by the shared fused_gated_activation code-5 pair kernel.
// Every intermediate is rounded to BF16 in the same order.
__device__ inline __nv_bfloat162 gated_silu_pair(__nv_bfloat162 gate, __nv_bfloat162 up) {
  const auto exponential = h2exp(__hneg2(gate));
  const auto denominator = __hadd2(__float2bfloat162_rn(1.0f), exponential);
  return __hmul2(__h2div(gate, denominator), up);
}

__global__ void
global_amax_kernel(const __nv_bfloat16* __restrict__ x, int64_t count, int* __restrict__ out) {
  const auto* vectors = reinterpret_cast<const VectorPack<__nv_bfloat16, 8>*>(x);
  const int64_t vector_count = count / 8;
  int local = 0;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < vector_count;
       i += int64_t(gridDim.x) * blockDim.x) {
    const auto pack = vectors[i];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const float value = fabsf(__bfloat162float(pack.value[lane]));
      if (isfinite(value))
        local = max(local, __float_as_int(value));
    }
  }
  block_atomic_max(local, out);
}

__device__ inline float decode_e4m3(uint8_t byte) {
  const __half_raw half = __nv_cvt_fp8_to_halfraw(byte, __NV_E4M3);
  return __half2float(*reinterpret_cast<const __half*>(&half));
}

__device__ inline uint64_t
nvfp4_encode_with_scale(const float* values, float gs, uint8_t scale_byte) {
  const float block_scale = decode_e4m3(scale_byte);
  uint64_t packed = 0;
  if (block_scale > 0.0f) {
    const float denom = block_scale * gs;
    const float reciprocal = __frcp_rn(denom);
#pragma unroll
    for (int pair = 0; pair < kBlock / 2; ++pair) {
      const float lo = isfinite(values[2 * pair]) ? __fmul_rn(values[2 * pair], reciprocal) : 0.0f;
      const float hi =
          isfinite(values[2 * pair + 1]) ? __fmul_rn(values[2 * pair + 1], reciprocal) : 0.0f;
      packed |= uint64_t(__nv_cvt_float2_to_fp4x2(make_float2(lo, hi), __NV_E2M1, cudaRoundNearest))
                << (8 * pair);
    }
  }
  return packed;
}

__device__ inline uint8_t nvfp4_standard_scale(const float* values, float gs) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlock; ++i)
    if (isfinite(values[i]))
      amax = fmaxf(amax, fabsf(values[i]));
  if (!(amax > 0.0f))
    return 0;
  const float target = __fdiv_rn(amax, kFp4Max * gs);
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(target, __NV_SATFINITE, __NV_E4M3));
}

__device__ inline uint8_t nvfp4_quantize_block(const float* values, float gs, uint64_t& packed) {
  const uint8_t scale_byte = nvfp4_standard_scale(values, gs);
  packed = nvfp4_encode_with_scale(values, gs, scale_byte);
  return scale_byte;
}

__global__ void nvfp4_quantize_kernel(const __nv_bfloat16* __restrict__ x,
                                      uint8_t* __restrict__ q,
                                      uint8_t* __restrict__ scales,
                                      int64_t cols,
                                      int64_t inner_dim,
                                      const int* __restrict__ amax_bits,
                                      float* __restrict__ global_scale,
                                      bool power2_global) {
  const int64_t row = blockIdx.x;
  const int64_t blocks = cols / kBlock;
  const float gs = global_scale_from_bits(amax_bits, power2_global);
  if (blockIdx.x == 0 && threadIdx.x == 0)
    global_scale[0] = gs;
  const auto* vectors = reinterpret_cast<const VectorPack<__nv_bfloat16, 8>*>(x + row * cols);
  for (int64_t block = threadIdx.x; block < blocks; block += blockDim.x) {
    const auto lo_pack = vectors[block * 2];
    const auto hi_pack = vectors[block * 2 + 1];
    float values[kBlock];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      values[lane] = __bfloat162float(lo_pack.value[lane]);
      values[8 + lane] = __bfloat162float(hi_pack.value[lane]);
    }
    uint64_t packed = 0;
    const uint8_t scale_byte = nvfp4_quantize_block(values, gs, packed);
    scales[scale_offset(row, block, inner_dim)] = scale_byte;
    *reinterpret_cast<uint64_t*>(q + row * (cols / 2) + block * (kBlock / 2)) = packed;
  }
}

__global__ void swiglu_activation_amax_kernel(const __nv_bfloat16* __restrict__ x,
                                              __nv_bfloat16* __restrict__ activated,
                                              int64_t width,
                                              int64_t tiles_per_row,
                                              int* __restrict__ out) {
  const int64_t row = int64_t(blockIdx.x) / tiles_per_row;
  const int64_t tile = int64_t(blockIdx.x) % tiles_per_row;
  const int64_t vectors_per_row = width / 8;
  const auto* base = reinterpret_cast<const VectorPack<__nv_bfloat162, 4>*>(x + row * 2 * width);
  auto* output = reinterpret_cast<VectorPack<__nv_bfloat162, 4>*>(activated + row * width);
  int local = 0;
  for (int64_t col = tile * int64_t(blockDim.x) + threadIdx.x; col < vectors_per_row;
       col += tiles_per_row * int64_t(blockDim.x)) {
    const auto gate = base[col];
    const auto up = base[vectors_per_row + col];
    VectorPack<__nv_bfloat162, 4> result;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
      const auto value = gated_silu_pair(gate.value[lane], up.value[lane]);
      result.value[lane] = value;
      const float lo = __bfloat162float(__low2bfloat16(value));
      const float hi = __bfloat162float(__high2bfloat16(value));
      if (isfinite(lo))
        local = max(local, __float_as_int(fabsf(lo)));
      if (isfinite(hi))
        local = max(local, __float_as_int(fabsf(hi)));
    }
    output[col] = result;
  }
  block_atomic_max(local, out);
}
} // namespace

int64_t nvfp4_scale_elements(int64_t rows, int64_t cols) {
  TORCH_CHECK(rows >= 0 && cols > 0 && cols % kBlock == 0,
              "NVFP4 scale geometry requires rows>=0 and K divisible by 16");
  return round_up(rows, kScaleTile) * scale_inner_dim(cols);
}

at::Tensor nvfp4_aligned_tensor(const at::Tensor& input,
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

Nvfp4Quantized nvfp4_quantize_rows(const at::Tensor& input, bool power2_global) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 2, "NVFP4 quantize requires 2D CUDA input");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "NVFP4 quantize requires BF16 input");
  const int64_t rows = input.size(0), cols = input.size(1);
  TORCH_CHECK(cols % kBlock == 0, "NVFP4 quantize requires K divisible by 16");
  c10::cuda::CUDAGuard guard(input.device());
  Nvfp4Quantized out;
  out.rows = rows;
  out.cols = cols;
  const auto byte_options = input.options().dtype(at::kByte);
  if (rows == 0) {
    out.values = at::empty({0, cols / 2}, byte_options).view(at::kFloat4_e2m1fn_x2);
    out.block_scales = at::zeros({0}, byte_options).view(at::kFloat8_e4m3fn);
    out.global_scale = at::ones({1}, input.options().dtype(at::kFloat));
    return out;
  }
  auto x = nvfp4_aligned_tensor(input, at::kBFloat16, 2, "NVFP4 quantize input");
  auto values_bytes = at::empty({rows, cols / 2}, byte_options);
  const int64_t inner = scale_inner_dim(cols);
  auto scales = at::zeros({nvfp4_scale_elements(rows, cols)}, byte_options);
  auto amax_bits = at::zeros({1}, input.options().dtype(at::kInt));
  auto global_scale = at::empty({1}, input.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t count = rows * cols;
  const int reduce_blocks =
      static_cast<int>(std::min<int64_t>(((count / 8) + kThreads - 1) / kThreads, 2048));
  global_amax_kernel<<<reduce_blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()), count, amax_bits.data_ptr<int>());
  nvfp4_quantize_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      reinterpret_cast<uint8_t*>(values_bytes.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(scales.mutable_data_ptr()),
      cols,
      inner,
      amax_bits.data_ptr<int>(),
      global_scale.data_ptr<float>(),
      power2_global);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  out.values = values_bytes.view(at::kFloat4_e2m1fn_x2);
  out.block_scales = scales.view(at::kFloat8_e4m3fn);
  out.global_scale = global_scale;
  return out;
}

Nvfp4Quantized nvfp4_swiglu_quantize(const at::Tensor& gate_up) {
  TORCH_CHECK(gate_up.is_cuda() && gate_up.dim() == 2 && gate_up.scalar_type() == at::kBFloat16,
              "NVFP4 SwiGLU quantization requires 2D BF16 CUDA gate/up rows");
  const int64_t rows = gate_up.size(0), cols = gate_up.size(1);
  TORCH_CHECK(cols % (2 * kBlock) == 0,
              "NVFP4 SwiGLU requires a gate/up width with an activated width divisible by 16");
  const int64_t width = cols / 2;
  TORCH_CHECK(width > 0, "NVFP4 SwiGLU requires a nonempty activated width");
  c10::cuda::CUDAGuard guard(gate_up.device());
  Nvfp4Quantized out;
  out.rows = rows;
  out.cols = width;
  const auto byte_options = gate_up.options().dtype(at::kByte);
  if (rows == 0) {
    out.values = at::empty({0, width / 2}, byte_options).view(at::kFloat4_e2m1fn_x2);
    out.block_scales = at::zeros({0}, byte_options).view(at::kFloat8_e4m3fn);
    out.global_scale = at::ones({1}, gate_up.options().dtype(at::kFloat));
    return out;
  }
  auto x = nvfp4_aligned_tensor(gate_up, at::kBFloat16, 2, "NVFP4 SwiGLU input");
  auto activated = at::empty({rows, width}, gate_up.options().dtype(at::kBFloat16));
  auto values_bytes = at::empty({rows, width / 2}, byte_options);
  const int64_t inner = scale_inner_dim(width);
  auto scales = at::zeros({nvfp4_scale_elements(rows, width)}, byte_options);
  auto amax_bits = at::zeros({1}, gate_up.options().dtype(at::kInt));
  auto global_scale = at::empty({1}, gate_up.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t vectors_per_row = width / 8;
  const int64_t tiles_per_row = (vectors_per_row + kThreads - 1) / kThreads;
  const int64_t total_tiles = rows * tiles_per_row;
  TORCH_CHECK(total_tiles <= 2147483647, "NVFP4 SwiGLU grid is too large");
  // Flatten to 1D: grid.y is limited to 65535, while model batches can exceed it.
  swiglu_activation_amax_kernel<<<static_cast<unsigned>(total_tiles), kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(activated.mutable_data_ptr()),
      width,
      tiles_per_row,
      amax_bits.data_ptr<int>());
  nvfp4_quantize_kernel<<<rows, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(activated.const_data_ptr()),
      reinterpret_cast<uint8_t*>(values_bytes.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(scales.mutable_data_ptr()),
      width,
      inner,
      amax_bits.data_ptr<int>(),
      global_scale.data_ptr<float>(),
      true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  out.values = values_bytes.view(at::kFloat4_e2m1fn_x2);
  out.block_scales = scales.view(at::kFloat8_e4m3fn);
  out.global_scale = global_scale;
  return out;
}
} // namespace infinity
