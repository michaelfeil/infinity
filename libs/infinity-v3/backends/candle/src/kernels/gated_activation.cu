// SPDX-License-Identifier: MIT
// Packed gated activations preserving the rounding of Candle's separate operations.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

// Match Candle unary.cu, including every intermediate half-precision rounding.
// Computing the activation in FP32 followed by a final cast is not equivalent.
template <typename T, bool GELU>
__device__ __forceinline__ T precise_activation(T x) {
    if constexpr (GELU) {
        const T x_sq = x * x;
        const T x_cube = x_sq * x;
        const T alpha = x + static_cast<T>(0.044715) * x_cube;
        const T tanh_input = static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha;
        const T tanh_value = static_cast<T>(tanhf(static_cast<float>(tanh_input)));
        return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanh_value);
    } else {
        const T exp_value = hexp(-x);
        const T denominator = T(1) + exp_value;
        return x / denominator;
    }
}

template <typename T>
__device__ __forceinline__ void packed_swiglu(
    const T* input, T* output, uint32_t width) {
    const size_t row = blockIdx.x;
    const T* gate = input + row * 2 * width;
    const T* up = gate + width;
    T* out = output + row * width;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
        const T x = gate[col];
        const T activation = precise_activation<T, false>(x);
        out[col] = activation * up[col];
    }
}

extern "C" __global__ void packed_swiglu_f16(
    const __half* input, __half* output, uint32_t width) {
    packed_swiglu(input, output, width);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void packed_swiglu_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t width) {
    packed_swiglu(input, output, width);
}
#endif

// Matches candle-kernels unary.cu::gelu_fwd<T> and cuda_utils.cuh::tanhg.
// Keep each half-precision intermediate and the same multiplication order.
template <typename T>
__device__ __forceinline__ void packed_geglu(
    const T* input, T* output, uint32_t width) {
    const size_t row = blockIdx.x;
    const T* value = input + row * 2 * width;
    const T* gate = value + width;
    T* out = output + row * width;
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
        const T x = value[col];
        const T activation = precise_activation<T, true>(x);
        out[col] = activation * gate[col];
    }
}

extern "C" __global__ void packed_geglu_f16(
    const __half* input, __half* output, uint32_t width) {
    packed_geglu(input, output, width);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void packed_geglu_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t width) {
    packed_geglu(input, output, width);
}
#endif

// Vectorized packed layout adapted from mistral.rs fused_split_glu_kernel_vec4:
// https://github.com/EricLBuehler/mistral.rs/blob/2370966bb91e2e3dafa0b1521b87c50fd5c01244/mistralrs-quant/kernels/ops/ops.cu
// Arithmetic is changed to preserve the pinned Candle implementation's rounding.
/*
MIT License

Copyright (c) 2024 Eric Buehler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

template <typename T, bool GELU>
__device__ __forceinline__ void packed_glu_vec4(
    const uint64_t* input, uint64_t* output, uint32_t width_vecs, uint64_t output_vecs) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_vecs) return;
    const uint64_t row = index / width_vecs;
    const uint32_t column = index - row * width_vecs;
    const uint64_t value_index = row * 2 * width_vecs + column;
    uint64_t value = input[value_index];
    uint64_t gate = input[value_index + width_vecs];
    uint64_t result;
#pragma unroll
    for (int element = 0; element < 4; ++element) {
        const T activated = precise_activation<T, GELU>(reinterpret_cast<T*>(&value)[element]);
        reinterpret_cast<T*>(&result)[element] = activated * reinterpret_cast<T*>(&gate)[element];
    }
    output[index] = result;
}
extern "C" __global__ void packed_swiglu_vec4_f16(
    const uint64_t* input, uint64_t* output, uint32_t width_vecs, uint64_t output_vecs) {
    packed_glu_vec4<__half, false>(input, output, width_vecs, output_vecs);
}
extern "C" __global__ void packed_geglu_vec4_f16(
    const uint64_t* input, uint64_t* output, uint32_t width_vecs, uint64_t output_vecs) {
    packed_glu_vec4<__half, true>(input, output, width_vecs, output_vecs);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void packed_swiglu_vec4_bf16(
    const uint64_t* input, uint64_t* output, uint32_t width_vecs, uint64_t output_vecs) {
    packed_glu_vec4<__nv_bfloat16, false>(input, output, width_vecs, output_vecs);
}
extern "C" __global__ void packed_geglu_vec4_bf16(
    const uint64_t* input, uint64_t* output, uint32_t width_vecs, uint64_t output_vecs) {
    packed_glu_vec4<__nv_bfloat16, true>(input, output, width_vecs, output_vecs);
}
#endif
