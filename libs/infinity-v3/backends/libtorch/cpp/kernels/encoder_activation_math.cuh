// SPDX-License-Identifier: MIT
// GELU arithmetic follows candle-kernels/src/unary.cu from Candle revision
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original permission notice:
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

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>

namespace infinity {
namespace activation_math {
__device__ __forceinline__ half model_fma(half a, half b, half c) {
  return __hfma(a, b, c);
}

__device__ __forceinline__ nv_bfloat16 model_fma(nv_bfloat16 a, nv_bfloat16 b, nv_bfloat16 c) {
  return __hfma(a, b, c);
}

__device__ __forceinline__ float model_fma(float a, float b, float c) {
  return fmaf(a, b, c);
}

template <class T> __device__ __forceinline__ T rounded_gelu(T x) {
  T square = T(x * x), cube = T(square * x);
  T alpha = model_fma(T(0.044715), cube, x);
  T argument = T(T(0.7978845608028654) * alpha);
  T tangent = T(tanhf(float(argument)));
  return T(T(T(0.5) * x) * T(T(1.0) + tangent));
}

template <class T> __device__ __forceinline__ T rounded_silu(T x) {
  // ATen model-dtype add/divide use float opmath with one rounding back to T.
  T exponential = T(expf(float(T(-x))));
  T denominator = T(float(T(1)) + float(exponential));
  return T(float(x) / float(denominator));
}

template <class T> __device__ __forceinline__ T rounded_relu(T x) {
  return float(x) < 0.f ? T(0) : x;
}
}
}
