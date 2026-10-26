// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace infinity {
// Convert adjacent values together, preserving the existing rounded division.
__device__ __forceinline__ __nv_fp8x2_e4m3
fp8_quantize_pair(float low, float high, float scale) {
  return __nv_fp8x2_e4m3(make_float2(__fdiv_rn(low, scale), __fdiv_rn(high, scale)));
}
}
