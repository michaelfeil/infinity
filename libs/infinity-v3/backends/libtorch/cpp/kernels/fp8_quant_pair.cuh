// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace infinity {
// Convert adjacent values together, preserving the existing rounded division.
__device__ __forceinline__ __nv_fp8x2_e4m3 fp8_quantize_pair(float low, float high, float scale) {
  return __nv_fp8x2_e4m3(make_float2(__fdiv_rn(low, scale), __fdiv_rn(high, scale)));
}

// Reuse the correctly rounded reciprocal with an FMA quotient/remainder correction. Callers
// use a finite row scale derived from the same model-dtype values, with a 1e-12 maximum floor.
// Quotients large enough to reach e4m3 are normal and their remainders are representable;
// smaller quotients round to FP8 zero. Restore the numerator's sign to preserve negative zero.
__device__ __forceinline__ __nv_fp8x2_e4m3 fp8_quantize_pair_recip(float low,
                                                                   float high,
                                                                   float scale,
                                                                   float reciprocal) {
  const auto quotient = [scale, reciprocal](float value) {
    const float approximate = __fmaf_rn(value, reciprocal, 0.0f);
    const float remainder = __fmaf_rn(-scale, approximate, value);
    return copysignf(__fmaf_rn(remainder, reciprocal, approximate), value);
  };
  return __nv_fp8x2_e4m3(make_float2(quotient(low), quotient(high)));
}

// Single-value form for the scalar row fallback. Same precondition as the pair form.
__device__ __forceinline__ __nv_fp8_e4m3 fp8_quantize_value_recip(float value,
                                                                  float scale,
                                                                  float reciprocal) {
  const float approximate = __fmaf_rn(value, reciprocal, 0.0f);
  const float remainder = __fmaf_rn(-scale, approximate, value);
  return __nv_fp8_e4m3(copysignf(__fmaf_rn(remainder, reciprocal, approximate), value));
}
}
