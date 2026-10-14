// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#pragma once
#include <cuda_bf16.h>
#include <cmath>

namespace tei {
// Round an inexact FP32 FMA to odd before narrowing to BF16. This preserves
// sticky information at BF16 midpoints, avoiding double rounding through FP32.
__device__ __forceinline__ __nv_bfloat16 bf16_fma_emulated(
    __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  const float x = __bfloat162float(a), y = __bfloat162float(b), z = __bfloat162float(c);
  const float truncated = __fmaf_rz(x, y, z);
  unsigned bits = __float_as_uint(truncated);
  if (isfinite(truncated) && __fmaf_rd(x, y, z) != __fmaf_ru(x, y, z))
    bits |= 1u;
  return __float2bfloat16_rn(__uint_as_float(bits));
}
}  // namespace tei

// CUDA supplies these BF16 overloads only for SM80+. Keep its native
// implementations unchanged there, and allow mixed FP16/BF16 translation
// units to compile for Turing as well.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ __nv_bfloat16 __hfma(
    __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) {
  return tei::bf16_fma_emulated(a, b, c);
}
__device__ __forceinline__ __nv_bfloat162 __hfma2(
    __nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
  return __halves2bfloat162(
      __hfma(__low2bfloat16(a), __low2bfloat16(b), __low2bfloat16(c)),
      __hfma(__high2bfloat16(a), __high2bfloat16(b), __high2bfloat16(c)));
}
#endif
