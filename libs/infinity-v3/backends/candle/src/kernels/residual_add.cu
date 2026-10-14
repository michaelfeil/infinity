// SPDX-License-Identifier: MIT
// Preserve scalar half/BF16 addition while processing four elements per thread.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template <typename T> struct PackedType;
template <> struct PackedType<__half> {
    using Pair = __half2;
};
template <> struct PackedType<__nv_bfloat16> {
    using Pair = __nv_bfloat162;
};

template <typename T>
__device__ void residual_add_vec4(const T *lhs, const T *rhs, T *output, uint64_t vectors) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= vectors)
        return;
    using Pair = typename PackedType<T>::Pair;
    union Vec {
        uint2 raw;
        Pair pair[2];
    };
    Vec a, b, result;
    a.raw = reinterpret_cast<const uint2 *>(lhs)[i];
    b.raw = reinterpret_cast<const uint2 *>(rhs)[i];
    result.pair[0] = __hadd2(a.pair[0], b.pair[0]);
    result.pair[1] = __hadd2(a.pair[1], b.pair[1]);
    reinterpret_cast<uint2 *>(output)[i] = result.raw;
}

extern "C" __global__ void residual_add_vec4_f16(const __half *lhs, const __half *rhs,
                                                 __half *output, uint64_t vectors) {
    residual_add_vec4(lhs, rhs, output, vectors);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void residual_add_vec4_bf16(const __nv_bfloat16 *lhs,
                                                  const __nv_bfloat16 *rhs, __nv_bfloat16 *output,
                                                  uint64_t vectors) {
    residual_add_vec4(lhs, rhs, output, vectors);
}
#endif
