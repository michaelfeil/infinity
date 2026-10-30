// SPDX-License-Identifier: MIT
// Fused Q/K RMSNorm and NeoX RoPE, retaining infinity_v3's intermediate rounding.
// Reuse the existing aligned vector type and preserve the layer-norm reduction order.
#include <functional>
#include "ln_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template <class T>
__device__ void fused_qk_norm_rope(const T* q,
                                   const T* k,
                                   const T* qw,
                                   const T* kw,
                                   const T* cos,
                                   const T* sin,
                                   T* out,
                                   int tokens,
                                   int qheads,
                                   int kheads,
                                   int qstride,
                                   int kstride,
                                   float eps) {
  // Each 16-thread group handles one 128-element head (eight heads per block).
  const int lane = threadIdx.x % 16, group = threadIdx.x / 16;
  const int qrows = tokens * qheads, rows = qrows + tokens * kheads;
  using V = layer_norm::Vec<T, 8>;
  for (int row = blockIdx.x * 8 + group; row < rows; row += gridDim.x * 8) {
    bool isq = row < qrows;
    int r = isq ? row : row - qrows;
    int token = r / (isq ? qheads : kheads);
    const T* in = isq ? q : k;
    const T* w = isq ? qw : kw;
    V x, g;
    float xf[8] = {};
    float mean = 0;

    const int heads = isq ? qheads : kheads;
    const size_t offset = size_t(token) * (isq ? qstride : kstride) + (r % heads) * 128;
    x.load_from(in, offset / 8 + lane);
    g.load_from(w, lane);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      xf[j] = float(x.data.elt[j]);
      mean += xf[j];
    }

#pragma unroll
    for (int off = 1; off < 16; off *= 2)
      mean += __shfl_xor_sync((threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, mean, off, 16);
    // The original 32-lane reduction adds zero from the unused upper half warp.
    mean += 0.f;
    mean *= 1.f / 128;
    float variance = 0;

#pragma unroll
    for (int j = 0; j < 8; j++) {
      float d = xf[j] - mean;
      variance += d * d;
    }

#pragma unroll
    for (int off = 1; off < 16; off *= 2)
      variance +=
          __shfl_xor_sync((threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, variance, off, 16);
    // Match the baseline: round mean squared before adding it.
    variance += 0.f;
    float inv = rsqrtf(__fadd_rn(fmaf(variance, 1.f / 128, eps), __fmul_rn(mean, mean)));
    // Round normalized values to the model dtype before applying RoPE.
    T vals[8];
#pragma unroll
    for (int j = 0; j < 8; j++)
      vals[j] = T(float(g.data.elt[j]) * (inv * (xf[j] - 0.f)) + 0.f);

    // Lower eight lanes hold the first NeoX half; exchange with the second half.
    V c, s, a, b;
    if (lane < 8) {
      c.load_from(cos, token * 8 + lane);
      s.load_from(sin, token * 8 + lane);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      T peer = T(__shfl_xor_sync(
          (threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, float(vals[j]), 8, 16));
      if (lane < 8) {
        T v = vals[j];
        T cv = c.data.elt[j], sv = s.data.elt[j];
        a.data.elt[j] = v * cv - peer * sv;
        b.data.elt[j] = peer * cv + v * sv;
      }
    }
    if (lane < 8) {
      a.store_to(out, row * 16 + lane);
      b.store_to(out, row * 16 + 8 + lane);
    }
  }
}

extern "C" __global__ void qk_norm_rope_f16(const __half* q,
                                            const __half* k,
                                            const __half* qw,
                                            const __half* kw,
                                            const __half* cos,
                                            const __half* sin,
                                            __half* out,
                                            int tokens,
                                            int qheads,
                                            int kheads,
                                            int qstride,
                                            int kstride,
                                            float eps) {
  fused_qk_norm_rope(q, k, qw, kw, cos, sin, out, tokens, qheads, kheads, qstride, kstride, eps);
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void qk_norm_rope_bf16(const __nv_bfloat16* q,
                                             const __nv_bfloat16* k,
                                             const __nv_bfloat16* qw,
                                             const __nv_bfloat16* kw,
                                             const __nv_bfloat16* cos,
                                             const __nv_bfloat16* sin,
                                             __nv_bfloat16* out,
                                             int tokens,
                                             int qheads,
                                             int kheads,
                                             int qstride,
                                             int kstride,
                                             float eps) {
  fused_qk_norm_rope(q, k, qw, kw, cos, sin, out, tokens, qheads, kheads, qstride, kstride, eps);
}

#endif

// Copy three independently laid-out projections in one launch, preserving all bits.
extern "C" __global__ void qkv_unfold_u16(const uint4* q,
                                          const uint4* k,
                                          const uint4* v,
                                          const uint32_t* ids,
                                          uint4* out,
                                          uint32_t tokens,
                                          uint32_t qwidth,
                                          uint32_t kvwidth,
                                          uint64_t vstride) {
  const uint64_t qsize = uint64_t(tokens) * qwidth;
  const uint64_t kvsize = uint64_t(tokens) * kvwidth;
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < qsize + 2 * kvsize;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const bool isq = i < qsize;
    const bool isk = !isq && i < qsize + kvsize;
    const uint64_t j = isq ? i : (isk ? i - qsize : i - qsize - kvsize);
    const uint32_t width = isq ? qwidth : kvwidth;
    const uint64_t row = ids[j / width];
    const uint4* src = isq ? q : (isk ? k : v);
    out[i] = src[row * (isq ? qwidth : (isk ? kvwidth : vstride)) + j % width];
  }
}
