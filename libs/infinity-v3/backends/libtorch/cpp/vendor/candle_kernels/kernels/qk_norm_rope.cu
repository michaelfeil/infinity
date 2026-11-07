// SPDX-License-Identifier: MIT
// Fused Q/K RMSNorm and NeoX RoPE, retaining infinity_v3's intermediate rounding.
// Reuse the existing aligned vector type and preserve the layer-norm reduction order.
#include <functional>
#include "ln_utils.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t qk_pair_bits(__half a, __half b) {
  return uint32_t(__half_as_ushort(a)) | (uint32_t(__half_as_ushort(b)) << 16);
}

__device__ __forceinline__ uint32_t qk_pair_bits(__nv_bfloat16 a, __nv_bfloat16 b) {
  return uint32_t(__bfloat16_as_ushort(a)) | (uint32_t(__bfloat16_as_ushort(b)) << 16);
}

__device__ __forceinline__ void qk_pair_halves(uint32_t bits, __half& a, __half& b) {
  a = __ushort_as_half(uint16_t(bits));
  b = __ushort_as_half(uint16_t(bits >> 16));
}

__device__ __forceinline__ void qk_pair_halves(uint32_t bits, __nv_bfloat16& a,
                                               __nv_bfloat16& b) {
  a = __ushort_as_bfloat16(uint16_t(bits));
  b = __ushort_as_bfloat16(uint16_t(bits >> 16));
}

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
    uint32_t packed[4], peer_bits[4];
#pragma unroll
    for (int q = 0; q < 4; q++)
      packed[q] = qk_pair_bits(vals[2 * q], vals[2 * q + 1]);
#pragma unroll
    for (int q = 0; q < 4; q++)
      peer_bits[q] = __shfl_xor_sync(
          (threadIdx.x % 32) < 16 ? 0x0000ffff : 0xffff0000, packed[q], 8, 16);
    if (lane < 8) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        T peer0, peer1;
        qk_pair_halves(peer_bits[q], peer0, peer1);
        T v0 = vals[2 * q], v1 = vals[2 * q + 1];
        T cv0 = c.data.elt[2 * q], cv1 = c.data.elt[2 * q + 1];
        T sv0 = s.data.elt[2 * q], sv1 = s.data.elt[2 * q + 1];
        a.data.elt[2 * q] = v0 * cv0 - peer0 * sv0;
        a.data.elt[2 * q + 1] = v1 * cv1 - peer1 * sv1;
        b.data.elt[2 * q] = peer0 * cv0 + v0 * sv0;
        b.data.elt[2 * q + 1] = peer1 * cv1 + v1 * sv1;
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
