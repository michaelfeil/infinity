// SPDX-License-Identifier: MIT
// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//
// Minimal additive CUDA module for the Qwen3.5 Gated DeltaNet AOT provider
// boundary. This module is compile-only in this task: it is not wired into any
// executor, factory, admission policy, or model route, and it does not alter
// the eight-export raw module in ../native_gdn/.
//
// Layout sources (read-only, Infinity SPDX MIT, revision c4c90b9):
//   kernels/qwen35_kernels.cu: pack_gdn_inputs
//   kernels/flashinfer_gdn.cpp: provider alpha/beta and [B,H,V,K] state
//   models/qwen35_models.cpp: native [B,H,K,V] state and source gather
//
// Pack ABI (bit-exact Native element mapping; no new arithmetic):
//   qk          F32  [tokens*key_heads, 256]: Q[0:128], K[128:256] per row
//   conv        BF16 [tokens, 2*key_heads*128 + value_heads*128]: V tail
//   prepared_ab F32  [tokens, 2*value_heads]
//   Outputs:
//   q, k        BF16 [tokens, key_heads, 128]
//   v           BF16 [tokens, value_heads, 128]
//   alpha, beta F32  [tokens, value_heads]
//
// The Native prepared A/B source is NOT an interleaved [token*head, 2] array.
// Its actual layout is alpha in [0, value_heads) followed by beta in
// [value_heads, 2*value_heads), per token (prepared_ab_offset below). This
// module deliberately reads that source layout; do not reinterpret it as
// adjacent (alpha, beta) pairs.
//
// State ABI (exact uint32 bit copies of the FP32 payloads):
//   native KV   [history, value_heads, key=128, value=128]  (key outer)
//   provider VK [batch, value_heads, value=128, key=128]    (value outer)
// kv_to_vk optionally gathers rows through an int64 source map and preserves
// duplicate and permuted source selections. vk_to_kv is sequence-indexed and
// never gathers. A null source map means identity.
//
// Every extent below is int64_t. Host code must overflow-check all products,
// strides, and launch-grid extents before use. The kernels guard per-element
// bounds and block tails; no host-to-device copy, allocation, or device
// synchronization occurs here.

#include <stdint.h>

#if defined(INFINITY_GDN_AOT_INDEX_ONLY)
#define INFINITY_GDN_LAYOUT inline
#else
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define INFINITY_GDN_LAYOUT __device__ __forceinline__
#endif

namespace infinity_gdn_aot {

// Width of every native GDN key and value head.
constexpr int64_t kHeadWidth = 128;
// State transpose tile edge.
constexpr int64_t kTile = 32;

// Packed native Q/K row: row = token * key_heads + head, Q then K.
INFINITY_GDN_LAYOUT int64_t packed_qk_offset(int64_t row, int64_t half, int64_t column,
                                             int64_t qk_stride) {
  return row * qk_stride + half * kHeadWidth + column;
}

// Packed native convolution row value tail at 2*key_heads*128.
INFINITY_GDN_LAYOUT int64_t packed_conv_value_offset(int64_t token, int64_t value_head,
                                                     int64_t value_column, int64_t key_heads,
                                                     int64_t conv_stride) {
  return token * conv_stride + 2 * key_heads * kHeadWidth +
         value_head * kHeadWidth + value_column;
}

// Actual Native prepared A/B layout: gate 0 is alpha, gate 1 is beta, stored
// as two halves of the token row, not as adjacent (alpha, beta) pairs.
INFINITY_GDN_LAYOUT int64_t prepared_ab_offset(int64_t token, int64_t head, int64_t gate,
                                               int64_t value_heads, int64_t ab_stride) {
  return token * ab_stride + gate * value_heads + head;
}

// Contiguous pack outputs.
INFINITY_GDN_LAYOUT int64_t packed_q_offset(int64_t token, int64_t head, int64_t column,
                                            int64_t key_heads) {
  return (token * key_heads + head) * kHeadWidth + column;
}

INFINITY_GDN_LAYOUT int64_t packed_v_offset(int64_t token, int64_t head, int64_t column,
                                            int64_t value_heads) {
  return (token * value_heads + head) * kHeadWidth + column;
}

INFINITY_GDN_LAYOUT int64_t packed_gate_offset(int64_t token, int64_t head,
                                               int64_t value_heads) {
  return token * value_heads + head;
}

// Native resident state: [source, value_head, key, value], value innermost.
INFINITY_GDN_LAYOUT int64_t native_kv_offset(int64_t source, int64_t head, int64_t key,
                                             int64_t value, int64_t value_heads,
                                             int64_t key_dim, int64_t value_dim) {
  return ((source * value_heads + head) * key_dim + key) * value_dim + value;
}

// Provider state: [batch, value_head, value, key], key innermost.
INFINITY_GDN_LAYOUT int64_t provider_vk_offset(int64_t batch, int64_t head, int64_t value,
                                               int64_t key, int64_t value_heads,
                                               int64_t key_dim, int64_t value_dim) {
  return ((batch * value_heads + head) * value_dim + value) * key_dim + key;
}

// Number of 32x32 transpose tiles for one batch/head set. A zero extent yields
// no work, which makes batch zero a valid no-op.
INFINITY_GDN_LAYOUT int64_t state_tile_count(int64_t batch, int64_t value_heads,
                                             int64_t key_dim, int64_t value_dim) {
  if (batch <= 0 || value_heads <= 0 || key_dim <= 0 || value_dim <= 0)
    return 0;
  return batch * value_heads * ((value_dim + kTile - 1) / kTile) *
         ((key_dim + kTile - 1) / kTile);
}

}  // namespace infinity_gdn_aot

#ifndef INFINITY_GDN_AOT_INDEX_ONLY

using infinity_gdn_aot::native_kv_offset;
using infinity_gdn_aot::packed_conv_value_offset;
using infinity_gdn_aot::packed_gate_offset;
using infinity_gdn_aot::packed_q_offset;
using infinity_gdn_aot::packed_qk_offset;
using infinity_gdn_aot::packed_v_offset;
using infinity_gdn_aot::prepared_ab_offset;
using infinity_gdn_aot::provider_vk_offset;

// Native pack geometry reproduced exactly from pack_gdn_inputs: one token per
// blockIdx.x, columns split across blockIdx.y/threadIdx.x. Each thread checks
// its own bound, so a ragged value width is handled without padding.
extern "C" __global__ void infinity_qwen35_gdn_aot_pack_bf16(
    const float* qk,
    const __nv_bfloat16* conv,
    const float* prepared_ab,
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    __nv_bfloat16* v,
    float* alpha,
    float* beta,
    int key_heads,
    int value_heads,
    int64_t qk_stride,
    int64_t conv_stride,
    int64_t ab_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int64_t column = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  const int64_t key_width = static_cast<int64_t>(key_heads) * infinity_gdn_aot::kHeadWidth;
  const int64_t value_width =
      static_cast<int64_t>(value_heads) * infinity_gdn_aot::kHeadWidth;
  const int64_t key_heads_wide = key_heads;
  const int64_t value_heads_wide = value_heads;
  if (column < key_width) {
    const int64_t head = column / infinity_gdn_aot::kHeadWidth;
    const int64_t element = column - head * infinity_gdn_aot::kHeadWidth;
    const int64_t row = token * key_heads_wide + head;
    const int64_t output = packed_q_offset(token, head, element, key_heads_wide);
    q[output] = __float2bfloat16_rn(qk[packed_qk_offset(row, 0, element, qk_stride)]);
    k[output] = __float2bfloat16_rn(qk[packed_qk_offset(row, 1, element, qk_stride)]);
  }
  if (column < value_width) {
    const int64_t head = column / infinity_gdn_aot::kHeadWidth;
    const int64_t element = column - head * infinity_gdn_aot::kHeadWidth;
    v[packed_v_offset(token, head, element, value_heads_wide)] =
        conv[packed_conv_value_offset(token, head, element, key_heads_wide, conv_stride)];
  }
  if (column < value_heads_wide) {
    const int64_t head = column;
    const int64_t output = packed_gate_offset(token, head, value_heads_wide);
    alpha[output] =
        prepared_ab[prepared_ab_offset(token, head, 0, value_heads_wide, ab_stride)];
    beta[output] =
        prepared_ab[prepared_ab_offset(token, head, 1, value_heads_wide, ab_stride)];
  }
}

// Shared 32x32 tiled transpose. The load side always uses threadIdx.x as the
// source-contiguous axis and the store side uses threadIdx.x as the
// destination-contiguous axis, so both global accesses are coalesced:
//   ToProvider: native KV (value contiguous) -> provider VK (key contiguous)
//   else:       provider VK (key contiguous) -> native KV (value contiguous)
template <bool ToProvider>
__device__ __forceinline__ void state_transpose_tile(
    const uint32_t* source,
    uint32_t* destination,
    const int64_t* source_indices,
    int64_t batch,
    int64_t source_count,
    int64_t value_heads,
    int64_t key_dim,
    int64_t value_dim) {
  __shared__ uint32_t tile[infinity_gdn_aot::kTile][infinity_gdn_aot::kTile + 1];
  const int64_t tiles_value =
      (value_dim + infinity_gdn_aot::kTile - 1) / infinity_gdn_aot::kTile;
  const int64_t tiles_key = (key_dim + infinity_gdn_aot::kTile - 1) / infinity_gdn_aot::kTile;
  const int64_t tiles_per_head = tiles_value * tiles_key;
  const int64_t total_tiles =
      infinity_gdn_aot::state_tile_count(batch, value_heads, key_dim, value_dim);
  if (tiles_per_head <= 0)
    return;
  const int64_t tile_index = static_cast<int64_t>(blockIdx.x);
  if (tile_index >= total_tiles)
    return;
  const int64_t head_row = tile_index / tiles_per_head;
  const int64_t remainder = tile_index - head_row * tiles_per_head;
  const int64_t tile_value = remainder % tiles_value;
  const int64_t tile_key = remainder / tiles_value;
  const int64_t batch_index = head_row / value_heads;
  const int64_t head = head_row - batch_index * value_heads;
  int64_t sequence = batch_index;
  if constexpr (ToProvider) {
    sequence = source_indices ? source_indices[batch_index] : batch_index;
    if (sequence < 0 || sequence >= source_count)
      return;
  }
  if constexpr (ToProvider) {
    const int64_t load_key = tile_key * infinity_gdn_aot::kTile + threadIdx.y;
    const int64_t load_value = tile_value * infinity_gdn_aot::kTile + threadIdx.x;
    if (load_key < key_dim && load_value < value_dim)
      tile[threadIdx.y][threadIdx.x] = source[native_kv_offset(
          sequence, head, load_key, load_value, value_heads, key_dim, value_dim)];
  } else {
    const int64_t load_value = tile_value * infinity_gdn_aot::kTile + threadIdx.y;
    const int64_t load_key = tile_key * infinity_gdn_aot::kTile + threadIdx.x;
    if (load_key < key_dim && load_value < value_dim)
      tile[threadIdx.y][threadIdx.x] = source[provider_vk_offset(
          batch_index, head, load_value, load_key, value_heads, key_dim, value_dim)];
  }
  __syncthreads();
  if constexpr (ToProvider) {
    const int64_t store_key = tile_key * infinity_gdn_aot::kTile + threadIdx.x;
    const int64_t store_value = tile_value * infinity_gdn_aot::kTile + threadIdx.y;
    if (store_key < key_dim && store_value < value_dim)
      destination[provider_vk_offset(batch_index, head, store_value, store_key, value_heads,
                                     key_dim, value_dim)] = tile[threadIdx.x][threadIdx.y];
  } else {
    const int64_t store_value = tile_value * infinity_gdn_aot::kTile + threadIdx.x;
    const int64_t store_key = tile_key * infinity_gdn_aot::kTile + threadIdx.y;
    if (store_value < value_dim && store_key < key_dim)
      destination[native_kv_offset(batch_index, head, store_key, store_value, value_heads,
                                   key_dim, value_dim)] = tile[threadIdx.x][threadIdx.y];
  }
}

// Gather native KV rows into provider VK. source_indices is optional int64
// [batch]; null means identity. Duplicate and permuted source rows are valid.
extern "C" __global__ void infinity_qwen35_gdn_aot_state_kv_to_vk(
    const uint32_t* source,
    uint32_t* destination,
    const int64_t* source_indices,
    int64_t batch,
    int64_t source_count,
    int64_t value_heads,
    int64_t key_dim,
    int64_t value_dim) {
  state_transpose_tile<true>(source, destination, source_indices, batch, source_count,
                             value_heads, key_dim, value_dim);
}

// Store provider VK back to native KV by sequence position. No source gather.
extern "C" __global__ void infinity_qwen35_gdn_aot_state_vk_to_kv(
    const uint32_t* source,
    uint32_t* destination,
    int64_t batch,
    int64_t value_heads,
    int64_t key_dim,
    int64_t value_dim) {
  state_transpose_tile<false>(source, destination, nullptr, batch, batch, value_heads,
                              key_dim, value_dim);
}

#endif  // INFINITY_GDN_AOT_INDEX_ONLY
