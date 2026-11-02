/* Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
 * Isolated head_dim=512 packed-attention provider ABI.
 *
 * This is a bounded Native512 bridge derived from the pinned Torch-Flash
 * head512 launch and the isolated FlashAttention 2.8.3 BSD-3-Clause sources in
 * native/torch_flash_attention.  It is not a general FlashAttention-2 backend
 * and it does not change the normal FA4 path.  It has no LibTorch, ATen, c10 or
 * Python runtime dependency.  The only CUDA runtime requirement is the exact
 * cudart recorded in the per-architecture manifest.
 *
 * ABI v2 is deliberately incompatible with v1: all public exports carry a
 * _v2 suffix, v1 symbols are absent, and cold preparation is mandatory.
 */
#ifndef INFINITY_HEAD512_PROVIDER_H
#define INFINITY_HEAD512_PROVIDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFINITY_HEAD512_ABI_VERSION 2
#define INFINITY_HEAD512_HEAD_DIM 512
#define INFINITY_HEAD512_NATIVE_SMEM_BYTES 131072
#define INFINITY_HEAD512_COMPACT_SMEM_BYTES 98304
/* Legacy native-shape alias retained for source compatibility. */
#define INFINITY_HEAD512_DYNAMIC_SMEM_BYTES INFINITY_HEAD512_NATIVE_SMEM_BYTES

/* Explicitly compiled kernel variants reported by infinity_head512_variant_v2. */
enum InfinityHead512KernelVariant {
  INFINITY_HEAD512_VARIANT_NATIVE = 1,
  INFINITY_HEAD512_VARIANT_COMPACT = 2
};

/* dtype: 0 = IEEE FP16, 1 = BF16. */
enum InfinityHead512DType {
  INFINITY_HEAD512_DTYPE_FP16 = 0,
  INFINITY_HEAD512_DTYPE_BF16 = 1
};

/* Bit masks reported by info. */
#define INFINITY_HEAD512_DTYPE_FP16_BIT (1u << 0)
#define INFINITY_HEAD512_DTYPE_BF16_BIT (1u << 1)
#define INFINITY_HEAD512_WINDOW_GLOBAL_BIT (1u << 0)
#define INFINITY_HEAD512_WINDOW_LOCAL_BIT (1u << 1)

typedef struct InfinityHead512Info {
  int32_t abi_version;
  int32_t head_dim;
  uint32_t dtype_mask;
  uint32_t window_mask;
  int64_t dynamic_shared_memory_bytes;
  int64_t min_optin_shared_memory_bytes;
  uint32_t flags; /* bit0 causal, bit1 noncausal, bit2 ragged-qk, bit3 runtime-gqa */
} InfinityHead512Info;

/* Packed [tokens, heads, 512] tensors with caller-owned strides.  The d
 * stride must be one.  q_offsets/k_offsets are device int32 cumulative arrays
 * of batch+1 entries.  Q and KV may have independent boundaries.
 *
 * Unsafe storage contract: Q/K/V must be valid non-overlapping device storage
 * with a >=16-byte-aligned base and row/head strides in whole 8-element vectors.
 * The bridge rejects misaligned Q/K/V before launch.  O row/head strides must
 * describe contiguous or positively padded row-major heads; overlapping output
 * layouts are rejected.  out_bytes/lse_bytes are byte capacities, and the
 * physical O capacity required is
 * ((total_q-1)*o_row + (heads-1)*o_head + 512) * element_bytes, overflow
 * checked.  Callers must keep this library, its single bound context, stream,
 * every input, offset array, output, lse and Work region alive and unmodified
 * until the GPU has completed the enqueued launch AND the provider error state
 * has been drained. */
typedef struct InfinityHead512Batch {
  uint64_t q;
  uint64_t k;
  uint64_t v;
  uint64_t q_offsets;
  uint64_t k_offsets;
  int64_t total_q;
  int64_t total_k;
  int64_t batch;
  int64_t max_q;
  int64_t max_k;
  int64_t heads;
  int64_t kv_heads;
  double scale;
  /* q_row,q_head,q_d, k_row,k_head,k_d, v_row,v_head,v_d, o_row,o_head,o_d */
  int64_t strides[12];
  int32_t dtype;
  int32_t causal;
  int32_t window_left;
  int32_t window_right;
} InfinityHead512Batch;

int32_t infinity_head512_abi_v2(void);
/* Kernel variant compiled into this exact artifact.  No device query. */
int32_t infinity_head512_variant_v2(void);
/* Cold, context-free capability query.  Returns 0 on success. */
int32_t infinity_head512_info_v2(InfinityHead512Info* info);
/* Last provider error for the calling host thread; never null. */
const char* infinity_head512_error_v2(void);

/* Cold preparation for exactly one already-bound CUDA context.  Applies the
 * fixed opt-in dynamic shared-memory attribute to the six compiled kernel
 * specializations (FP16/BF16 x global-noncausal/causal/local-noncausal).  It
 * performs no allocation, GPU launch, synchronization, graph capture or device
 * query.  The caller owns the context lifetime and must not reset it, nor
 * unload this library, until all enqueued work has completed and errors have
 * been drained.  Repeated calls on the same context are idempotent. */
int32_t infinity_head512_prepare_v2(void);

/* Enqueue one launch on the caller's stream.  Output and softmax_lse are
 * caller-owned (outer Work); this ABI performs no allocation, synchronization,
 * device/context query, cudaFuncSetAttribute, graph capture or JIT work.
 * out_bytes is the physical byte capacity of out, lse_bytes of lse.  Local
 * windows are provider-local experimental and are not claimed as
 * Native-qualified.  infinity_head512_prepare_v2 must have completed on the
 * current context first. */
int32_t infinity_head512_enqueue_v2(const InfinityHead512Batch* batch,
                                    void* out,
                                    uint64_t out_bytes,
                                    void* lse,
                                    uint64_t lse_bytes,
                                    void* stream);

#ifdef __cplusplus
}
#endif
#endif
