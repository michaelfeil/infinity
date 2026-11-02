/* Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
 * Test-only C ABI stub used to exercise the Rust cold-prepare/cold-load/enqueue
 * contract without CUDA. It never performs a kernel launch, context query or
 * attribute update and is never part of a production bundle. */
#include "head512_provider.h"

static _Thread_local char stub_error[1] = {0};
static int stub_prepare_calls = 0;
static int stub_enqueue_calls = 0;

#ifdef INFINITY_HEAD512_STUB_ABI_V1
/* Deliberately old ABI: only v1 exports exist. */
int32_t infinity_head512_abi_v1(void) { return 1; }

int32_t infinity_head512_variant_v1(void) {
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  return INFINITY_HEAD512_VARIANT_COMPACT;
#else
  return INFINITY_HEAD512_VARIANT_NATIVE;
#endif
}

int32_t infinity_head512_info_v1(InfinityHead512Info* info) {
  if (info == 0)
    return -1;
  info->abi_version = 1;
  info->head_dim = INFINITY_HEAD512_HEAD_DIM;
  info->dtype_mask = INFINITY_HEAD512_DTYPE_FP16_BIT | INFINITY_HEAD512_DTYPE_BF16_BIT;
  info->window_mask = INFINITY_HEAD512_WINDOW_GLOBAL_BIT | INFINITY_HEAD512_WINDOW_LOCAL_BIT;
  info->dynamic_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
  info->min_optin_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
  info->flags = 0x0f;
  return 0;
}

const char* infinity_head512_error_v1(void) { return stub_error; }

int32_t infinity_head512_enqueue_v1(const InfinityHead512Batch* batch,
                                    void* out,
                                    uint64_t out_bytes,
                                    void* lse,
                                    uint64_t lse_bytes,
                                    void* stream) {
  (void)batch;
  (void)out;
  (void)stream;
  if (out_bytes == 0 || lse_bytes == 0)
    return -1;
  return 0;
}

#else

int32_t infinity_head512_abi_v2(void) { return INFINITY_HEAD512_ABI_VERSION; }

int32_t infinity_head512_variant_v2(void) {
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  return INFINITY_HEAD512_VARIANT_COMPACT;
#else
  return INFINITY_HEAD512_VARIANT_NATIVE;
#endif
}

int32_t infinity_head512_info_v2(InfinityHead512Info* info) {
  if (info == 0)
    return -1;
  info->abi_version = INFINITY_HEAD512_ABI_VERSION;
  info->head_dim = INFINITY_HEAD512_HEAD_DIM;
  info->dtype_mask = INFINITY_HEAD512_DTYPE_FP16_BIT | INFINITY_HEAD512_DTYPE_BF16_BIT;
  info->window_mask = INFINITY_HEAD512_WINDOW_GLOBAL_BIT | INFINITY_HEAD512_WINDOW_LOCAL_BIT;
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  info->dynamic_shared_memory_bytes = INFINITY_HEAD512_COMPACT_SMEM_BYTES;
  info->min_optin_shared_memory_bytes = INFINITY_HEAD512_COMPACT_SMEM_BYTES;
#else
  info->dynamic_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
  info->min_optin_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
#endif
  info->flags = 0x0f;
  return 0;
}

const char* infinity_head512_error_v2(void) { return stub_error; }

#ifndef INFINITY_HEAD512_STUB_NO_PREPARE
int32_t infinity_head512_prepare_v2(void) {
  ++stub_prepare_calls;
#ifdef INFINITY_HEAD512_STUB_PREPARE_FAIL
  return -1;
#else
  return 0;
#endif
}
#endif

int32_t infinity_head512_enqueue_v2(const InfinityHead512Batch* batch,
                                    void* out,
                                    uint64_t out_bytes,
                                    void* lse,
                                    uint64_t lse_bytes,
                                    void* stream) {
  (void)batch;
  (void)out;
  (void)stream;
  ++stub_enqueue_calls;
  if (out_bytes == 0 || lse_bytes == 0)
    return -1;
  return 0;
}

int infinity_head512_stub_prepare_calls(void) { return stub_prepare_calls; }

int infinity_head512_stub_enqueue_calls(void) { return stub_enqueue_calls; }

#endif
