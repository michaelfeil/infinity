// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Torch-free C ABI for the isolated head_dim=512 Native Flash launch.
#include "head512_provider.h"
#include "head512_launch.h"

#include <cuda_runtime_api.h>

#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE infinity_head512_flash
#include "flash.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <limits>

namespace {
thread_local char last_error[512] = {0};

void clear_error() { last_error[0] = '\0'; }

void set_error(const char* message) {
  std::snprintf(last_error, sizeof(last_error), "%s", message ? message : "unknown error");
}

void set_error_code(const char* prefix, int32_t code) {
  const char* detail = cudaGetErrorString(static_cast<cudaError_t>(code));
  std::snprintf(last_error, sizeof(last_error), "%s: %s (code %d)",
                prefix ? prefix : "CUDA error", detail ? detail : "unknown", int(code));
}

bool finite_positive(double value) {
  return std::isfinite(value) && value > 0.0 &&
         std::isfinite(static_cast<float>(value)) && static_cast<float>(value) > 0.0f;
}

constexpr uint64_t kVectorBaseAlignmentBytes = 16;
constexpr int64_t kVectorStrideAlignmentElements = 8;

bool checked_pointer_span(uint64_t base, uint64_t last_element, int64_t element_bytes) {
  if (base == 0 || element_bytes <= 0)
    return false;
  __int128 elements = static_cast<__int128>(last_element) + 1;
  __int128 bytes = elements * element_bytes;
  if (bytes <= 0 || bytes > std::numeric_limits<uint64_t>::max())
    return false;
  const uint64_t span = static_cast<uint64_t>(bytes);
  const uint64_t max_address = std::numeric_limits<uintptr_t>::max();
  return base <= max_address && span <= max_address - base;
}

bool checked_input_extent(int64_t tokens, int64_t heads, const int64_t* strides,
                          uint64_t base, int64_t element_bytes,
                          int64_t* last_element) {
  if (tokens <= 0 || heads <= 0 || strides == nullptr)
    return false;
  if (base == 0 || base % kVectorBaseAlignmentBytes != 0)
    return false;
  if (strides[0] <= 0 || strides[1] < INFINITY_HEAD512_HEAD_DIM || strides[2] != 1)
    return false;
  if (strides[0] % kVectorStrideAlignmentElements != 0 ||
      strides[1] % kVectorStrideAlignmentElements != 0)
    return false;
  const __int128 extent = static_cast<__int128>(tokens - 1) * strides[0] +
                          static_cast<__int128>(heads - 1) * strides[1] +
                          static_cast<__int128>(INFINITY_HEAD512_HEAD_DIM - 1) * strides[2];
  if (extent < 0 || extent > std::numeric_limits<int64_t>::max())
    return false;
  *last_element = static_cast<int64_t>(extent);
  return checked_pointer_span(base, static_cast<uint64_t>(*last_element), element_bytes);
}

bool checked_output_extent(int64_t tokens, int64_t heads, const int64_t* strides,
                           uint64_t base, int64_t element_bytes, int64_t* last_element,
                           uint64_t* physical_bytes) {
  if (tokens <= 0 || heads <= 0 || strides == nullptr)
    return false;
  if (strides[0] <= 0 || strides[1] < INFINITY_HEAD512_HEAD_DIM || strides[2] != 1)
    return false;
  if (base == 0 || base % 16 != 0 || strides[0] % 8 != 0 || strides[1] % 8 != 0)
    return false;
  const __int128 minimum_row_stride =
      static_cast<__int128>(heads - 1) * strides[1] + INFINITY_HEAD512_HEAD_DIM;
  if (static_cast<__int128>(strides[0]) < minimum_row_stride)
    return false;
  const __int128 extent = static_cast<__int128>(tokens - 1) * strides[0] +
                          static_cast<__int128>(heads - 1) * strides[1] +
                          static_cast<__int128>(INFINITY_HEAD512_HEAD_DIM - 1) * strides[2];
  if (extent < 0 || extent > std::numeric_limits<int64_t>::max())
    return false;
  *last_element = static_cast<int64_t>(extent);
  const __int128 bytes =
      (static_cast<__int128>(*last_element) + 1) * element_bytes;
  if (bytes <= 0 || bytes > std::numeric_limits<uint64_t>::max())
    return false;
  *physical_bytes = static_cast<uint64_t>(bytes);
  return checked_pointer_span(base, static_cast<uint64_t>(*last_element), element_bytes);
}

bool checked_bytes(int64_t tokens, int64_t heads, int64_t width, int64_t element_bytes,
                   uint64_t capacity) {
  if (tokens <= 0 || heads <= 0 || width <= 0 || element_bytes <= 0)
    return false;
  __int128 bytes = static_cast<__int128>(tokens) * heads * width * element_bytes;
  return bytes > 0 && static_cast<unsigned __int128>(bytes) <= capacity;
}

int32_t bind_and_enqueue(const InfinityHead512Batch* batch,
                         void* out,
                         uint64_t out_bytes,
                         void* lse,
                         uint64_t lse_bytes,
                         void* stream) {
  if (batch == nullptr || out == nullptr || lse == nullptr ||
      batch->q == 0 || batch->k == 0 || batch->v == 0 ||
      batch->q_offsets == 0 || batch->k_offsets == 0) {
    set_error("missing head512 pointer or batch");
    return -1;
  }
  if (reinterpret_cast<uintptr_t>(lse) % 4 != 0 || batch->q_offsets % 4 != 0 || batch->k_offsets % 4 != 0) {
    set_error("head512 LSE/offset pointers require four-byte alignment");
    return -1;
  }
  if (batch->dtype != INFINITY_HEAD512_DTYPE_FP16 &&
      batch->dtype != INFINITY_HEAD512_DTYPE_BF16) {
    set_error("unsupported head512 dtype");
    return -1;
  }
  if (batch->causal != 0 && batch->causal != 1) {
    set_error("head512 causal must be boolean");
    return -1;
  }
  if (batch->window_left < -1 || batch->window_right < -1) {
    set_error("head512 windows must be -1 or nonnegative");
    return -1;
  }
  if (batch->total_q <= 0 || batch->total_k <= 0 || batch->batch <= 0 ||
      batch->max_q <= 0 || batch->max_k <= 0 || batch->heads <= 0 ||
      batch->kv_heads <= 0) {
    set_error("head512 geometry must be positive");
    return -1;
  }
  if (batch->batch > batch->total_q || batch->batch > batch->total_k ||
      batch->max_q > batch->total_q || batch->max_k > batch->total_k) {
    set_error("head512 token counts are inconsistent");
    return -1;
  }
  if (batch->heads % batch->kv_heads != 0) {
    set_error("head512 heads must be divisible by kv_heads");
    return -1;
  }
  if (!finite_positive(batch->scale)) {
    set_error("head512 scale must be finite and positive");
    return -1;
  }
  if (batch->total_q > std::numeric_limits<int>::max() ||
      batch->total_k > std::numeric_limits<int>::max() ||
      batch->batch > std::numeric_limits<int>::max() ||
      batch->heads > std::numeric_limits<int>::max() ||
      batch->kv_heads > std::numeric_limits<int>::max() ||
      batch->max_q > std::numeric_limits<int>::max() - 127 ||
      batch->max_k > std::numeric_limits<int>::max() - 127) {
    set_error("head512 geometry exceeds int32 kernel bounds");
    return -1;
  }
  if (!checked_bytes(batch->total_q, batch->heads, 1, 4, lse_bytes)) {
    set_error("head512 lse capacity is too small");
    return -1;
  }

  int64_t q_last = 0, k_last = 0, v_last = 0, o_last = 0;
  if (!checked_input_extent(batch->total_q, batch->heads, batch->strides + 0, batch->q,
                            2, &q_last) ||
      !checked_input_extent(batch->total_k, batch->kv_heads, batch->strides + 3, batch->k,
                            2, &k_last) ||
      !checked_input_extent(batch->total_k, batch->kv_heads, batch->strides + 6, batch->v,
                            2, &v_last)) {
    set_error("head512 q/k/v storage is misaligned or its stride extent overflows");
    return -1;
  }
  uint64_t output_bytes = 0;
  if (!checked_output_extent(batch->total_q, batch->heads, batch->strides + 9,
                             reinterpret_cast<uint64_t>(out), 2, &o_last,
                             &output_bytes) ||
      output_bytes > out_bytes) {
    set_error("head512 output layout overlaps, wraps, or exceeds its physical capacity");
    return -1;
  }
  (void)q_last;
  (void)k_last;
  (void)v_last;
  (void)o_last;

  int64_t left = batch->window_left;
  int64_t right = batch->window_right;
  if (left > batch->max_k)
    left = -1;
  if (right > batch->max_k)
    right = -1;
  if (batch->causal)
    right = 0;
  const bool is_causal = left < 0 && right == 0;
  if (left < 0 && right >= 0)
    left = batch->max_k;
  if (left >= 0 && right < 0)
    right = batch->max_k;

  infinity_head512_flash::Flash_fwd_params params{};
  params.q_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(batch->q));
  params.k_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(batch->k));
  params.v_ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(batch->v));
  params.o_ptr = out;
  params.softmax_lse_ptr = lse;
  params.q_row_stride = batch->strides[0];
  params.q_head_stride = batch->strides[1];
  params.k_row_stride = batch->strides[3];
  params.k_head_stride = batch->strides[4];
  params.v_row_stride = batch->strides[6];
  params.v_head_stride = batch->strides[7];
  params.o_row_stride = batch->strides[9];
  params.o_head_stride = batch->strides[10];
  params.b = static_cast<int>(batch->batch);
  params.h = static_cast<int>(batch->heads);
  params.h_k = static_cast<int>(batch->kv_heads);
  params.h_h_k_ratio = params.h / params.h_k;
  params.seqlen_q = static_cast<int>(batch->max_q);
  params.seqlen_k = static_cast<int>(batch->max_k);
  params.seqlen_q_rounded = static_cast<int>(((batch->max_q + 127) / 128) * 128);
  params.seqlen_k_rounded = static_cast<int>(((batch->max_k + 127) / 128) * 128);
  params.d = INFINITY_HEAD512_HEAD_DIM;
  params.d_rounded = INFINITY_HEAD512_HEAD_DIM;
  params.total_q = static_cast<int>(batch->total_q);
  params.cu_seqlens_q =
      reinterpret_cast<int*>(static_cast<uintptr_t>(batch->q_offsets));
  params.cu_seqlens_k =
      reinterpret_cast<int*>(static_cast<uintptr_t>(batch->k_offsets));
  params.unpadded_lse = true;
  params.is_seqlens_k_cumulative = true;
  params.scale_softmax = static_cast<float>(batch->scale);
  // Match Native exactly: F32 scale times the double M_LOG2E, then F32 round.
  params.scale_softmax_log2 = static_cast<float>(
      static_cast<double>(params.scale_softmax) * M_LOG2E);
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;
  params.scale_softmax_rp_dropout = static_cast<float>(batch->scale);
  params.window_size_left = static_cast<int>(left);
  params.window_size_right = static_cast<int>(right);
  params.is_bf16 = batch->dtype == INFINITY_HEAD512_DTYPE_BF16;
  params.is_causal = is_causal;
  params.num_splits = 1;

  const int32_t status = infinity_head512_launch_raw_v2(&params, stream);
  if (status != 0) {
    set_error_code("head512 launch failed", status);
    return -1;
  }
  clear_error();
  return 0;
}
}  // namespace

extern "C" int32_t infinity_head512_abi_v2(void) {
  return INFINITY_HEAD512_ABI_VERSION;
}

extern "C" int32_t infinity_head512_variant_v2(void) {
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  return INFINITY_HEAD512_VARIANT_COMPACT;
#else
  return INFINITY_HEAD512_VARIANT_NATIVE;
#endif
}

extern "C" int32_t infinity_head512_info_v2(InfinityHead512Info* info) {
  if (info == nullptr) {
    set_error("missing head512 info pointer");
    return -1;
  }
  info->abi_version = INFINITY_HEAD512_ABI_VERSION;
  info->head_dim = INFINITY_HEAD512_HEAD_DIM;
  info->dtype_mask =
      INFINITY_HEAD512_DTYPE_FP16_BIT | INFINITY_HEAD512_DTYPE_BF16_BIT;
  info->window_mask =
      INFINITY_HEAD512_WINDOW_GLOBAL_BIT | INFINITY_HEAD512_WINDOW_LOCAL_BIT;
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  info->dynamic_shared_memory_bytes = INFINITY_HEAD512_COMPACT_SMEM_BYTES;
  info->min_optin_shared_memory_bytes = INFINITY_HEAD512_COMPACT_SMEM_BYTES;
#else
  info->dynamic_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
  info->min_optin_shared_memory_bytes = INFINITY_HEAD512_NATIVE_SMEM_BYTES;
#endif
  info->flags = 0x0f;  // causal, noncausal, ragged Q/KV, runtime GQA
  clear_error();
  return 0;
}

extern "C" const char* infinity_head512_error_v2(void) {
  return last_error;
}

extern "C" int32_t infinity_head512_prepare_v2(void) {
  const int32_t status = infinity_head512_prepare_raw_v2();
  if (status != 0) {
    set_error_code("head512 cold prepare failed", status);
    return -1;
  }
  clear_error();
  return 0;
}

extern "C" int32_t infinity_head512_enqueue_v2(const InfinityHead512Batch* batch,
                                               void* out,
                                               uint64_t out_bytes,
                                               void* lse,
                                               uint64_t lse_bytes,
                                               void* stream) {
  return bind_and_enqueue(batch, out, out_bytes, lse, lse_bytes, stream);
}
