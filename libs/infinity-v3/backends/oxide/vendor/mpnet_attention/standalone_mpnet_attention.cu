/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Standalone adapter for PyTorch's native memory-efficient attention forward
 * kernel. It is derived from:
 *   aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h
 *   aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h
 *   aten/src/ATen/native/transformers/cuda/attention.cu
 * at PyTorch commit 5c4886908584029761b579af026dcfb627c84070.
 *
 * The included CUTLASS headers remain under NVIDIA's CUTLASS license. The
 * PyTorch LICENSE file must be distributed with this file.
 *
 * This file does NOT link torch, c10, torch_cpu or torch_cuda. It expects:
 *   - the complete public mem_eff_attention header tree,
 *   - ATen/cuda/detail/{PhiloxCudaStateRaw,UnpackRaw}.cuh,
 *   - c10/util/Exception.h (parse-only here; check_supported is not called),
 *   - exact public CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3,
 *   - kernel_forward.h patched with the bucket-bias extension from
 *     PATCH_PLAN.md (compile-time gate below).
 *
 * Call mpnet_attn_prepare_bf16_h12d64(plan) once for the current CUDA
 * context/device before launching. The launch path never sets or queries
 * function attributes, allocates, or synchronizes.
 *
 * No compile, link, run or numerical validation is claimed by this file.
 */

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>

/*
 * The stock public kernel only loads a materialized [B, H, M, N] bias.
 * PATCH_PLAN.md adds Params::bias_table_ptr and friends plus the on-the-fly
 * bucket add. Refuse to compile a silent fallback.
 */
#ifndef MPNET_ATTENTION_HAVE_BUCKET_BIAS
#error "standalone_mpnet_attention.cu requires the PATCH_PLAN.md bucket-bias extension: define MPNET_ATTENTION_HAVE_BUCKET_BIAS after patching kernel_forward.h"
#endif

#include <cutlass/bfloat16.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>

#define MPNET_ATTN_ABI_VERSION 1u
#define MPNET_ATTN_DTYPE_BF16 1u

typedef enum mpnet_attn_status {
  MPNET_ATTN_OK = 0,
  MPNET_ATTN_EINVAL = 1,      /* host-side geometry rejected */
  MPNET_ATTN_EUNSUPPORTED = 2,/* geometry valid but not this adapter */
  MPNET_ATTN_ECUDA = 3,       /* CUDA attribute/launch error */
  MPNET_ATTN_EUNPREPARED = 4  /* missing/invalid caller-owned prepared plan */
} mpnet_attn_status;

/*
 * Bounded typed geometry. All pointers are borrowed. The adapter never owns
 * memory, never creates a stream, and never synchronizes.
 *
 * Element types are BF16 for qkv, bias_table and out. seqstart_q/seqstart_k
 * are int32 device arrays with num_batches + 1 entries.
 */
typedef struct mpnet_attn_geometry {
  uint32_t abi_version;       /* MPNET_ATTN_ABI_VERSION */
  uint32_t dtype;             /* MPNET_ATTN_DTYPE_BF16 */
  uint32_t num_heads;         /* must be 12 */
  uint32_t head_dim;          /* must be 64 */
  uint32_t qkv_row_stride;    /* elements; must be 3*H*d = 2304 */
  uint32_t q_offset;          /* elements; must be 0 */
  uint32_t k_offset;          /* elements; must be H*d = 768 */
  uint32_t v_offset;          /* elements; must be 2*H*d = 1536 */
  uint32_t num_batches;       /* >= 1, bounded */
  uint32_t max_seqlen_q;      /* >= 1, <= 512 */
  uint32_t max_seqlen_k;      /* >= 1, <= 512 */
  uint32_t bias_num_buckets;  /* must be 32 */
  uint32_t bias_num_heads;    /* must equal num_heads */
  uint32_t bias_bucket_stride;/* elements between buckets */
  uint32_t bias_head_stride;  /* elements between heads */
  uint32_t out_row_stride;    /* elements; >= H*d */
  uint32_t flags;             /* must be 0 */
  uint64_t qkv_num_rows;      /* total packed tokens */
  uint64_t out_num_rows;      /* must equal qkv_num_rows */

  const void* qkv;            /* BF16 [qkv_num_rows, qkv_row_stride] */
  const int32_t* seqstart_q;  /* [num_batches + 1] */
  const int32_t* seqstart_k;  /* [num_batches + 1] */
  const void* bias_table;     /* BF16 [32, 12] or strided equivalent */
  void* out;                  /* BF16 [out_num_rows, out_row_stride] */

  cudaStream_t stream;        /* caller's existing stream; may be null */
} mpnet_attn_geometry;

// Caller-owned cold state; no process-global context/readiness cache. The safe
// owner must retain this library/context and keep that context current for each
// launch, just as for its borrowed device pointers and stream.
typedef struct mpnet_attn_prepared {
  uint32_t abi_version;
  uint32_t struct_bytes;
  uint64_t shared_memory_bytes;
  CUcontext context;
} mpnet_attn_prepared;

namespace {

using mpnet_scalar_t = cutlass::bfloat16_t;

using MpnetKernel = PyTorchMemEffAttention::AttentionKernel<
    mpnet_scalar_t,         /* scalar_t */
    cutlass::arch::Sm80,    /* ArchTag, selected by public bf16 dispatch */
    true,                   /* isAligned */
    64,                     /* kQueriesPerBlock */
    64,                     /* kKeysPerBlock */
    64,                     /* kMaxK */
    false,                  /* kSupportsDropout */
    true>;                  /* kSupportsBias: required to keep scale placement */

static_assert(MpnetKernel::kSupportsDropout == false, "");
static_assert(MpnetKernel::kSupportsBias == true, "");
static_assert(MpnetKernel::kMaxK == 64, "");
static_assert(MpnetKernel::kSingleValueIteration, "");
static_assert(MpnetKernel::kKeepOutputInRF, "");
static_assert(!MpnetKernel::kNeedsOutputAccumulatorBuffer, "");
static_assert(MpnetKernel::kNumThreads == 128, "");



/*
 * Mirrors attention_kernel_batched_impl at the end of kernel_forward.h.
 * The public fmha_cutlassF_* entry points are declarations only in the staged
 * headers, so the adapter supplies its own definition.
 */
template <typename AK>
__global__ void __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
mpnet_attn_kernel(typename AK::Params p) {
  if (!p.advance_to_block()) {
    return;
  }
  AK::attention_kernel(p);
}

bool ptr_aligned_to(const void* ptr, int64_t alignment_elems) {
  const uint64_t alignment_bytes =
      uint64_t(alignment_elems) * sizeof(mpnet_scalar_t);
  return (reinterpret_cast<uintptr_t>(ptr) % alignment_bytes) == 0;
}

int validate_geometry(const mpnet_attn_geometry* g) {
  if (g == nullptr) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->abi_version != MPNET_ATTN_ABI_VERSION ||
      g->dtype != MPNET_ATTN_DTYPE_BF16 || g->flags != 0u) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->num_heads != 12u || g->head_dim != 64u) {
    return MPNET_ATTN_EUNSUPPORTED;
  }
  if (g->qkv_row_stride != 3u * g->num_heads * g->head_dim) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->q_offset != 0u ||
      g->k_offset != g->num_heads * g->head_dim ||
      g->v_offset != 2u * g->num_heads * g->head_dim) {
    return MPNET_ATTN_EINVAL;
  }
  /* grid.z is num_batches; CUDA gridDim.z max is 65535. */
  if (g->num_batches == 0u || g->num_batches > 65535u) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->max_seqlen_q == 0u || g->max_seqlen_q > 512u ||
      g->max_seqlen_k == 0u || g->max_seqlen_k > 512u) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->bias_num_buckets != 32u || g->bias_num_heads != g->num_heads) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->bias_bucket_stride < 1u || g->bias_head_stride < 1u) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->out_row_stride < g->num_heads * g->head_dim ||
      g->out_row_stride > uint32_t(std::numeric_limits<int32_t>::max())) {
    return MPNET_ATTN_EINVAL;
  }
  if (g->qkv_num_rows == 0u || g->qkv_num_rows != g->out_num_rows ||
      g->qkv_num_rows > uint64_t(std::numeric_limits<int32_t>::max())) {
    return MPNET_ATTN_EINVAL;
  }
  {
    const uint64_t last_bias_elem =
        uint64_t(g->bias_num_buckets - 1u) * g->bias_bucket_stride +
        uint64_t(g->bias_num_heads - 1u) * g->bias_head_stride + 1u;
    if (last_bias_elem > uint64_t(std::numeric_limits<int32_t>::max())) {
      return MPNET_ATTN_EINVAL;
    }
  }
  if (g->qkv == nullptr || g->seqstart_q == nullptr ||
      g->seqstart_k == nullptr || g->bias_table == nullptr ||
      g->out == nullptr) {
    return MPNET_ATTN_EINVAL;
  }

  /* Native alignment invariants, mirrored without TORCH_CHECK. */
  if ((g->qkv_row_stride % MpnetKernel::kAlignmentQ) != 0u ||
      (g->qkv_row_stride % MpnetKernel::kAlignmentK) != 0u ||
      (g->qkv_row_stride % MpnetKernel::kAlignmentV) != 0u ||
      (g->head_dim % MpnetKernel::kAlignmentQ) != 0u ||
      (g->head_dim % MpnetKernel::kAlignmentK) != 0u) {
    return MPNET_ATTN_EUNSUPPORTED;
  }
  if (!ptr_aligned_to(g->qkv, MpnetKernel::kAlignmentQ) ||
      !ptr_aligned_to(static_cast<const char*>(g->qkv) +
                          uint64_t(g->k_offset) * sizeof(mpnet_scalar_t),
                      MpnetKernel::kAlignmentK) ||
      !ptr_aligned_to(static_cast<const char*>(g->qkv) +
                          uint64_t(g->v_offset) * sizeof(mpnet_scalar_t),
                      MpnetKernel::kAlignmentV) ||
      !ptr_aligned_to(g->out, MpnetKernel::kAlignmentQ)) {
    return MPNET_ATTN_EUNSUPPORTED;
  }
  if ((reinterpret_cast<uintptr_t>(g->bias_table) % sizeof(mpnet_scalar_t)) !=
      0u) {
    return MPNET_ATTN_EUNSUPPORTED;
  }
  if ((g->out_row_stride % MpnetKernel::kAlignmentQ) != 0u) {
    return MPNET_ATTN_EUNSUPPORTED;
  }
  return MPNET_ATTN_OK;
}

}  // namespace

extern "C" {

size_t mpnet_attn_smem_bytes(void) {
  return sizeof(MpnetKernel::SharedStorage);
}

/*
 * Cold-path setup for the current CUDA context/device. Call once (or again
 * after switching devices/contexts) before mpnet_attn_launch_bf16_h12d64.
 * This is the only place that touches the function shared-memory attribute.
 */
int mpnet_attn_prepare_bf16_h12d64(mpnet_attn_prepared* prepared) {
  if (prepared == nullptr) return MPNET_ATTN_EINVAL;
  *prepared = {};
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr)
    return MPNET_ATTN_ECUDA;
  const size_t smem_bytes = sizeof(MpnetKernel::SharedStorage);
  if (smem_bytes > size_t(std::numeric_limits<int>::max()))
    return MPNET_ATTN_EUNSUPPORTED;
  if (smem_bytes > 48u * 1024u) {
    const cudaError_t status = cudaFuncSetAttribute(
        (const void*)mpnet_attn_kernel<MpnetKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    if (status != cudaSuccess) return MPNET_ATTN_ECUDA;
  }
  prepared->abi_version = MPNET_ATTN_ABI_VERSION;
  prepared->struct_bytes = sizeof(mpnet_attn_prepared);
  prepared->shared_memory_bytes = smem_bytes;
  prepared->context = context;
  return MPNET_ATTN_OK;
}

int mpnet_attn_validate(const mpnet_attn_geometry* geometry) {
  return validate_geometry(geometry);
}

/*
 * Returns MPNET_ATTN_OK when the launch was issued. The kernel is asynchronous
 * on geometry->stream; this function does not synchronize and does not check
 * asynchronous execution errors.
 *
 * Host-side contract that cannot be checked without reading device memory:
 *   seqstart_q[0] == 0, seqstart_q[num_batches] == qkv_num_rows,
 *   seqstart_k[0] == 0, seqstart_k[num_batches] == qkv_num_rows,
 *   both arrays are nondecreasing, every per-batch length is in [1, 512],
 *   and seqstart_q[b+1] - seqstart_q[b] == seqstart_k[b+1] - seqstart_k[b]
 *   for every b (the packed QKV row stores Q, K and V for the same token).
 *   Empty per-batch sequences are not handled by this adapter. Violating
 *   this contract is an out-of-bounds read or a degenerate softmax.
 * mpnet_attn_prepare_bf16_h12d64(prepared) must have succeeded in the current CUDA
 * context/device. The caller must make the stream's CUDA device current before
 * the call and it must equal prepared->context; the adapter does not call
 * cudaSetDevice or query context/attributes on the launch path. A safe owner
 * must enforce the same context identity for the plan, buffers and stream.
 */
int mpnet_attn_launch_bf16_h12d64(
    const mpnet_attn_prepared* prepared, const mpnet_attn_geometry* geometry) {
  const int status = validate_geometry(geometry);
  if (status != MPNET_ATTN_OK) {
    return status;
  }
  if (prepared == nullptr || prepared->abi_version != MPNET_ATTN_ABI_VERSION ||
      prepared->struct_bytes != sizeof(mpnet_attn_prepared) ||
      prepared->shared_memory_bytes != sizeof(MpnetKernel::SharedStorage) ||
      prepared->context == nullptr) {
    return MPNET_ATTN_EUNPREPARED;
  }

  const auto* qkv = static_cast<const mpnet_scalar_t*>(geometry->qkv);
  const auto* table = static_cast<const mpnet_scalar_t*>(geometry->bias_table);
  auto* out = static_cast<mpnet_scalar_t*>(geometry->out);

  MpnetKernel::Params p;

  p.query_ptr = qkv + geometry->q_offset;
  p.key_ptr = qkv + geometry->k_offset;
  p.value_ptr = qkv + geometry->v_offset;
  p.seqstart_q_ptr = geometry->seqstart_q;
  p.seqstart_k_ptr = geometry->seqstart_k;
  p.seqlen_k_ptr = nullptr;

  p.output_ptr = out;
  p.output_accum_ptr = nullptr;  /* kNeedsOutputAccumulatorBuffer == false */
  p.logsumexp_ptr = nullptr;

  /* PATCH_PLAN.md fields. attn_bias_ptr stays null: no materialized bias. */
  p.bias_table_ptr = table;
  p.bias_num_buckets = static_cast<int32_t>(geometry->bias_num_buckets);
  p.bias_num_heads = static_cast<int32_t>(geometry->bias_num_heads);
  p.bias_bucket_stride = static_cast<int32_t>(geometry->bias_bucket_stride);
  p.bias_head_stride = static_cast<int32_t>(geometry->bias_head_stride);
  p.attn_bias_ptr = nullptr;

  p.window_size = 0;
  p.scale = 0.125f;  /* fixed qk scale required by TASK.txt */
  p.head_dim = 64;
  p.head_dim_value = 64;
  p.num_queries = static_cast<int32_t>(geometry->max_seqlen_q);
  p.num_keys = static_cast<int32_t>(geometry->max_seqlen_k);
  p.num_keys_absolute = static_cast<int32_t>(geometry->max_seqlen_k);
  p.custom_mask_type = MpnetKernel::NoCustomMask;

  p.q_strideM = static_cast<int32_t>(geometry->qkv_row_stride);
  p.k_strideM = static_cast<int32_t>(geometry->qkv_row_stride);
  p.v_strideM = static_cast<int32_t>(geometry->qkv_row_stride);
  p.o_strideM = static_cast<int32_t>(geometry->out_row_stride);
  p.q_strideH = 64;
  p.k_strideH = 64;
  p.v_strideH = 64;

  /* Unused on the varlen path (seqstart_q_ptr != nullptr). */
  p.q_strideB = 0;
  p.k_strideB = 0;
  p.v_strideB = 0;

  p.num_batches = static_cast<int32_t>(geometry->num_batches);
  p.num_heads = 12;
  p.q_heads_per_kv = 1;

  p.use_dropout = false;
  p.dropout_prob = 0.0f;
  p.dropout_batch_head_rng_offset = 0;
  p.extragraph_offset = nullptr;
  p.seed = nullptr;

  const size_t smem_bytes = sizeof(MpnetKernel::SharedStorage);

  const dim3 blocks = p.getBlocksGrid();
  const dim3 threads = p.getThreadsGrid();
  if (blocks.x == 0u || blocks.y == 0u || blocks.z == 0u) {
    return MPNET_ATTN_OK;
  }

  mpnet_attn_kernel<MpnetKernel>
      <<<blocks, threads, smem_bytes, geometry->stream>>>(p);

  const cudaError_t launch_err = cudaGetLastError();
  return launch_err == cudaSuccess ? MPNET_ATTN_OK : MPNET_ATTN_ECUDA;
}

}  // extern "C"
