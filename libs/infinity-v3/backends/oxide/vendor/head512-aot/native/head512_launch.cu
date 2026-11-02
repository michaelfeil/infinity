// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Isolated head_dim=512 launch for the pinned Native Torch-Flash kernel.
// Upstream FlashAttention sources remain BSD-3-Clause; see LICENSES.
//
// This file contains only the bounded host bridge: cold shared-memory
// preparation and a fixed warm kernel selection.  The FlashAttention 2.8.3
// kernel body, softmax, masks, traits and launch template stay byte-identical
// in native/torch_flash_attention.
#ifndef FLASHATTENTION_DISABLE_DROPOUT
#error "head512 provider must compile with FLASHATTENTION_DISABLE_DROPOUT"
#endif
#ifndef FLASHATTENTION_DISABLE_ALIBI
#error "head512 provider must compile with FLASHATTENTION_DISABLE_ALIBI"
#endif
#ifndef FLASHATTENTION_DISABLE_SOFTCAP
#error "head512 provider must compile with FLASHATTENTION_DISABLE_SOFTCAP"
#endif

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE infinity_head512_flash
#include "flash.h"
#include "flash_fwd_launch_template.h"

#include <cuda_runtime_api.h>

#include "head512_launch.h"

namespace {
thread_local int head512_cuda_error = 0;
}

extern "C" void infinity_head512_cuda_note_error(int code) {
  if (head512_cuda_error == 0 && code != 0)
    head512_cuda_error = code;
}

extern "C" int infinity_head512_cuda_take_error(void) {
  int code = head512_cuda_error;
  head512_cuda_error = 0;
  return code;
}

namespace infinity_head512_flash {
// Native is the exact SM90/SM100 M64/N32/W4/128 KiB shape. Compact is the
// explicit M32/N32/W2/96 KiB shape for minimum SM120 support and H100 math
// qualification. Only the warp-M tile changes; the kernel body is untouched.
template <typename Element>
struct Head512Traits {
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  using Type = Flash_fwd_kernel_traits<512, 32, 32, 2, false, false, Element>;
#else
  using Type = Flash_fwd_kernel_traits<512, 64, 32, 4, false, false, Element>;
#endif
};

template <typename Traits>
constexpr void check_head512_traits() {
  static_assert(Traits::kHeadDim == 512, "Wide-head trait must stay at d512");
  static_assert(Traits::kBlockM == 16 * Traits::kNWarps,
                "Wide-head M tile must exactly cover the warp layout");
  static_assert(Traits::kBlockN % 32 == 0,
                "Wide-head score tiles must be 32-row aligned");
#ifdef INFINITY_HEAD512_KERNEL_COMPACT
  static_assert(Traits::kBlockM == 32 && Traits::kBlockN == 32 && Traits::kNWarps == 2,
                "Compact wide-head tile must be M32/N32/W2");
  static_assert(Traits::kSmemSize == 96 * 1024,
                "Compact wide-head shared-memory guard must be 96 KiB");
#else
  static_assert(Traits::kBlockM == 64 && Traits::kBlockN == 32 && Traits::kNWarps == 4,
                "Native wide-head tile must be M64/N32/W4");
  static_assert(Traits::kSmemSize == 128 * 1024,
                "Native wide-head shared-memory guard must be 128 KiB");
#endif
}

// Cold attribute preparation for one exact specialization.  The upstream
// kernel template itself is referenced; no math is copied.
template <typename Element, bool IsCausal, bool IsLocal>
void prepare_fixed() {
  static_assert(!(IsCausal && IsLocal),
                "A kernel specialization cannot be both causal and local");
  using Traits = typename Head512Traits<Element>::Type;
  check_head512_traits<Traits>();
  auto kernel = &flash_fwd_kernel<Traits, false, IsCausal, IsLocal, false, false,
                                  true, false, false>;
  HEAD512_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(Traits::kSmemSize)));
}

// Exactly the valid mask pairs for the fixed d512 contract: noncausal global,
// causal global, noncausal local.  Causal-local is rejected upstream.
template <typename Element>
void prepare_typed() {
  prepare_fixed<Element, false, false>();
  prepare_fixed<Element, true, false>();
  prepare_fixed<Element, false, true>();
}

void prepare_all() {
  prepare_typed<cutlass::half_t>();
  prepare_typed<cutlass::bfloat16_t>();
}

// Warm path: one direct launch of the same upstream kernel template.  The
// only runtime work after validation is cudaGetLastError, which surfaces a
// typed launch failure.
template <typename Element, bool IsCausal, bool IsLocal>
void launch_fixed(Flash_fwd_params& params, cudaStream_t stream) {
  using Traits = typename Head512Traits<Element>::Type;
  check_head512_traits<Traits>();
  const int num_m_block =
      (params.seqlen_q + Traits::kBlockM - 1) / Traits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  auto kernel = &flash_fwd_kernel<Traits, false, IsCausal, IsLocal, false, false,
                                  true, false, false>;
  kernel<<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(params);
  HEAD512_KERNEL_LAUNCH_CHECK();
}

template <typename Element>
void launch_typed(Flash_fwd_params& params, cudaStream_t stream) {
  const bool is_local =
      (params.window_size_left >= 0 || params.window_size_right >= 0) &&
      !params.is_causal;
  if (params.is_causal) {
    launch_fixed<Element, true, false>(params, stream);
  } else if (is_local) {
    launch_fixed<Element, false, true>(params, stream);
  } else {
    launch_fixed<Element, false, false>(params, stream);
  }
}

void launch_512(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.is_bf16)
    launch_typed<cutlass::bfloat16_t>(params, stream);
  else
    launch_typed<cutlass::half_t>(params, stream);
}
}  // namespace infinity_head512_flash

extern "C" int32_t infinity_head512_prepare_raw_v2(void) {
  (void)infinity_head512_cuda_take_error();
  infinity_head512_flash::prepare_all();
  return static_cast<int32_t>(infinity_head512_cuda_take_error());
}

extern "C" int32_t infinity_head512_launch_raw_v2(void* params, void* stream) {
  if (params == nullptr)
    return static_cast<int32_t>(cudaErrorInvalidValue);
  infinity_head512_flash::launch_512(
      *static_cast<infinity_head512_flash::Flash_fwd_params*>(params),
      static_cast<cudaStream_t>(stream));
  return static_cast<int32_t>(infinity_head512_cuda_take_error());
}
