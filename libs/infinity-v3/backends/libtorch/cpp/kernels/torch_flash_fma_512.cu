// Native Torch pinned Flash source; vendor/torch_flash_attention retains
// its complete BSD-3-Clause license and AUTHORS notices.
// Global causal/noncausal specialization for actual head width512.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash {
template <typename Element> void launch_512_typed(Flash_fwd_params& params, cudaStream_t stream) {
  using Traits = Flash_fwd_kernel_traits<512, 64, 32, 4, false, false, Element>;
  static_assert(Traits::kBlockM >= 16 * Traits::kNWarps && Traits::kBlockN % 32 == 0,
                "Wide-head tiles must cover the warp and score layouts");
  static_assert(Traits::kSmemSize == 128 * 1024, "Wide-head guard must match actual shared memory");
  if (params.is_causal)
    run_flash_fwd<Traits, false, true>(params, stream);
  else
    run_flash_fwd<Traits, false, false>(params, stream);
}

void launch_512(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.is_bf16)
    launch_512_typed<cutlass::bfloat16_t>(params, stream);
  else
    launch_512_typed<cutlass::half_t>(params, stream);
}
}
