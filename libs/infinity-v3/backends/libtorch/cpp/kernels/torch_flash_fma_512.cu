// Native Torch pinned Flash source; vendor/torch_flash_attention retains
// its complete BSD-3-Clause license and AUTHORS notices.
// Proven SM90 BF16 global causal/noncausal specialization for actual head width512.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash {
void launch_512(Flash_fwd_params& params, cudaStream_t stream) {
  using Traits = Flash_fwd_kernel_traits<512, 64, 32, 4, false, false, cutlass::bfloat16_t>;
  static_assert(Traits::kSmemSize == 128 * 1024,
                "Wide-head architecture guard must match actual shared memory");
  // No token or head-dimension padding; batch/sequence strides are supplied by
  // the existing packed adapter and the kernel masks true cumulative lengths.
  if (params.is_causal)
    run_flash_fwd<Traits, false, true>(params, stream);
  else
    run_flash_fwd<Traits, false, false>(params, stream);
}
}
