// Native Torch pinned Flash source; vendor/torch_flash_attention retains
// its complete BSD-3-Clause license and AUTHORS notices.
// Proven SM90 BF16 global attention on original head width72. Internal96-wide
// MMA/shared-memory tiles mask params.d=72; no input/output head or token padding.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"
namespace tei_torch_flash {
void launch_72(Flash_fwd_params& params,cudaStream_t stream) {
  using Traits=Flash_fwd_kernel_traits<96,128,64,4,false,false,cutlass::bfloat16_t>;
  static_assert(Traits::kSmemSize==48*1024,"Head72 architecture guard must match actual shared memory");
  run_mha_fwd_hdim96<cutlass::bfloat16_t,false>(params,stream);
}
}
