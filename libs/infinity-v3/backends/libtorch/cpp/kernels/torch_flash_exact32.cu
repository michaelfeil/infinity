// Pinned native Torch Flash specialization; upstream vendor notices are retained.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash_exact32
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash_exact32 {
void launch(Flash_fwd_params& p, cudaStream_t stream) {
  if (p.is_bf16)
    run_mha_fwd_hdim32<cutlass::bfloat16_t, false>(p, stream);
  else
    run_mha_fwd_hdim32<cutlass::half_t, false>(p, stream);
}
}
