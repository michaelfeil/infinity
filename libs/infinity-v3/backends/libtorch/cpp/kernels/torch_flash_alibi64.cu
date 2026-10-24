// Isolated specialization of pinned native Torch Flash; vendor BSD notices retained.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#ifdef FLASHATTENTION_DISABLE_ALIBI
#undef FLASHATTENTION_DISABLE_ALIBI
#endif
#define FLASH_NAMESPACE tei_torch_flash_alibi64
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash_alibi64 {
void launch_alibi64(Flash_fwd_params& p, cudaStream_t stream) {
  if (p.is_bf16)
    run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(p, stream);
  else
    run_mha_fwd_hdim64<cutlass::half_t, false>(p, stream);
}
}
