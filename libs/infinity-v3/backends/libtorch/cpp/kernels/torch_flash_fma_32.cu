// Isolated native Torch pinned Flash specialization; upstream notices retained in vendor headers.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash {
void launch_alibi32(Flash_fwd_params& p, cudaStream_t stream) {
  run_mha_fwd_hdim32<cutlass::half_t, false>(p, stream);
}
}
