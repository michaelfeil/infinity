// Native Torch pinned Flash source; vendor/torch_flash_attention retains notices.
// Preserve the fork's fused softmax arithmetic without altering upstream headers.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "flash_launch_setup.h"
#define cudaFuncSetAttribute tei::flash_set_attribute
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"
#undef cudaFuncSetAttribute

namespace tei_torch_flash {
void launch_64(Flash_fwd_params& params, cudaStream_t stream) {
  // Short sequences and small local windows waste query work in a 128-row
  // tile. Retain 128 key columns and the existing softmax accumulation order.
  if (params.is_bf16 && !params.is_causal && params.total_q >= 4096 && params.seqlen_q <= 512 &&
      params.seqlen_k <= 512 &&
      ((params.window_size_left >= 0 && params.window_size_left <= 64 &&
        params.window_size_right >= 0 && params.window_size_right <= 64) ||
       (params.window_size_left == -1 && params.window_size_right == -1))) {
    run_flash_fwd<Flash_fwd_kernel_traits<64, 64, 128, 4, false, false, cutlass::bfloat16_t>,
                  false,
                  false>(params, stream);
    return;
  }
  if (params.is_bf16) {
    if (params.is_causal)
      run_mha_fwd_hdim64<cutlass::bfloat16_t, true>(params, stream);
    else
      run_mha_fwd_hdim64<cutlass::bfloat16_t, false>(params, stream);
  } else {
    if (params.is_causal)
      run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
    else
      run_mha_fwd_hdim64<cutlass::half_t, false>(params, stream);
  }
}
}
