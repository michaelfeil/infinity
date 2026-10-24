// Native Torch pinned Flash source; vendor/torch_flash_attention retains notices.
// Preserve the fork's fused softmax arithmetic without altering upstream headers.
#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"

namespace tei_torch_flash {
void launch_128(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.is_bf16 && params.is_causal && params.seqlen_q > 1 && params.seqlen_q <= 64 &&
      params.seqlen_q < params.seqlen_k) {
    const auto [major, minor] = get_compute_capability(get_current_device());
    if (major == 9 && minor == 0) {
      // Keep64-key reductions and fused softmax. Multiple32-row query blocks
      // avoid the original128-query tile's padding for short suffixes.
      run_flash_fwd<Flash_fwd_kernel_traits<128, 32, 64, 2, false, false, cutlass::bfloat16_t>,
                    false,
                    true>(params, stream);
      return;
    }
  }
  if (params.is_bf16 && params.is_causal && params.seqlen_q == 1 && params.seqlen_k > 1) {
    const auto [major, minor] = get_compute_capability(get_current_device());
    if (major == 9 && minor == 0) {
      // Keep the original64-key reduction blocks and fused softmax. One warp
      // covers the sole query without executing a128-query tile on SM90.
      run_flash_fwd<Flash_fwd_kernel_traits<128, 16, 64, 1, false, false, cutlass::bfloat16_t>,
                    false,
                    true>(params, stream);
      return;
    }
  }
  if (params.is_bf16) {
    if (params.is_causal)
      run_mha_fwd_hdim128<cutlass::bfloat16_t, true>(params, stream);
    else
      run_mha_fwd_hdim128<cutlass::bfloat16_t, false>(params, stream);
  } else {
    if (params.is_causal)
      run_mha_fwd_hdim128<cutlass::half_t, true>(params, stream);
    else
      run_mha_fwd_hdim128<cutlass::half_t, false>(params, stream);
  }
}
}
