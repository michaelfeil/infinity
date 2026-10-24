// Default Torch arithmetic from the pinned native Flash source. The unmodified
// vendor/torch_flash_attention headers and BSD-3-Clause notices remain intact.
#ifndef UNFUSE_FMA
#define UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash_unfused
#include "flash_launch_setup.h"
#define cudaFuncSetAttribute infinity::flash_set_attribute
#include "vendor/torch_flash_attention/flash_fwd_launch_template.h"
#undef cudaFuncSetAttribute
#include <cstring>
#include <type_traits>

namespace tei_torch_flash_unfused {
template <typename T> void launch(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.d == 64) {
    if (params.is_causal)
      run_mha_fwd_hdim64<T, true>(params, stream);
    else
      run_mha_fwd_hdim64<T, false>(params, stream);
  } else {
    TORCH_CHECK(params.d == 128, "Unsupported default Torch prefix head dimension");
    if (params.is_causal)
      run_mha_fwd_hdim128<T, true>(params, stream);
    else
      run_mha_fwd_hdim128<T, false>(params, stream);
  }
}

void launch(const void* source, size_t bytes, cudaStream_t stream) {
  // Both namespaces compile the same pinned parameter declaration. Copy its
  // trivially copyable representation without aliasing distinct C++ types.
  static_assert(std::is_trivially_copyable_v<Flash_fwd_params>);
  Flash_fwd_params params;
  TORCH_CHECK(bytes == sizeof(params), "Private Flash parameter layout mismatch");
  std::memcpy(&params, source, bytes);
  if (params.is_bf16)
    launch<cutlass::bfloat16_t>(params, stream);
  else
    launch<cutlass::half_t>(params, stream);
}
}
