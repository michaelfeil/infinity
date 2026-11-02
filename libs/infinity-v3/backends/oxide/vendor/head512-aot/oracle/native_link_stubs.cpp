// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only link stubs: the Native adapter declares every head-width launch,
// while this bounded oracle compiles only the actual Native head512 unit.
#include <cstddef>

#include <cuda_runtime_api.h>

#ifdef UNFUSE_FMA
#undef UNFUSE_FMA
#endif
#define FLASH_NAMESPACE tei_torch_flash
#include "vendor/torch_flash_attention/flash.h"

namespace tei_torch_flash {
void launch_alibi32(Flash_fwd_params&, cudaStream_t) {}
void launch_72(Flash_fwd_params&, cudaStream_t) {}
void launch_64(Flash_fwd_params&, cudaStream_t) {}
void launch_128(Flash_fwd_params&, cudaStream_t) {}
void launch_256(Flash_fwd_params&, cudaStream_t) {}
}  // namespace tei_torch_flash

namespace tei_torch_flash_unfused {
void launch(const void*, size_t, cudaStream_t) {}
}  // namespace tei_torch_flash_unfused
