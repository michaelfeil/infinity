# Local modifications to the pinned FlashAttention headers

The files in this directory are copied from
`libs/infinity-v3/backends/libtorch/cpp/vendor/torch_flash_attention` at
commit `2f970a5b83e86608cce7b800db818eccc283e853`. The complete BSD-3-Clause
license and AUTHORS file remain in this directory and in `../../LICENSES/`.

Only these changes were made:

1. `philox_unpack.cuh`: the two ATen includes are wrapped in
   `#ifndef FLASHATTENTION_DISABLE_DROPOUT`.
2. `flash_fwd_launch_template.h`: `#include <c10/cuda/CUDAException.h>` was
   replaced with `#include "head512_cuda_error.h"`, and the
   `C10_CUDA_CHECK` / `C10_CUDA_KERNEL_LAUNCH_CHECK` macro names were replaced
   with `HEAD512_CUDA_CHECK` / `HEAD512_KERNEL_LAUNCH_CHECK`.

No kernel source, tile traits, mask calculations, numeric types, namespace
machinery or license text was changed.
