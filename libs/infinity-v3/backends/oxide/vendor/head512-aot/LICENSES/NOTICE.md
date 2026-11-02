# Notices for infinity-head512-aot

The isolated CUDA launch in `native/head512_launch.cu` and the copied headers
under `native/torch_flash_attention/` form an isolated FlashAttention
2.8.3-derived Native512 bridge. They are derived from the Native Torch-Flash
head512 specialization in this repository, which in turn vendors
FlashAttention 2.8.3 (submodule revision
`14c377950125c70b7a9dabf9c561fca53715ac7d` from the Torch revision recorded
in `upstream-provenance.json`). This bridge is not a general FlashAttention-2
backend, and the normal FA4 path is unchanged.

FlashAttention is Copyright (c) 2022 the respective contributors, as shown by
`AUTHORS`, and is distributed under the BSD 3-Clause license in
`BSD-3-Clause.txt`. The original license and AUTHORS file are retained verbatim.

Local changes to the copied upstream headers are limited to:
- `philox_unpack.cuh`: ATen includes are guarded by
  `FLASHATTENTION_DISABLE_DROPOUT`.
- `flash_fwd_launch_template.h`: the LibTorch `c10` CUDA exception shim is
  replaced by `head512_cuda_error.h`, and `C10_CUDA_CHECK` /
  `C10_CUDA_KERNEL_LAUNCH_CHECK` are renamed to the provider's `HEAD512_*`
  equivalents. Kernel math, tile traits, namespace structure and licenses are
  unchanged.

The provider wrapper, ABI-v2 cold-preparation bridge, fixed six-specialization
warm launcher, build script, Rust interface, manifests and tests are original
Infinity contributions covered by the repository LICENSE. The warm launcher
references the upstream kernel template directly and does not copy or modify
kernel math, softmax, masks or traits. No
LibTorch, ATen, c10, Python or CuTe-DSL runtime is linked by the provider
library. The exact cudart dependency is recorded in every generated manifest.
