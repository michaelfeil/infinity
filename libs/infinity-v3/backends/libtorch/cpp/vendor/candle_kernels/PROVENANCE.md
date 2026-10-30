# Retained native CUDA kernels

Moved unchanged from Infinity's removed Rust Candle backend. These kernels
remain part of the LibTorch implementation; no Rust Candle runtime is bundled.
The source revision before relocation is `d9073968443369d57cdb378db2e0ff957c5ec027`.

- `layer_norm/`: `backends/candle/extensions/candle-layer-norm/kernels/`.
- `moe/`: `backends/candle/extensions/candle-gemma4-moe/kernels/`.
- `kernels/`: corresponding files under `backends/candle/src/kernels/`.
- `pooling/mean_pool.cu`: `backends/candle/src/pooling_kernels/mean_pool.cu`.

Original file headers and MIT, Apache-2.0, BSD-3-Clause, and CUTLASS notices are
retained. Only the relative includes in the two grouped-GEMM files changed.
`LICENSE.cublaslt-MIT` and `notices/` retain notices for native FP8 derivatives.
