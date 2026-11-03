# Third-party notices

This index identifies existing notices for bundled source components. The
referenced texts and file headers remain authoritative for their respective
components. This is not a complete audit of Cargo dependencies, Python packages,
CUDA libraries, model weights, or container base images.

| Component | Retained license and attribution locations |
| --- | --- |
| fastsafetensors non-GDS reader | Apache-2.0; [license](backends/libtorch/cpp/vendor/fastsafetensors/LICENSE) and [pinned source/adaptation provenance](backends/libtorch/cpp/vendor/fastsafetensors/provenance.json). Only the native worker reader is bundled; no Python, GDS, libaio, io_uring or NUMA dependency. |
| Hugging Face Text Embeddings Inference | [LICENSE](LICENSE), [NOTICE](NOTICE), and inherited file headers. |
| Vendored TEI regression-test snapshots | Apache-2.0 comparison baselines; [router snapshot provenance](router/tests/snapshots/NOTICE) list the fixtures, full Git blob hashes, and pinned upstream revisions. |
| Adapted Candle CPU operators | [Apache text](LICENSES/Apache-2.0.txt) and source attribution in [ModernBERT CPU math](backends/libtorch/cpp/helpers/modern_cpu_math.cpp), based on Candle revision `036f39e`. |
| Candle CUDA activation kernel | [MIT text](backends/libtorch/cpp/vendor/candle_kernels/kernels/LICENSE.candle-MIT) and [kernel notices](backends/libtorch/cpp/kernels/fast_kernels.cu). |
| Native prefix Q/K normalization and rotary kernel | Derived from [Candle Q/K normalization](backends/libtorch/cpp/vendor/candle_kernels/kernels/qk_norm_rope.cu), with the MIT permission notice retained in [the derived kernel](backends/libtorch/cpp/kernels/prefix_qkv_norm_rope.cuh). |
| Candle-derived native FP8 arithmetic | [Apache text](backends/libtorch/cpp/vendor/candle_kernels/notices/LICENSE-APACHE) and [MIT text](backends/libtorch/cpp/vendor/candle_kernels/LICENSE.cublaslt-MIT). |
| Candle layer-norm extension | [Apache text](backends/libtorch/cpp/vendor/candle_kernels/layer_norm/LICENSE-APACHE), [MIT text](backends/libtorch/cpp/vendor/candle_kernels/layer_norm/LICENSE-MIT), and [BSD text](backends/libtorch/cpp/vendor/candle_kernels/layer_norm/LICENSE). |
| RadixMLP-derived layers | MIT attribution retained in [native RadixMLP](backends/libtorch/cpp/batching/radix.cpp). |
| CUTLASS-derived Gemma4 and Nomic MoE kernels | [BSD-3-Clause text](backends/libtorch/cpp/vendor/candle_kernels/moe/LICENSE.cutlass). |
| xxHash prefix fingerprints | [BSD-2-Clause text](backends/libtorch/cpp/vendor/xxhash/LICENSE), upstream header notices, and [pinned source provenance](backends/libtorch/cpp/vendor/xxhash/provenance.json). |

The native Torch [embedding normalization kernel](backends/libtorch/cpp/kernels/encoder_embedding_norm.cuh)
derives from the bundled Candle layer-norm extension's
`kernels/ln_fwd_kernels.cuh`. It retains the BSD-3-Clause copyright and license
header; the extension's BSD license text is also installed with the native library.

Basetenkenizer is a Cargo dependency rather than vendored source. Its published
package carries Apache-2.0 licensing and additional component notices; consult
its [license audit](https://github.com/basetenlabs/basetenkenizer/blob/main/THIRD_PARTY_LICENSES.md)
for the version used by the release. Other dependencies must likewise be audited
from their actual pinned versions when assembling distributed artifacts.

The native Nomic eight-expert router in `backends/libtorch/cpp/kernels/fast_kernels.cu`
derives its strict bitonic ordering from Candle `sort.cu` at revision
`036f39e44015e78ba18d716d73dfdc2e78730c44`, adapted upstream from llama.cpp.
The selected MIT permission notice is retained beside the derived kernel.

Mistral3/Pixtral model equations and processor behavior are adapted from
Hugging Face Transformers 5.12.1 (Apache-2.0). The retained license and source
attribution are in `backends/libtorch/cpp/vendor/transformers_pixtral/`.

The packed CUDA Gated DeltaNet chunkwise transform is adapted from
Hugging Face Transformers 5.19.0 (Apache-2.0). The retained license, source hash,
and adaptation notice are in `backends/libtorch/cpp/vendor/transformers_gdn/`.

Hopper Gated DeltaNet prefill uses FlashInfer 0.6.13 at revision
`57ba7eeb7ea3003a2d6ad5d9a057c4f952709bac` (Apache-2.0). Its LICENSE, NOTICE,
and third-party license texts are installed under `share/licenses/tei_torch/flashinfer/`.
Its device launcher is locally patched to initialize shared-memory opt-in on each CUDA device/context.
