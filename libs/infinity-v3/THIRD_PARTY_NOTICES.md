# Third-party notices

This index identifies existing notices for bundled source components. The
referenced texts and file headers remain authoritative for their respective
components. This is not a complete audit of Cargo dependencies, Python packages,
CUDA libraries, model weights, or container base images.

| Component | Retained license and attribution locations |
| --- | --- |
| Hugging Face Text Embeddings Inference | [LICENSE](LICENSE), [NOTICE](NOTICE), and inherited file headers. |
| Vendored TEI regression-test snapshots | Apache-2.0 comparison baselines; [Candle snapshot provenance](backends/candle/tests/snapshots/NOTICE) and [router snapshot provenance](router/tests/snapshots/NOTICE) list the fixtures, full Git blob hashes, and pinned upstream revisions. |
| Adapted Candle CPU operators | [Apache text](LICENSES/Apache-2.0.txt) and source attribution in [ModernBERT CPU math](backends/libtorch/cpp/helpers/modern_cpu_math.cpp), based on Candle revision `036f39e`. |
| Candle CUDA activation kernel | [MIT text](backends/candle/src/kernels/LICENSE.candle-MIT) and [kernel notices](backends/candle/src/kernels/gated_activation.cu). |
| Candle cuBLASLt extension | [Apache text](backends/candle/extensions/candle-cublaslt/LICENSE-APACHE) and [MIT text](backends/candle/extensions/candle-cublaslt/LICENSE-MIT). |
| Candle index-select extension | [Apache text](backends/candle/extensions/candle-index-select-cu/LICENSE-APACHE), [MIT text](backends/candle/extensions/candle-index-select-cu/LICENSE-MIT), and file headers. |
| Candle layer-norm extension | [Apache text](backends/candle/extensions/candle-layer-norm/LICENSE-APACHE), [MIT text](backends/candle/extensions/candle-layer-norm/LICENSE-MIT), and [BSD text](backends/candle/extensions/candle-layer-norm/LICENSE). |
| Candle rotary extension | [Apache text](backends/candle/extensions/candle-rotary/LICENSE-APACHE) and [MIT text](backends/candle/extensions/candle-rotary/LICENSE-MIT). |
| RadixMLP-derived layers | MIT attribution in [index-select](backends/candle/src/layers/index_select.rs) and [RadixMLP](backends/candle/src/layers/radix_mlp.rs) source headers. |
| CUTLASS-derived Gemma4 and Nomic MoE kernels | [BSD-3-Clause text](backends/candle/extensions/candle-gemma4-moe/LICENSE.cutlass). |
| ALiBi implementation | [Source notices](backends/candle/src/alibi.rs) for Google AI Language Team, Hugging Face, and Jina AI. |

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
