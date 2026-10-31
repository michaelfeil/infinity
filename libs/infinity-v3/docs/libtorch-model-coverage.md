# Experimental native LibTorch model coverage

For named checkpoints, launch settings and request examples, use the
[supported model catalog](supported-models.md).

This lists implemented model paths, not a general accuracy or performance
qualification.
All inference preserves actual token counts; CUDA uses packed attention or
exact-length biased attention without padded token activations.

| Model family / aliases | Native implementation |
| --- | --- |
| BERT | Packed encoder, fused QKV/GELU, pooling, classifier and SPLADE heads |
| ColBERTv2 / Answer.AI ColBERT Small | Packed BERT, learned token projection, L2 normalization and multi-vector output |
| RoBERTa, XLM-RoBERTa, CamemBERT | BERT path with checkpoint prefixes and matching heads |
| DistilBERT | Packed encoder, SPLADE projection |
| GTE / New | Packed rotary encoder, gated MLP and classifier |
| ModernBERT | Packed local/global rotary attention, gated MLP and heads |
| UltraBERT | Bidirectional packed GQA, per-layer RoPE bases and symmetric windows, embedding and sequence/token classification heads |
| Nomic BERT | Packed rotary encoder, dense/MoE MLP |
| Jina BERT / JinaCode | Packed encoder, one Torch efficient varlen ALiBi call per layer |
| Llama / Llama bidirectional | Packed GQA, RoPE, RMSNorm, fused projections, sequence/token classification heads |
| Mistral / Ministral3 | Packed GQA, sliding windows, YaRN |
| NVIDIA Nemotron 3 / 3.5 Embed 1B / 8B | Bidirectional Ministral3, model-dtype mean pooling; 3.5 adds Pixtral images and native retrieval prefixes; [usage and parity checks](nemotron-embeddings.md) |
| Mistral3 / Ministral 3 multimodal | Pixtral vision, 2D rotary positions, spatial patch merger/projector, packed Ministral text |
| Mistral3ForSequenceClassification (Nemotron 3.5 Rerank) | Mistral3 backbone with a mean-pooled, bias-free score head; query/document chat template |
| Qwen2 | Packed GQA with projection biases, sequence/token classification heads |
| Perplexity `pplx-embed-v1-0.6b` / `pplx-embed-v1-4b` | Bidirectional Qwen3, mean pooling; floating-point output only |
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection; sequence heads and dense Qwen3 token classification |
| Gemma3 text | Packed local/global attention, Gemma norm semantics, linear full-attention RoPE scaling; sentence or last-token pooling |
| NVIDIA Nemotron ColEmbed V2 4B / 8B | Bidirectional Qwen3-VL, pre-final-norm token states, normalized multi-vector text/image outputs |
| ColQwen3 (TomoroAI 4B) | Qwen3-VL backbone, learned 320-wide token projection, normalized multi-vector text/image outputs |
| EmbeddingGemma2 | Packed text model, shared KV, PLE, embedding projection and image/audio towers |
| Gemma4 / Gemma4 unified | Packed text plus image tower implementation |
| Qwen3-VL | Native vision tower, mRoPE, patch/deepstack mergers and decoder injection |
| Qwen3.5 dense/MoE/text aliases | Packed full attention, DeltaNet, MoE/shared experts and vision tower |
| MPNet | Native exact-length relative-biased attention |
| DeBERTa-v2/v3 (experimental) | Native disentangled relative attention, convolution and classifier |

## Dynamic FP8

Use `--enable-fp8-dynamic` with CUDA and `--dtype float16` or `--dtype bfloat16`.
Dense floating-point checkpoints are quantized during initialization. The native
FP8 kernels require SM89, SM90, SM100 or SM120 and projection widths divisible
by 16. CPU inference and already-quantized checkpoints are not supported by this
option. FP8 remains opt-in and can change model outputs.

| Backbone | FP8 projections |
| --- | --- |
| BERT, RoBERTa, XLM-RoBERTa, CamemBERT | MLP |
| DistilBERT, ModernBERT, dense Nomic BERT, GTE/New, Jina BERT/JinaCode, MPNet, DeBERTa-v2/v3 | MLP |
| Dense Llama, Mistral/Ministral3, Qwen2, Qwen3, UltraBERT | MLP and attention projections; all-linear scope is the default when FP8 is enabled |
| Dense Qwen3.5 | MLP and attention projections, with BF16 activations |
| Gemma3 text, dense Gemma4, EmbeddingGemma2 | MLP |
| Mistral3 multimodal | Text decoder MLP and attention projections |
| Qwen3-VL (including ColQwen3 and Nemotron ColEmbed) and UltraVBERT | Text backbone MLP and attention projections |

Attention calculations, normalization, embedding/output heads and media towers
retain their existing precision. Routed-expert/MoE variants are excluded. Set
`INFINITY_TORCH_FP8_SCOPE=mlp` to limit eligible decoders to feed-forward
projections; `all` is accepted only for the listed dense decoder families.

Classifier and SPLADE heads are architecture-specific. Typed decisions support
Laya, Clef, Pplx and option-token contracts, with checkpoint sidecar loading.
`pplx-decider-flash-v1.2` selects Perplexity’s joint marker readouts, shared text/images,
calibrated answers and question splitting on `/v1/systemone`. Start with
`infinity_v3 --model-id pplx-decider-flash-v1.2 --backend libtorch --dtype bfloat16`.
Unknown custom checkpoint modules and incomplete or ambiguous media towers are
rejected. Gemma3 is text-only; EmbeddingGemma2 includes image/audio towers when
present. The official Gemma4 checkpoint has no audio tower. Qwen3-VL requires
last-token pooling. Gemma3 embedding checkpoints that use last-token pooling,
such as `tencent/KaLM-Embedding-Gemma3-12B-2511`, are served through the same
text path; its [Tencent KaLM community license](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511/blob/main/LICENSE.txt)
is not intended for use within the European Union, so review the checkpoint's
license before deployment.

ColBERT checkpoints declaring `HF_ColBERT` return one matrix per input on
`/embeddings`, `/v1/embeddings`, `/embed`, and `/embed_all`. Rows contain token
vectors, not a pooled sentence embedding. Float, base64-per-row, and tensor
encodings preserve the token matrix. Sentence pooling, dimension overrides, and
single-vector `/similarity` scoring are rejected.

Plain text follows the unaugmented document encoding path, with punctuation
tokens excluded from scoring. Clients can supply final token IDs for model-specific
query/document formatting. Query-marker inputs retain punctuation; query expansion
MASK tokens remain output vectors but don't participate as attention keys.
The server doesn't insert markers or expand queries automatically.

Causal text prefix reuse is available only where the native model reports radix
support. Bidirectional, media and typed decision requests reject folded inputs.
CUDA attention requires FP16 or BF16; CPU uses FP32. MPS and XPU are not supported.
GPU replicas each load a complete model; they do not shard its weights.

Synthetic native and Rust fixtures exercise model paths, heads and sequence
isolation. Independent Transformers references cover supported numerical paths;
latency depends on the model, checkpoint, precision and GPU.

ColQwen3 uses the same matrix response formats as ColBERT. Supply the trained
query suffix (ten `<|endoftext|>` tokens) in text inputs. For image documents, use
a user conversation with an image and `Describe the image.`; the processor ends
the user turn with EOS, without an assistant prefix. ColQwen2/2.5 adapter
checkpoints require separate architecture and adapter support.

NVIDIA ColEmbed V2 4B and 8B use the same matrix API. Provide the checkpoint's
retrieval instructions in user conversations; no query/document prefix is
automatically inserted. The 3B Eagle checkpoint uses a separate architecture
and is not covered by this path.
