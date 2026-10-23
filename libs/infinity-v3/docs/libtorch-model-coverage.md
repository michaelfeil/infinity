# Experimental native LibTorch model coverage

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
| Mistral3 / Ministral 3 multimodal | Pixtral vision, 2D rotary positions, spatial patch merger/projector, packed Ministral text |
| Mistral3ForSequenceClassification (Nemotron 3.5 Rerank) | Mistral3 backbone with a mean-pooled, bias-free score head; query/document chat template |
| Qwen2 | Packed GQA with projection biases, sequence/token classification heads |
| Perplexity `pplx-embed-v1-0.6b` / `pplx-embed-v1-4b` | Bidirectional Qwen3, mean pooling; floating-point output only |
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection; sequence heads and dense Qwen3 token classification |
| Gemma3 text | Packed local/global attention, Gemma norm semantics |
| EmbeddingGemma2 | Packed text model, shared KV, PLE, embedding projection and image/audio towers |
| Gemma4 / Gemma4 unified | Packed text plus image tower implementation |
| Qwen3-VL | Native vision tower, mRoPE, patch/deepstack mergers and decoder injection |
| Qwen3.5 dense/MoE/text aliases | Packed full attention, DeltaNet, MoE/shared experts and vision tower |
| MPNet | Native exact-length relative-biased attention |
| DeBERTa-v2/v3 (experimental) | Native disentangled relative attention, convolution and classifier |

Classifier and SPLADE heads are architecture-specific. Typed decisions support
Laya, Clef, Pplx and option-token contracts, with checkpoint sidecar loading.
`pplx-decider-flash-v1.2` selects Perplexity’s joint marker readouts, shared text/images,
calibrated answers and question splitting on `/v1/systemone`. Start with
`infinity_v3 --model-id pplx-decider-flash-v1.2 --backend libtorch --dtype bfloat16`.
Unknown custom checkpoint modules and incomplete or ambiguous media towers are
rejected. Gemma3 is text-only; EmbeddingGemma2 includes image/audio towers when
present. The official Gemma4 checkpoint has no audio tower. Qwen3-VL requires
last-token pooling.

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
isolation. Some Candle model/head combinations cannot execute the corresponding
packed request, so a direct Candle comparison is unavailable for those cases.
The latency target remains within 5% of Candle for comparable workloads; this PR
does not claim that target for every model, checkpoint, precision or GPU.
