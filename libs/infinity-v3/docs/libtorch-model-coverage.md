# Experimental native LibTorch model coverage

This lists implemented model paths, not a general accuracy or performance
qualification.
All inference preserves actual token counts; CUDA uses packed attention or
exact-length biased attention without padded token activations.

| Model family / aliases | Native implementation |
| --- | --- |
| BERT | Packed encoder, fused QKV/GELU, pooling, classifier and SPLADE heads |
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
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection; sequence heads and dense Qwen3 token classification |
| Perplexity embeddings (all eight v1/v2 exports) | Bidirectional Qwen3/Qwen3.5; dense, chunk-pooled contextual and 128-dimensional token heads |
| Gemma3 text | Packed local/global attention, Gemma norm semantics |
| EmbeddingGemma2 | Packed text model, shared KV, PLE, embedding projection and image/audio towers |
| Gemma4 / Gemma4 unified | Packed text plus image tower implementation |
| Qwen3-VL | Native vision tower, mRoPE, patch/deepstack mergers and decoder injection |
| Qwen3.5 dense/MoE/text aliases | Packed full attention, DeltaNet, MoE/shared experts and vision tower |
| MPNet | Native exact-length relative-biased attention |
| DeBERTa-v2/v3 (experimental) | Native disentangled relative attention, convolution and classifier |

Classifier and SPLADE heads are architecture-specific. Typed decisions support
Laya, Clef, Pplx and option-token contracts, with checkpoint sidecar loading.
Unknown custom checkpoint modules and incomplete or ambiguous media towers are
rejected. Gemma3 is text-only; EmbeddingGemma2 includes image/audio towers when
present. The official Gemma4 checkpoint has no audio tower. Qwen3-VL requires
last-token pooling.

Causal text prefix reuse is available only where the native model reports radix
support. Bidirectional, media and typed decision requests reject folded inputs.
CUDA attention requires FP16 or BF16; CPU uses FP32. MPS and XPU are not supported.
GPU replicas each load a complete model; they do not shard its weights.

Synthetic native and Rust fixtures exercise model paths, heads and sequence
isolation. Some Candle model/head combinations cannot execute the corresponding
packed request, so a direct Candle comparison is unavailable for those cases.
The latency target remains within 5% of Candle for comparable workloads; this PR
does not claim that target for every model, checkpoint, precision or GPU.

## Perplexity embedding exports

All model IDs below use the `perplexity-ai/` Hugging Face namespace. These are
embedding models, separate from the `pplx` typed decision protocol.

| Model IDs | Embedding contract |
| --- | --- |
| `pplx-embed-v1-0.6b`, `pplx-embed-v1-4b` | Mean pooling, tanh quantization; 1024 / 2560 dimensions |
| `pplx-embed-context-v1-0.6b`, `pplx-embed-context-v1-4b` | Encode all supplied document chunks together, then mean-pool each chunk; 1024 / 2560 dimensions |
| `pplx-embed-v1-late-0.6b` | Bidirectional Qwen3, query expansion, punctuation masking; 128 dimensions per token |
| `pplx-embed-v2-context-9b-preview` | Qwen3.5, chunk pooling before its FP32 projection and tanh quantization; 2048 dimensions |
| `pplx-embed-v2-late-0.6b`, `pplx-embed-v2-late-9b` | Bidirectional full-attention layers with causal DeltaNet layers, text/images, punctuation masking; 128 dimensions per token |

For example, launch the small dense model:

```sh
infinity_v3 --model-id perplexity-ai/pplx-embed-v1-0.6b --backend libtorch --dtype bfloat16
```

Use `--torch-device cpu --dtype float32` for CPU inference. The Candle CUDA
backend recognizes the same exports; Qwen3.5 requires BF16 CUDA with FlashAttention.
Serving does not load or execute checkpoint Python code.

Dense inputs use `/v1/embeddings` (or `/embed` with `inputs`):

```json
{"input":["A query","A document"],"quantization":"int8"}
```

The default `int8` output contains unnormalized integer-valued embeddings.
Use cosine similarity. V1 dense/contextual models also accept `binary` and
`ubinary`; `ubinary` returns packed byte values, with the first dimension in
the most significant bit. `dimensions` selects the leading dimensions before
binary packing. `normalize: true` applies after quantization and dimension
selection; it is incompatible with packed binary output. The existing `float`,
`base64`, and `tensor` wire encoding formats remain available (their wire dtype
is float32, including integer-valued or packed-byte-valued embeddings).

Contextual models accept `/v1/contextualizedembeddings` with nested documents:

```json
{"input":[["First chunk.","Second chunk."],["Another document."]]}
```

Responses preserve document and chunk order: `data[document].data[chunk].embedding`.
Chunks share document context; separate documents remain independent. With the
default int8 quantization, empty chunks produce zero vectors. The v2 contextual model also accepts `task: "query"`
with one string per input document. V1 contextual indexes normally use the
matching v1 dense model for queries. The `/v1/embeddings` route accepts the same
nested input as an extension for contextual models.

Late-interaction inputs use `/v1/embeddings` or `/embed_all`:

```json
{"input":["What is science?"],"task":"query"}
```

Use `task: "document"` for documents (the default). Outputs are matrices of
unit-normalized 128-dimensional token vectors; use MaxSim for retrieval.
Task markers, v1 query expansion and document punctuation masking are automatic.
These fixed-width token heads do not accept `dimensions`.

V2 late-interaction models also accept image-only inputs:

```json
{"input":{"image_url":"data:image/png;base64,..."},"task":"document"}
```

A batch is a list of image objects. Image preprocessing uses the checkpoint's
saved Qwen processor, token IDs and mRoPE positions. Remote image URLs follow
the server's existing allowed-host settings. Text and image batches are separate.

Perplexity adapters reject inputs beyond their task-specific limit or configured
`--max-batch-tokens`; they do not silently truncate chunks or query expansion.
V1 contextual chunks must not contain the reserved `<|endoftext|>` separator.
Long contextual readout also needs sufficient host-output/response memory budgets:
raw hidden states are transferred before per-chunk FP32 readout. Native backbones
and late-interaction token projections execute on the selected accelerator.

## Perplexity validation

`scripts/perplexity-reference.py` pins each public export revision and can generate
small models with the original tokenizers and head contracts, then compare the
native HTTP responses against the public Python implementations:

```sh
python scripts/perplexity-reference.py --output /tmp/pplx --tiny
python scripts/perplexity-reference.py --output /tmp/pplx --check-only --server target/debug/infinity_v3
```

The script documents its reference dependencies. Use `--model` to select an
export; omit `--tiny` to use trained weights. CPU FP32 checks cover all eight
synthetic exports, contextual boundaries/empty chunks, query expansion and
punctuation masking. Trained 0.6B dense and v2 late-interaction checks additionally
cover real text and image outputs. These numerical checks do not establish
retrieval benchmark accuracy or performance for every checkpoint and dtype.
