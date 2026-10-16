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
| Llama / Llama bidirectional | Packed GQA, RoPE, RMSNorm, fused projections |
| Mistral / Ministral3 | Packed GQA, sliding windows, YaRN |
| Mistral3 / Ministral 3 multimodal | Pixtral vision, 2D rotary positions, spatial patch merger/projector, packed Ministral text |
| Qwen2 | Packed GQA with projection biases |
| Qwen3 / Qwen3 MoE | Packed Q/K norm, GQA, dense/routed MLP and optional projection |
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

### Nemotron 3 Embed 1B

`nvidia/Nemotron-3-Embed-1B-BF16` is a bidirectional Ministral3 decoder with
2048 residual channels, 24 query heads and an explicit head dimension of 128.
Its attention width is 3072; residual width divisibility is not required for
supported decoder families with a positive explicit `head_dim`. Encoder
divisibility checks and native projection-shape validation remain in place.

Generate independent Transformers references from the pinned public checkpoint
and compare packed BF16 mean pooling:

```sh
python scripts/nemotron3-reference.py --device cuda:0 --dtype bfloat16 \
  --output /tmp/nemotron3-reference.json
NEMOTRON3_REFERENCE=/tmp/nemotron3-reference.json NEMOTRON3_DEVICE=cuda:0 \
  cargo test --release -p infinity-backend-libtorch \
  --test decoder_attention_width nemotron3_packed_embeddings_match_transformers \
  -- --ignored --nocapture
```

For strict sampled raw token-state comparisons, generate a second reference with
`--dtype float32` and run the test with `NEMOTRON3_DEVICE=cpu`. Individual BF16
token states can diverge substantially across arithmetic implementations even
when pooled embeddings agree; this test does not assert BF16 raw-state parity.

The fixture includes query/passage prefixes, multilingual text and sequence
lengths through 2049 tokens. Synthetic CPU tests additionally exercise independent
attention widths, model-family aliases, ragged sequence isolation and invalid
dimensions without downloading a checkpoint. This is correctness validation,
not a retrieval benchmark or full-context performance qualification.
