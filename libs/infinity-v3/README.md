# Infinity V3 ♾️

Infinity V3 is a high-throughput, low-latency REST API for serving embeddings,
reranking, classification and typed decision models. The server is written in
Rust and uses **LibTorch, the C++ interface of PyTorch**, for model inference.

## Getting started

Launch a compatible Hugging Face model or a local checkpoint:

```bash
infinity_v3 --model-id sentence-transformers/all-MiniLM-L6-v2
```

Check `infinity_v3 --help` for CLI options and environment variables. Open `/docs`
on the running server for the interactive API, or `/api-doc/openapi.json` for
its OpenAPI schema. See [running the server](docs/cli.md) and
[building Infinity V3](CONTRIBUTING.md#working-in-the-infinity-monorepo).

### Supported Tasks and Models by Infinity V3

Click on the sections below to find tasks and **example models**. Support depends
on the checkpoint architecture and trained output head. The models below use
the LibTorch backend.
The [native model coverage](docs/libtorch-model-coverage.md) lists implemented
LibTorch paths and their limitations; it is not a claim that every checkpoint
in each family has been validated.

<details>
  <summary>Text Embeddings — BGE, GTE, ModernBERT, Qwen3 and more</summary>

Text embeddings represent sentences and documents as vectors for semantic
search, clustering and recommendations. Use `/v1/embeddings` or `/embed`.

Example models:

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), dense embeddings
- **GTE embeddings**: [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5), using the native GTE/New encoder path.
- **ModernBERT embeddings**: [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base), using native local/global rotary attention.
- [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) and [pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b), floating-point output only; contextual variants are unsupported
- [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M), native bidirectional encoder with per-layer rotary positions and sliding windows

Native LibTorch families also include RoBERTa, XLM-RoBERTa, CamemBERT,
DistilBERT, ModernBERT, Nomic BERT, Jina BERT/JinaCode, MPNet, Llama,
Mistral/Ministral3, Qwen2, Qwen3/MoE and Gemma3 text.
Use the checkpoint's query/document instructions and pooling configuration.

</details>

<details>
  <summary>Reranking</summary>

Given a query and a list of documents, reranking scores the documents for
relevance. Use `/rerank` with a supported sequence-classification checkpoint
and its trained scoring head.

Example models:

- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16), text and document images through the [native Mistral3 reranking recipe](docs/ministral-3-multimodal.md)

LibTorch supports architecture-specific encoder and decoder classification
heads, including experimental DeBERTa-v2/v3. A generative language-model
checkpoint does not automatically become a reranker.

</details>

<details>
  <summary>Multi-modal Embeddings — Qwen3-VL, EmbeddingGemma2 and Nemotron</summary>

Multi-modal models encode text and media for retrieval. Send images as
`image_url` content parts in user messages on `/v1/embeddings`.

Models and native LibTorch paths:

- **Qwen3-VL Embedding**: [2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and [8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B), text/image embeddings with last-token pooling.
- **EmbeddingGemma2** (`embedding_gemma2`): text embeddings with the checkpoint's trained projection, plus image and audio inputs when the corresponding towers are present.
- **Mistral3 / Ministral 3**: Pixtral vision tower and native text/image preprocessing. See the [multimodal input recipe](docs/ministral-3-multimodal.md).
- **Qwen3.5 dense and MoE**: native text/image backbones with full attention, DeltaNet and vision processing.
- **Gemma4**: native text/image backbone. Gemma3 support is text-only.

Backbone support requires a checkpoint with an appropriate trained output head;
it does not make every instruction model a retrieval embedding model.

**Nemotron 3 and 3.5 dense embeddings:** the Ministral3/Mistral3 backbones are
implemented. Complete query/document prefix handling and embedding parity for
[Nemotron 3 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
[Nemotron 3 8B](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16),
[Nemotron 3.5 1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) and
[Nemotron 3.5 8B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16)
are tracked in [PR #800](https://github.com/michaelfeil/infinity/pull/800).
Nemotron 3 is text-only; 3.5 adds document images. This integration is pending,
not yet part of the supported release list.

For token-vector retrieval, see **ColBERT, ColQwen3 and Nemotron ColEmbed** below.
Full architecture details are in [native model coverage](docs/libtorch-model-coverage.md).

</details>

<details>
  <summary>Late-interaction Embeddings — ColBERT, ColQwen3 and Nemotron ColEmbed</summary>

Late-interaction models return a matrix of token vectors instead of one pooled
vector. Use MaxSim scoring in a compatible retrieval system.

Supported models:

- **ColBERTv2**: [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0)
- **Answer.AI ColBERT Small**: [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1)
- **ColQwen3 (TomoroAI 4B)**, text/image token vectors with a learned projection
- **NVIDIA Nemotron ColEmbed V2 4B and 8B**, text/image token vectors; the 3B Eagle architecture is not supported by this path

ColBERT returns text token vectors; ColQwen3 and Nemotron ColEmbed also encode
images. ColQwen2/2.5 adapters and the Python server's other ColBERT variants are
not implied by these native implementations.

Use the checkpoint's query/document formatting.
Responses preserve the token matrix through float, base64 or packed tensor
encoding. Sentence pooling, dimension overrides and single-vector `/similarity`
scoring are unsupported for these models. See [embedding tensor formats](docs/embedding-tensors.md)
and [model-specific input requirements](docs/libtorch-model-coverage.md).

</details>

<details>
  <summary>Sparse Embeddings and Text Classification</summary>

SPLADE checkpoints produce sparse vocabulary vectors through `/embed_sparse`.
LibTorch implements SPLADE heads for BERT and DistilBERT; head support is
architecture-specific.

Text classification returns trained label scores through `/v1/classify`.
Supported encoder and decoder families need a compatible classification head
and label configuration. See [native model coverage](docs/libtorch-model-coverage.md).

</details>

<details>
  <summary>Typed Decisions — OneJev, Laya, Rune, Clef, Perplexity Decider and D3</summary>

Typed decision models answer structured questions through `/v1/systemone`.
These checkpoints require their trained decision heads and protocol metadata.

Supported model families and protocols (LibTorch):

- **Laya**: ModernBERT-based decision checkpoints with `rl_agent_config.json` and trained decision heads. Accepts text or structured state; native conversations are unsupported.
- **OneJev**, including **Elephant and Mammoth**: Qwen3.5 dense/MoE decision checkpoints using `--decision-protocol onejev`. Supports block-attention question batches and image inputs when the checkpoint includes its vision tower.
- **Perplexity Decider / Pplx**: [v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) and [flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2). Flash v1.2 supports shared text/images, joint question readouts and calibrated answers. The short model aliases select the protocol automatically.
- **Rune**: Gemma4-based decision checkpoints using `--decision-protocol rune`.
- **Clef**: dense Qwen3.5 checkpoints with `joint_head_config.json`, using `--decision-protocol clef`.
- **D3**: [vllm-sr/d3](https://huggingface.co/vllm-sr/d3), a dense Qwen3.5 decision model with its code-readout head. The `d3` alias selects `--decision-protocol d3` automatically.

For example:

```bash
infinity_v3 --model-id pplx-decider-flash-v1.2 --backend libtorch --dtype bfloat16
```

Use the request schema for the checkpoint's protocol: OneJev, Laya and Perplexity
joint v1.2 do not share an interchangeable payload format. See the running
server's `/docs`, [decision inputs](docs/cli.md#structured-decision-inputs) and
[OneJev configuration](docs/libtorch-backend.md#onejev-block-attention).

</details>

## Documentation

- [CLI and deployment options](docs/cli.md)
- [Native LibTorch backend](docs/libtorch-backend.md)
- [Model coverage and limitations](docs/libtorch-model-coverage.md)
- [API and documentation index](docs/README.md)
- [Licensing](LICENSING.md): Infinity Restricted MIT License 1.0, with retained third-party and earlier grants
