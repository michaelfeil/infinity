# Infinity V3 ♾️

Infinity V3 is a high-throughput, low-latency REST API for serving embeddings,
reranking, classification and typed decision models. The server is written in
Rust, with Candle and native LibTorch inference backends.

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
on the checkpoint architecture, output head and backend included in your build.
The [native model coverage](docs/libtorch-model-coverage.md) lists implemented
LibTorch paths and their limitations; it is not a claim that every checkpoint
in each family has been validated.

<details>
  <summary>Text Embeddings</summary>

Text embeddings represent sentences and documents as vectors for semantic
search, clustering and recommendations. Use `/v1/embeddings` or `/embed`.

Example models:

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), dense embeddings
- [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) and [pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b), floating-point output only; contextual variants are unsupported
- [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M), see the [Candle launch recipe](docs/cli.md#ultrabert-embeddings)

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
  <summary>Multi-modal Embeddings</summary>

Multi-modal models encode text and media for retrieval. Send images as
`image_url` content parts in user messages on `/v1/embeddings`.

Example model:

- [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B), native text/image embeddings with last-token pooling

The LibTorch backend also implements Mistral3/Pixtral, Qwen3.5 dense/MoE,
EmbeddingGemma2 and Gemma4 vision paths. These are backbone implementations;
use a checkpoint with the output head and training appropriate for your task.
EmbeddingGemma2 supports image/audio towers when present. Gemma3 is text-only.
See [model coverage](docs/libtorch-model-coverage.md) and
[Ministral 3 multimodal inputs](docs/ministral-3-multimodal.md).

</details>

<details>
  <summary>ColBERT-style and Multi-modal Late-interaction Embeddings</summary>

Late-interaction models return a matrix of token vectors instead of one pooled
vector. Use MaxSim scoring in a compatible retrieval system.

Supported models:

- [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0)
- [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1)
- ColQwen3 (TomoroAI 4B), text/image token vectors with a learned projection
- NVIDIA Nemotron ColEmbed V2 4B and 8B, text/image token vectors; the 3B Eagle architecture is not supported by this path

Use the LibTorch backend and the checkpoint's query/document formatting.
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
  <summary>Typed Decisions</summary>

Typed decision models answer structured questions through `/v1/systemone`.
These checkpoints require their trained decision heads and protocol metadata.

LibTorch supports OneJev, Rune, Laya, Clef, Pplx and option-token contracts.
`pplx-decider-flash-v1.2` supports shared text/images and calibrated answers.
See [decision model configuration](docs/libtorch-backend.md) for architecture,
precision and sidecar requirements.

</details>

## Documentation

- [CLI and deployment options](docs/cli.md)
- [Native LibTorch backend](docs/libtorch-backend.md)
- [Model coverage and limitations](docs/libtorch-model-coverage.md)
- [API and documentation index](docs/README.md)
- [Licensing](LICENSING.md): Infinity Restricted MIT License 1.0, with retained third-party and earlier grants
