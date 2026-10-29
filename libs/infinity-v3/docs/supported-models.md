# Supported models and configuration

Infinity v3 serves dense, sparse and token-vector embeddings, reranking,
classification, and structured decisions with the native LibTorch backend.
Choose a checkpoint below by task. The [architecture coverage](libtorch-model-coverage.md)
explains the underlying model paths and their limitations.

Use documentation matching your image version; `main` tracks ongoing development.
Use an image containing the model integration you need. Listing a checkpoint describes its supported
task, not an accuracy or throughput guarantee for every precision and GPU.
The support-basis column identifies the implemented architecture and task head
or pooling used by each checkpoint. Related checkpoints are included when their
published configuration matches that path; sharing a training organization does
not by itself establish support. Checkpoint-specific execution and architecture
fixture coverage are different levels of validation.

## Find a model by task

| Task | Endpoint | Output | Model list |
| --- | --- | --- | --- |
| Text retrieval, similarity, clustering | `/v1/embeddings` | One dense vector per input | [Dense text embeddings](#dense-text-embeddings) |
| Image, document or audio retrieval | `/v1/embeddings` | One dense vector per input | [Multimodal embeddings](#multimodal-embeddings) |
| Token-vector retrieval / MaxSim | `/v1/embeddings` | A token-vector matrix per input | [Late-interaction embeddings](#late-interaction-embeddings) |
| Query/document relevance | `/rerank` | Ranked document scores | [Rerankers](#rerankers) |
| PII, named entities, medical entities | `/predict_tokens` | Token labels or aggregated spans | [Token classification and NER](#token-classification-and-ner) |
| Sentiment, sequence labels, reward scoring | `/v1/classify` | Class scores or raw reward logits | [Classification and reward scoring](#text-classification-and-reward-scoring) |
| Structured questions and decisions | `/v1/systemone` | Typed answers | [Typed decisions](#typed-decisions) |
| Vocabulary-based sparse retrieval | `/embed_sparse` | Vocabulary index/value pairs | [Sparse embeddings](#sparse-embeddings) |

## Common launch configuration

Use a published NVIDIA image from the
[container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3).
The example pins `nvidia-3.0.0`; pin a version or digest for reproducible deployments.
CUDA requires an Ampere/SM80 or newer GPU and NVIDIA Container Toolkit.

```bash
docker run --rm --gpus all -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0 \
  --model-id BAAI/bge-small-en-v1.5 \
  --torch-device cuda --dtype bfloat16 \
  --max-batch-tokens 8192 \
  --hostname 0.0.0.0 --port 8000
```

Replace `--model-id` and the task settings with a table row below. The native
image selects LibTorch automatically. For a locally installed binary, pass the
same flags to `infinity_v3`. Tables use BF16 compute unless a row specifies a
different `--dtype`; replace the common dtype flag in that case. Checkpoint
storage precision does not set compute precision.
For CPU, use a LibTorch CPU build, `--torch-device cpu` and `--dtype float32`.
The published `cpu-*` image uses Candle and has different model coverage.

`--max-batch-tokens` is the total batch capacity, not a model's context length or
a memory guarantee. These are starting configurations: lower the budget for
smaller GPUs and leave room for activations, media tokens and warmup.
`--max-sequence-length` can impose a separate per-input cap. Automatic text
truncation is enabled by default; use `--auto-truncate=false` to reject oversized
text inputs instead. Some tasks and media inputs reject oversized inputs regardless
of this setting. Disable truncation when dropping text would change the intended
result. See [sequence and batch limits](cli.md#sequence-length-and-batch-capacity).

Use `--revision` to pin a Hugging Face checkpoint, or pass a local directory
containing its weights, tokenizer, pooling/head configuration and task sidecars.
Gated checkpoints also require access credentials. A base language model needs
a trained retrieval or classification head to provide that task's intended scores.

All visible CUDA GPUs are used by default. Each GPU loads a full replica,
including every MoE expert. To select only two or four visible GPUs, add
`--backend-device-ids 0,1` or `--backend-device-ids 0,1,2,3`.
Every selected GPU must fit the full model; this does not shard weights.
Wait for `/health` before sending traffic; model warmup is part of startup.

## Dense text embeddings

Use `/v1/embeddings` with `input` strings. Pooling is read from the checkpoint
metadata automatically; the examples omit redundant overrides. Keep the saved
`1_Pooling/config.json` when copying a checkpoint to a local directory. Explicit
`--pooling splade` selects the sparse task described below.

For Nomic v1.5, prefix retrieval queries with `search_query: ` and documents with
`search_document: `. It returns 768 dimensions by default. Qwen3 and Harrier
queries use the checkpoint's instruction format (`Instruct: ...\nQuery: ...`);
documents use their ordinary text. E5 v2 and non-instruct multilingual E5 use
`query: ` and `passage: ` prefixes, including for non-English text. E5-Mistral,
multilingual E5 large instruct and GTE-Qwen2 use their model-card
instruction/query formats.
Jina code models distinguish code and text retrieval tasks through their trained
prompts. Nomic v1 uses the same task-prefix convention as v1.5.
GTE, BGE, Snowflake and Mixedbread also have model-specific
retrieval instructions: follow the linked model card consistently when building
and querying an index. A saved `prompt_name` can select a checkpoint prompt;
do not also insert the same prefix manually.

Perplexity Embed v1 supports floating-point output only; contextual variants are
unsupported. BGE-M3 here provides dense embeddings. Nemotron Embed 3 supports
text; 3.5 additionally supports document images. Use their query/document
instructions and checkpoint formatting; they are separate from Nemotron ColEmbed.
See [Nemotron embedding configuration](nemotron-embeddings.md) for named prefixes
and text/image request formats.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) | GTE rotary encoder + sentence pooling | `--max-batch-tokens 8192` |
| [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) | GTE rotary encoder + sentence pooling | `--max-batch-tokens 8192` |
| [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | GTE rotary encoder + sentence pooling | `--max-batch-tokens 8192` |
| [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) | Qwen2 + sentence pooling | `--max-batch-tokens 8192` |
| [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) | Qwen2 + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-base-zh-v1.5](https://huggingface.co/BAAI/bge-base-zh-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 16384` |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [dmis-lab/UltraBERT-Embed-150M](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) | UltraBERT + sentence pooling | `--max-batch-tokens 32768` |
| [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) | Mistral + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [jinaai/jina-code-embeddings-0.5b](https://huggingface.co/jinaai/jina-code-embeddings-0.5b) | Qwen2 + sentence pooling; [non-commercial license](#model-licenses) | `--max-batch-tokens 8192` |
| [jinaai/jina-code-embeddings-1.5b](https://huggingface.co/jinaai/jina-code-embeddings-1.5b) | Qwen2 + sentence pooling; [non-commercial license](#model-licenses) | `--max-batch-tokens 8192` |
| [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) | JinaCode encoder + sentence pooling | `--max-batch-tokens 8192` |
| [jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en) | Jina BERT/ALiBi + sentence pooling | `--max-batch-tokens 2048` |
| [jinaai/jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en) | Jina BERT/ALiBi + sentence pooling | `--max-batch-tokens 8192` |
| [microsoft/harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m) | Gemma3 text + last-token pooling | `--max-batch-tokens 8192` |
| [microsoft/harrier-oss-v1-0.6b](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | Qwen3 + last-token pooling | `--max-batch-tokens 8192` |
| [microsoft/harrier-oss-v1-27b](https://huggingface.co/microsoft/harrier-oss-v1-27b) | Gemma3 text + last-token pooling, linear RoPE scaling | `--max-batch-tokens 8192` |
| [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [nomic-ai/modernbert-embed-base](https://huggingface.co/nomic-ai/modernbert-embed-base) | ModernBERT + sentence pooling | `--max-batch-tokens 8192` |
| [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) | Nomic BERT + sentence pooling | `--max-batch-tokens 8192` |
| [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | Nomic BERT + sentence pooling | `--max-batch-tokens 8192` |
| [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) | Nomic BERT MoE + sentence pooling | `--max-batch-tokens 8192` |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | Bidirectional Ministral3 + sentence pooling | `--max-batch-tokens 8192` |
| [nvidia/Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) | Bidirectional Ministral3 + sentence pooling | `--max-batch-tokens 8192` |
| [perplexity-ai/pplx-embed-v1-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b) | Bidirectional Qwen3 + sentence pooling | `--max-batch-tokens 8192` |
| [perplexity-ai/pplx-embed-v1-4b](https://huggingface.co/perplexity-ai/pplx-embed-v1-4b) | Bidirectional Qwen3 + sentence pooling | `--max-batch-tokens 8192` |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Qwen3 + sentence pooling | `--max-batch-tokens 8192` |
| [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | Qwen3 + sentence pooling | `--max-batch-tokens 8192` |
| [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | Qwen3 + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/all-distilroberta-v1](https://huggingface.co/sentence-transformers/all-distilroberta-v1) | RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | MPNet + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/all-roberta-large-v1](https://huggingface.co/sentence-transformers/all-roberta-large-v1) | RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/bert-base-nli-mean-tokens](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) | DistilBERT + sentence pooling + saved Dense projection | `--max-batch-tokens 8192` |
| [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) | BERT + sentence pooling + saved Dense projection | `--max-batch-tokens 8192` |
| [sentence-transformers/msmarco-distilbert-base-v4](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v4) | DistilBERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/msmarco-MiniLM-L12-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L12-cos-v5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/msmarco-MiniLM-L6-cos-v5](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L6-cos-v5) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/multi-qa-MiniLM-L6-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/multi-qa-MiniLM-L6-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-dot-v1) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/multi-qa-mpnet-base-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1) | MPNet + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1) | MPNet + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/paraphrase-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L12-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) | XLM-RoBERTa + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) | Nomic BERT + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-m-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) | GTE rotary encoder + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [Snowflake/snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [tencent/KaLM-Embedding-Gemma3-12B-2511](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) | Gemma3 text + last-token pooling; [community license, not for EU use](#model-licenses) | `--max-batch-tokens 8192` |
| [thenlper/gte-base](https://huggingface.co/thenlper/gte-base) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [thenlper/gte-large](https://huggingface.co/thenlper/gte-large) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) | BERT + sentence pooling | `--max-batch-tokens 8192` |
| [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano) | Qwen3 + sentence pooling | `--max-batch-tokens 8192` |

For the Multi-QA `*-dot-v1` checkpoints and `msmarco-distilbert-base-v4`,
use `/embed` with `normalize: false` when retaining the trained vector magnitudes
for dot-product retrieval. `/v1/embeddings` normalizes outputs. The `*-cos-*`
variants use normalized vectors. Keep document and query preprocessing consistent.
LaBSE and DistilUSE also require their saved Dense projection modules; retain
those subdirectories and `modules.json` when copying weights to a local directory.

## Multimodal embeddings

Use `/v1/embeddings`. Qwen3-VL requires last-token pooling (selected by its
checkpoint defaults). EmbeddingGemma2 supports image/audio when the downloaded
checkpoint includes the corresponding towers. Nemotron 3.5 accepts text and
document images. Media inputs must fit the actual token and memory budgets.
See [multimodal requests](#multimodal-request) and the linked model cards.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | EmbeddingGemma2 projection + media towers | `--max-batch-tokens 8192` |
| [nvidia/Nemotron-3.5-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) | Bidirectional Mistral3 + mean pooling + image tower | `--max-batch-tokens 8192` |
| [nvidia/Nemotron-3.5-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16) | Bidirectional Mistral3 + mean pooling + image tower | `--max-batch-tokens 8192` |
| [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Qwen3-VL + last-token pooling + image tower | `--max-batch-tokens 16384` |
| [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | Qwen3-VL + last-token pooling + image tower | `--max-batch-tokens 16384` |

## Late-interaction embeddings

Use `/v1/embeddings` to receive a **matrix of token vectors per input**, then
score with MaxSim in your retrieval system. Do not force sentence pooling,
request a dimension override, or use single-vector `/similarity` scoring.
ColQwen3 additionally accepts images. It needs the trained query suffix of ten
`<|endoftext|>` tokens; image documents use a user image message with
`Describe the image.` and no assistant prefix.

ColBERT does not automatically add query/document markers or expand queries.
Use the checkpoint's exact input formatting; final token IDs can express that
formatting directly. Native NVIDIA Nemotron ColEmbed V2 4B/8B is also supported
with normalized text/image token vectors and its checkpoint retrieval instructions;
the separate 3B Eagle architecture is outside this path.
See [matrix formats](embedding-tensors.md) and
[model-specific requirements](libtorch-model-coverage.md).

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [answerdotai/answerai-colbert-small-v1](https://huggingface.co/answerdotai/answerai-colbert-small-v1) | Packed BERT + ColBERT token projection | `--max-batch-tokens 8192` |
| [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0) | Packed BERT + ColBERT token projection | `--max-batch-tokens 8192` |
| [nvidia/nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | Bidirectional Qwen3-VL + ColEmbed token readout | `--max-batch-tokens 8192` |
| [nvidia/nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | Bidirectional Qwen3-VL + ColEmbed token readout | `--max-batch-tokens 8192` |
| [TomoroAI/tomoro-colqwen3-embed-4b](https://huggingface.co/TomoroAI/tomoro-colqwen3-embed-4b) | Qwen3-VL + learned ColQwen3 token projection | `--max-batch-tokens 8192` |

## Rerankers

Use `/rerank` with `query` and `texts`. These models use their trained score
heads; embedding pooling flags do not apply. Nemotron 1B uses its saved average
pooling and temperature. Nemotron 3.5 renders the query/document chat template;
image documents use `/v1/classify` with `query`/`document` messages and
`use_activation: false`. See the [Nemotron multimodal reranking recipe](ministral-3-multimodal.md#cross-encoder-reranking).

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | GTE rotary encoder + trained relevance head | `--max-batch-tokens 8192` |
| [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) | ModernBERT + trained relevance head | `--max-batch-tokens 8192` |
| [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | XLM-RoBERTa + trained relevance head | `--max-batch-tokens 8192` |
| [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) | XLM-RoBERTa + trained relevance head | `--max-batch-tokens 8192` |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | XLM-RoBERTa + trained relevance head | `--max-batch-tokens 8192` |
| [cross-encoder/mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | XLM-RoBERTa + trained relevance head | `--max-batch-tokens 8192` |
| [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [cross-encoder/ms-marco-MiniLM-L4-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L4-v2) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [cross-encoder/ms-marco-TinyBERT-L2-v2](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [mixedbread-ai/mxbai-rerank-base-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [mixedbread-ai/mxbai-rerank-large-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | BERT + trained relevance head | `--max-batch-tokens 8192` |
| [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) | Bidirectional Llama + trained relevance head | `--max-batch-tokens 8192` |
| [nvidia/Nemotron-3.5-Rerank-8B-BF16](https://huggingface.co/nvidia/Nemotron-3.5-Rerank-8B-BF16) | Mistral3 + trained relevance head | `--max-batch-tokens 8192` |

## Token classification and NER

Use `/predict_tokens` with `inputs`, rather than `/v1/embeddings`. Label names
come from the checkpoint's `id2label`. `aggregation_strategy: "simple"` groups
tokens into spans; choose a strategy appropriate for the checkpoint's labeling
scheme. PII detection and medical NER have different label sets and training domains.
The BERT CoNLL and English RoBERTa models cover general entities; Davlan and
WikiNEuRal provide multilingual entities, CamemBERT covers French, and the
clinical/de-identification models have domain-specific labels. DeBERTa-v2/v3
execution remains experimental. Training on clinical text does not imply coverage
of every medical concept or de-identification policy.

The native loader requires `tokenizer.json` at the checkpoint root; `vocab.txt`
alone is insufficient. For `dslim/bert-base-NER`, use `--revision refs/pr/27`
until [the upstream tokenizer addition](https://huggingface.co/dslim/bert-base-NER/discussions/27)
is merged. This revision adds the converted tokenizer without changing weights.
It has been validated with Infinity 3.0.0 GPU NER inference and all aggregation
modes. For a local copy, include this revision's root `tokenizer.json` alongside
the model weights and configuration.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [Babelscape/wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | BERT + token-label head; [non-commercial license](#model-licenses) | `--max-batch-tokens 8192` |
| [Clinical-AI-Apollo/Medical-NER](https://huggingface.co/Clinical-AI-Apollo/Medical-NER) | Experimental DeBERTa-v2 + token-label head | `--max-batch-tokens 8192` |
| [Davlan/bert-base-multilingual-cased-ner-hrl](https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl) | BERT + token-label head | `--max-batch-tokens 8192` |
| [Davlan/xlm-roberta-base-ner-hrl](https://huggingface.co/Davlan/xlm-roberta-base-ner-hrl) | XLM-RoBERTa + token-label head | `--max-batch-tokens 8192` |
| [dbmdz/bert-large-cased-finetuned-conll03-english](https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english) | BERT + token-label head | `--max-batch-tokens 8192` |
| [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | BERT + token-label head; requires tokenizer revision above | `--revision refs/pr/27 --max-batch-tokens 8192` |
| [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) | BERT + token-label head | `--max-batch-tokens 8192` |
| [gravitee-io/bert-small-pii-detection](https://huggingface.co/gravitee-io/bert-small-pii-detection) | BERT + token-label head | `--max-batch-tokens 8192` |
| [Jean-Baptiste/camembert-ner](https://huggingface.co/Jean-Baptiste/camembert-ner) | CamemBERT + token-label head | `--max-batch-tokens 8192` |
| [Jean-Baptiste/roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | RoBERTa + token-label head | `--max-batch-tokens 8192` |
| [obi/deid_bert_i2b2](https://huggingface.co/obi/deid_bert_i2b2) | BERT + token-label head | `--max-batch-tokens 8192` |
| [obi/deid_roberta_i2b2](https://huggingface.co/obi/deid_roberta_i2b2) | RoBERTa + token-label head | `--max-batch-tokens 8192` |
| [OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M](https://huggingface.co/OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M) | BERT + token-label head | `--max-batch-tokens 8192` |
| [OpenMed/OpenMed-NER-GenomicDetect-ModernMed-395M](https://huggingface.co/OpenMed/OpenMed-NER-GenomicDetect-ModernMed-395M) | ModernBERT + token-label head | `--max-batch-tokens 8192` |
| [OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M](https://huggingface.co/OpenMed/OpenMed-NER-OncologyDetect-MultiMed-568M) | XLM-RoBERTa + token-label head | `--max-batch-tokens 8192` |
| [OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M](https://huggingface.co/OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M) | XLM-RoBERTa + token-label head | `--max-batch-tokens 8192` |

## Text classification and reward scoring

Use `/v1/classify`. Skywork scores a user/assistant conversation; send native
`messages` and `use_activation: false` to obtain a raw reward score. Applying
softmax to a single-label head would return 1.0 instead of a useful reward.
FinBERT provides financial-sentiment labels; CardiffNLP provides its saved
social-text sentiment labels. Use `input` for those sequence classifiers and
retain their checkpoint label mapping. Preprocess mentions, URLs and other text
as required by the corresponding model card.
Other supported encoder/decoder classification checkpoints need a saved trained
head and matching label configuration; architecture support alone does not
supply labels or turn an embedding model into a classifier.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | RoBERTa + sequence-label/reward head | `--max-batch-tokens 8192` |
| [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) | BERT + sequence-label/reward head | `--max-batch-tokens 8192` |
| [Skywork/Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B) | Qwen3 + sequence-label/reward head | `--max-batch-tokens 8192` |

## Typed decisions

Use `/v1/systemone` and retain the checkpoint's trained heads and protocol
sidecars. Set the protocol explicitly as shown; Laya is detected from its
`rl_agent_config.json` and needs no decoder protocol flag. A local path or custom
served model name does not select a protocol by its name. The `d3`, `d3-mini`, `d3-ultra`,
`pplx-decider-v1.1-27b` and `pplx-decider-flash-v1.2` short aliases select their
registered checkpoints and protocols automatically.

The protocol determines supported question types, media and request semantics;
do not assume every model accepts the same payload. Laya accepts text/JSON state
and rejects native message conversations. OneJev supports image inputs when the
vision tower is present; Perplexity Flash v1.2 supports shared text/images and
calibrated joint readouts. See [decision inputs](cli.md#structured-decision-inputs),
[OneJev configuration](libtorch-backend.md#onejev-block-attention), and the running
server's `/docs` for the selected protocol.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [baselabs/elephant-1](https://huggingface.co/baselabs/elephant-1) | Qwen3.5 dense/MoE + OneJev decision head | `--decision-protocol onejev` |
| [baselabs/mammoth-1](https://huggingface.co/baselabs/mammoth-1) | Qwen3.5 dense/MoE + OneJev decision head | `--decision-protocol onejev` |
| [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) | Qwen3.5 + Clef joint head | `--decision-protocol clef --max-batch-tokens 8192` |
| [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) | Qwen3.5 + Clef joint head | `--decision-protocol clef --max-batch-tokens 16384` |
| [michaelfeil/laya-typed-decisions](https://huggingface.co/michaelfeil/laya-typed-decisions) | ModernBERT encoder + Laya decision heads | `--dtype float16` |
| [michaelfeil/rune-26b-a4b](https://huggingface.co/michaelfeil/rune-26b-a4b) | Gemma4 + Rune decision head | `--decision-protocol rune --max-batch-tokens 8192` |
| [OmniJev/OneJev1.6-0.8B](https://huggingface.co/OmniJev/OneJev1.6-0.8B) | Qwen3.5 dense/MoE + OneJev decision head | `--decision-protocol onejev` |
| [perplexity-ai/pplx-decider-flash-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | Qwen3.5 + Perplexity decision readouts | `--decision-protocol pplx --max-batch-tokens 8192` |
| [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | Qwen3.5 + Perplexity decision readouts | `--decision-protocol pplx --max-batch-tokens 8192` |
| [perplexity-ai/pplx-decider-v1.2](https://huggingface.co/perplexity-ai/pplx-decider-v1.2) | Qwen3.5 + Perplexity decision readouts | `--decision-protocol pplx --max-batch-tokens 8192` |
| [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) | Qwen3.5 + D3 code-readout head | `--decision-protocol d3 --max-batch-tokens 32768` |
| [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini) | Qwen3.5 + D3 code-readout head | `--decision-protocol d3 --max-batch-tokens 32768` |
| [vllm-sr/d3-ultra](https://huggingface.co/vllm-sr/d3-ultra) | Qwen3.5-27B + D3 prefix-bidirectional code-readout head | `--decision-protocol d3 --dtype bfloat16 --max-batch-tokens 8192` |

## Sparse embeddings

Use `/embed_sparse` with `{"inputs":["A document to index."]}`. Set
`--pooling splade` to select the trained masked-LM head and sparse max-pooling
path. The response contains vocabulary index/value pairs. These checkpoints
provide vocabulary scores, not dense sentence vectors.

| Checkpoint | Native implementation / support basis | Additional launch settings |
| --- | --- | --- |
| [naver/splade-cocondenser-ensembledistil](https://huggingface.co/naver/splade-cocondenser-ensembledistil) | BERT masked-LM head + SPLADE pooling | `--pooling splade --max-batch-tokens 8192` |
| [naver/splade-cocondenser-selfdistil](https://huggingface.co/naver/splade-cocondenser-selfdistil) | BERT masked-LM head + SPLADE pooling | `--pooling splade --max-batch-tokens 8192` |
| [naver/splade_v2_distil](https://huggingface.co/naver/splade_v2_distil) | DistilBERT masked-LM head + SPLADE pooling | `--pooling splade --max-batch-tokens 8192` |
| [naver/splade_v2_max](https://huggingface.co/naver/splade_v2_max) | DistilBERT masked-LM head + SPLADE pooling | `--pooling splade --max-batch-tokens 8192` |

## Request examples

Run the corresponding model first. The `model` field is optional with one loaded
model; if provided, it must match its ID or configured `--served-model-name`.

### Dense embedding request

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-small-en-v1.5","input":["A document about semantic search."],"encoding_format":"float"}'
```

### Multimodal request

After starting `Qwen/Qwen3-VL-Embedding-2B`, send a user conversation. Replace
`<base64 PNG>` with an actual base64-encoded PNG (without the angle brackets).
Inline media avoids configuring a remote-image host allowlist.

```json
{
  "model": "Qwen/Qwen3-VL-Embedding-2B",
  "input": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64 PNG>"}},
    {"type": "text", "text": "Represent this image for retrieval."}
  ]}],
  "encoding_format": "float"
}
```

Post this JSON to `/v1/embeddings`. Use the checkpoint's trained retrieval prompt
for production queries and documents. Audio/video support is model-specific;
Qwen3-VL's image path does not imply support for those modalities.

### Reranking request

```bash
curl http://localhost:8000/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-reranker-base","query":"What is semantic search?","texts":["Retrieval by meaning.","A cake recipe."],"raw_scores":true}'
```

The response ranks the supplied documents. Raw logits are relevance scores,
not calibrated probabilities.

### Token classification request

```bash
curl http://localhost:8000/predict_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"gravitee-io/bert-small-pii-detection","inputs":["Contact Jane Smith in San Francisco."],"aggregation_strategy":"simple"}'
```

The response reports predicted labels, scores and spans according to the
checkpoint and aggregation strategy.

### Reward scoring request

```bash
curl http://localhost:8000/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"model":"Skywork/Skywork-Reward-V2-Qwen3-0.6B","messages":[{"role":"user","content":"Why is the sky blue?"},{"role":"assistant","content":"Air molecules scatter shorter blue wavelengths more strongly."}],"use_activation":false}'
```

### Sentiment classification request

After starting `ProsusAI/finbert`, send strings to the sequence classifier:

```bash
curl http://localhost:8000/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"model":"ProsusAI/finbert","input":["The company reported stronger earnings."]}'
```

Class scores use the checkpoint's positive, negative and neutral labels.
They are separate from Skywork's raw reward scores.

### Sparse embedding request

Start `naver/splade-cocondenser-ensembledistil` with `--pooling splade`, then:

```bash
curl http://localhost:8000/embed_sparse \
  -H 'Content-Type: application/json' \
  -d '{"model":"naver/splade-cocondenser-ensembledistil","inputs":["A document about semantic search."]}'
```

### D3-mini decision request

```bash
curl http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"vllm-sr/d3-mini","state":"The parcel arrived damaged. Please send a replacement.","questions":{"route":{"type":"choice","instructions":"Which team should handle this request?","criteria":{"returns":"Damaged items and replacements","billing":"Payments and invoices"}}}}'
```

This returns a typed decision for `route`. Use the running schema for other
protocols and question types. D3-mini can also process supported image state;
text-only examples do not require a media configuration.

D3 Ultra uses the same endpoint and `choice`, `noul`, and `score` question types.
Its checkpoint selects the `header-v1` text prompt, prefix-bidirectional attention
and question-count temperature calibration automatically. Text and shared
top-level image/video inputs are supported; video decoding requires `ffmpeg` and
`ffprobe` in the server environment. Box/Track questions and image references
inside instructions or criteria are not implemented. BF16 weights require
approximately 52 GB before inference buffers; validate batch limits against
available GPU memory.

### Vela3 core decisions

Vela3 models use `/v1/systemone` with the native LibTorch backend. The short
aliases below and their full `vllm-sr/` IDs select `--decision-protocol vela3`
automatically. For a local checkpoint directory, pass that protocol explicitly.

| Model | Native backbone | Core question types | Inputs |
| --- | --- | --- | --- |
| [vela3-fast](https://huggingface.co/vllm-sr/vela3-fast) | UltraBERT-380M with its distilled cosine + MLP decision head | `choice`, `noul` | Text or JSON |
| [vela3-fast-vision](https://huggingface.co/vllm-sr/vela3-fast-vision) | UltraVBERT: UltraBERT + SigLIP + pixel-shuffle connector, using the decision-trained backbone | `choice`, `noul` | Text or JSON, optionally 1–3 shared images |
| [vela3-lite](https://huggingface.co/vllm-sr/vela3-lite) | Pinned D3-lite / Qwen3.5-0.8B | `choice`, `noul`, `score` | Text or JSON, shared media through the D3 path |
| [vela3-mini](https://huggingface.co/vllm-sr/vela3-mini) | Pinned D3-mini / Qwen3.5-4B | `choice`, `noul`, `score` | Text or JSON, shared media through the D3 path |
| [vela3-flash](https://huggingface.co/vllm-sr/vela3-flash) | Pinned D3-flash / Qwen3.5-9B | `choice`, `noul`, `score` | Text or JSON, shared media through the D3 path |
| [vela3-turbo](https://huggingface.co/vllm-sr/vela3-turbo) | Pinned D3-turbo / Qwen3.5-27B, prefix-bidirectional attention | `choice`, `noul`, `score` | Text or JSON, shared media through the D3 path |

```bash
infinity_v3 --model-id vela3-fast --max-batch-tokens 4096
```

```json
{"state":"My card was charged twice. Please refund the duplicate.","questions":{"route":{"type":"choice","instructions":"What does the customer want?","criteria":{"refund":"Money back for the duplicate charge","cancel":"Cancel the order"}},"urgent":{"type":"noul","instructions":"The customer needs a fast response."}}}
```

This integration serves **general decisions with task adapters off**. Task-specific
router signals, adapter opt-in, span extraction (including NER, PII, relations,
evidence and hallucination spans), generated label definitions, Box/Track and OCR
readouts are not implemented. Requests matching trained adapter routes are rejected
instead of silently using a different head. Use a general question ID and wording
for ordinary decisions; changing the ID is not a substitute for a trained router
signal. The encoder models do not implement `score`, even though some family
descriptions list it; the Qwen-based models do.

The Qwen packages download the D3 base revision pinned in `vela3_config.json`,
including its tokenizer, prompt and FP32 readout. They do not substitute the base
model's current `main`. No checkpoint Python code executes in the server.
CUDA defaults to BF16 for Vela encoders; all learned decision heads remain FP32.
The maximum formatted input is 32,768 tokens, also bounded by the configured
sequence and batch token limits. Inputs are never truncated. D3 video processing
requires `ffmpeg` and `ffprobe`; the UltraVBERT encoder accepts images only.


## Model licenses

Model weight licenses are separate from [Infinity's license](../LICENSING.md).
Technical support includes checkpoints with non-commercial terms. The following
catalog entries publish such terms; use their linked model cards and license
files for the applicable conditions, attribution and any separate permissions.
This is not an exhaustive license table for the catalog.

| Checkpoint | Published model license | Task |
| --- | --- | --- |
| `jinaai/jina-code-embeddings-0.5b` | [CC-BY-NC-4.0](https://huggingface.co/jinaai/jina-code-embeddings-0.5b/blob/main/README.md) | Code/text embeddings |
| `jinaai/jina-code-embeddings-1.5b` | [CC-BY-NC-4.0](https://huggingface.co/jinaai/jina-code-embeddings-1.5b/blob/main/README.md) | Code/text embeddings |
| `Babelscape/wikineural-multilingual-ner` | [CC-BY-NC-SA-4.0](https://huggingface.co/Babelscape/wikineural-multilingual-ner/blob/main/README.md) | Multilingual NER |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` | [Tencent KaLM-Embedding community license](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511/blob/main/LICENSE.txt), not intended for use within the EU | Text embeddings |

Nomic Embed v1, v1.5 and v2 MoE publish Apache-2.0 model licenses. Jina's
license varies by checkpoint: the v2 text/code encoder family and the Qwen2
code-embedding family do not all share the same weight license. Other catalog
entries may also carry custom or restrictive terms; check the exact revision.

## Similar models that need additional integration

A familiar organization name or Hugging Face task tag is insufficient to select
a native implementation. These examples are excluded from the supported model
tables above:

| Model or family | Missing native integration |
| --- | --- |
| `jinaai/jina-embeddings-v3` | Custom XLM-RoBERTa Flash rotary encoder and task-specific LoRA adapters; ordinary XLM-RoBERTa support does not cover this contract. |
| `jinaai/jina-reranker-v2-base-multilingual` | Custom XLM-RoBERTa Flash configuration/loader; this checkpoint does not declare a standard native `model_type`. |
| Jina reranker v1 tiny/turbo | Checkpoints declare `JinaBertModel` rather than a native sequence-classification architecture; their reranker head/loading contract needs integration. |
| `cross-encoder/ms-marco-electra-base` | ELECTRA backbone and classifier support. |
| GLiNER | Span/label-conditioned extraction architecture and API, rather than a fixed token-label head. |
| `numind/NuNER-v2.0` | Publishes a RoBERTa representation encoder rather than a ready-to-serve `ForTokenClassification` head. |

## Architecture support beyond this catalog

The model list is not exhaustive. Compatible checkpoints also use native
RoBERTa, XLM-RoBERTa, CamemBERT, DistilBERT, Nomic BERT/MoE, JinaCode,
Llama, Mistral/Ministral3, Qwen2, Qwen3/MoE, Qwen3.5 dense/MoE, Gemma3 text,
Gemma4 and experimental DeBERTa-v2/v3 paths. Support depends on architecture,
trained output head, tokenizer and media metadata. Instruction checkpoints can
expose hidden-state embeddings without being trained retrieval models.
Consult [native model coverage](libtorch-model-coverage.md) before substituting
a checkpoint; arbitrary remote custom code is not executed to add a new architecture.
