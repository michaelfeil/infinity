<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Supported models and hardware

The Candle backend is deprecated. For models supported by the native LibTorch
backend, select `--backend libtorch` or `INFINITY_BACKEND=libtorch` with a build
that includes the `libtorch` feature. Existing Candle deployments emit a startup
deprecation warning.

We are continually expanding our support for other model types and plan to include them in future updates.

## Supported embeddings models

infinity_v3 currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT
model with Alibi positions and Mistral, Alibaba GTE, Qwen2 models with Rope positions, ModernBERT, Qwen3, Gemma3, and dense Gemma4 text models.

Below are some examples of the currently supported models:

| MTEB Rank | Model Size             | Model Type     | Model ID                                                                                         |
|-----------|------------------------|----------------|--------------------------------------------------------------------------------------------------|
| 2         | 7.57B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-8B](https://hf.co/Qwen/Qwen3-Embedding-8B)                                 |
| 3         | 4.02B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-4B](https://hf.co/Qwen/Qwen3-Embedding-4B)                                 |
| 4         | 509M                   | Qwen3          | [Qwen/Qwen3-Embedding-0.6B](https://hf.co/Qwen/Qwen3-Embedding-0.6B)                             |
| 6         | 7.61B (Very Expensive) | Qwen2          | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-7B-instruct)             |
| 7         | 560M                   | XLM-RoBERTa    | [intfloat/multilingual-e5-large-instruct](https://hf.co/intfloat/multilingual-e5-large-instruct) |
| 8         | 308M                   | Gemma3         | [google/embeddinggemma-300m](https://hf.co/google/embeddinggemma-300m) (gated)                   |
| 15        | 1.78B (Expensive)      | Qwen2          | [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)         |
| 18        | 7.11B (Very Expensive) | Mistral        | [Salesforce/SFR-Embedding-2_R](https://hf.co/Salesforce/SFR-Embedding-2_R)                       |
| 35        | 568M                   | XLM-RoBERTa    | [Snowflake/snowflake-arctic-embed-l-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-l-v2.0) |
| 41        | 305M                   | Alibaba GTE    | [Snowflake/snowflake-arctic-embed-m-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-m-v2.0) |
| 52        | 335M                   | BERT           | [WhereIsAI/UAE-Large-V1](https://hf.co/WhereIsAI/UAE-Large-V1)                                   |
| 58        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1](https://hf.co/nomic-ai/nomic-embed-text-v1)                       |
| 79        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1.5](https://hf.co/nomic-ai/nomic-embed-text-v1.5)                   |
| N/A       | 1B                     | Ministral3     | [nvidia/Nemotron-3-Embed-1B-BF16](https://hf.co/nvidia/Nemotron-3-Embed-1B-BF16)                 |
| N/A       | 8B                     | Ministral3     | [nvidia/Nemotron-3-Embed-8B-BF16](https://hf.co/nvidia/Nemotron-3-Embed-8B-BF16)                 |
| N/A       | 475M-A305M             | NomicBERT      | [nomic-ai/nomic-embed-text-v2-moe](https://hf.co/nomic-ai/nomic-embed-text-v2-moe)               |
| N/A       | 434M                   | Alibaba GTE    | [Alibaba-NLP/gte-large-en-v1.5](https://hf.co/Alibaba-NLP/gte-large-en-v1.5)                     |
| N/A       | 396M                   | ModernBERT     | [answerdotai/ModernBERT-large](https://hf.co/answerdotai/ModernBERT-large)                       |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)             |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-code](https://hf.co/jinaai/jina-embeddings-v2-base-code)         |

To explore the list of best performing text embeddings models, visit the
[Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## EmbeddingGemma 2 multimodal embeddings

`google/embeddinggemma-2` supports text, images, video, audio, and interleaved
messages on CUDA with BF16 and FlashAttention v2. The model loads its 768-dimensional
output projection and mean pooling automatically. The shared 8,192-token context
covers text and media together. Images reuse Gemma4 preprocessing and vision
encoding; video samples up to 32 frames at 1 fps; WAV/MP3 audio uses mono 16 kHz
log-mel features and the Gemma4 conformer encoder. Audio/video require FFmpeg.

Use string inputs for text and ordered message content parts (`text`, `image_url`,
`video_url`, `input_audio`) for media. `/embed`, `/v1/embeddings`, and `/embed_all`
share this input path. CUDA runtime images include the required media decoders.
CPU, Metal, FP16, and classification execution are unsupported for this model.

## Supported re-rankers and sequence classification models

infinity_v3 supports encoder classification models and native
`LlamaForSequenceClassification`, `Qwen2ForSequenceClassification`,
`Qwen3ForSequenceClassification` and `Qwen3MoeForSequenceClassification`
checkpoints with a `score.weight` head.
Decoder classification selects the final non-padding token and reuses the
existing batching and Radix execution paths.

```bash
infinity_v3 --model-id /path/to/qwen3-sequence-classifier --dtype float16
```

```bash
curl http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"A fully formatted classifier input","raw_scores":true}'
```

`raw_scores=true` returns logits; the existing API applies sigmoid for one output
or softmax for multiple outputs when raw scores are disabled. Single-label heads
also use the existing `/rerank` endpoint. Multi-class classification uses
`/predict`. Decoder classifiers require their checkpoint's prompt formatting;
`/predict` does not insert a chat or reranker template.

For Qwen3-Reranker, use a sequence-classification conversion with `score.weight`
(e.g. the original LM head's `no` and `yes` rows) and the checkpoint's prescribed
query/document prompt. Unconverted `Qwen3ForCausalLM` weights do not automatically
become a classifier. Explicit `id2label` mappings are preserved; omitted mappings use Hugging Face
`LABEL_0`, `LABEL_1`, … defaults from `num_labels` (two when omitted).

Qwen3-MoE also supports Voyage-style embeddings with
`use_bidirectional_attention: true`, a root `linear.weight` projection, and the
checkpoint's trained pooling mode. Bidirectional models disable RadixMLP.
Single-output MoE classifier heads use `/rerank`; multiple-output heads use
`/predict`.

Below are some examples of the currently supported models:

| Task               | Model Type  | Model ID                                                                                                        |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                       |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                         |
| Re-Ranking         | GTE         | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |
| Re-Ranking         | ModernBert  | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

## Supported hardware

infinity_v3 supports can be used on CPU, Turing (T4, RTX 2000 series, ...), Ampere 80 (A100, A30),
Ampere 86 (A10, A40, ...), Ada Lovelace (RTX 4000 series, ...), Hopper (H100),
and Blackwell SM120 (RTX Pro 6000 Blackwell, ...) architectures.

The library does **not** support CUDA compute capabilities < 7.5, which means V100, Titan V, GTX 1000 series, etc. are not supported.

To leverage your GPUs, make sure to install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and use
NVIDIA drivers with CUDA version 12.2 or higher.

Find the appropriate Docker image for your hardware in the following table:

The following are suggested local image tags. Build them from this fork using
[custom containers](custom_container); they are not pre-published upstream images.

| Architecture                        | Image                                                                    |
|-------------------------------------|--------------------------------------------------------------------------|
| CPU                                 | infinity_v3:cpu-latest                    |
| Volta                               | NOT SUPPORTED                                                            |
| Turing (T4, RTX 2000 series, ...)   | infinity_v3:turing-latest (experimental)  |
| Ampere 80 (A100, A30)               | infinity_v3:latest                        |
| Ampere 86 (A10, A40, ...)           | infinity_v3:86-latest                     |
| Ada Lovelace (RTX 4000 series, ...) | infinity_v3:89-latest                     |
| Hopper (H100)                       | infinity_v3:hopper-latest (experimental)  |
| Blackwell SM120 (RTX Pro 6000, ...) | Build `Dockerfile-cuda` with `--build-arg CUDA_COMPUTE_CAP=120` (`sm120-` image family) |

Turing uses packed FlashAttention v1 with float16. ALiBi and sliding-window
attention require FlashAttention v2 on Ampere or newer GPUs.

Padded and ONNX backends are removed. MPNet, DistilBERT classification,
CPU BF16, and Metal are unsupported.

### Experimental packed DeBERTa-v2/v3 (SM90 and SM120)

The Hopper and SM120 CUDA builds include this backend and its native kernels.
The CUDA-all build includes DeBERTa in its SM90 binary only.
It is selected for supported DeBERTa models independently of `ATTN_BACKEND`;
DeBERTa's relative attention has no FA2 fallback in this implementation.

The `experimental-deberta` build feature adds Candle inference for
`model_type: deberta-v2`, including Microsoft's DeBERTa-v3 and mDeBERTa-v3
backbones. It requires the FA4 native bundle built with `--deberta` and
`FA4_NATIVE_LIB_DIR` at build time (plus its shared libraries at runtime).
The current kernel supports SM90 and SM120 builds, FP16/BF16, and head dimension 64.
SM120 exports are verified by offline compilation and native linking; end-to-end
correctness against Transformers is verified on H100/SM90. SM120 runtime
correctness remains untested. SM8x cannot use the upstream custom score hook
required for DeBERTa relative attention.

Tokens remain packed through embeddings, transformer layers, and pooling.
Relative attention uses token-by-bucket tables and a linear relative-position
lookup; no padded batch or sequence-by-sequence attention matrix is built.
The scheduler budgets actual tokens. Relative tables still consume memory
proportional to total tokens, head count, and relative bucket count.

Supports shared/separate position projections, bucketed/clipped relative
positions, optional input position/type embeddings, and per-sequence v2
convolution. Standard sequence/token classification heads are supported;
GLiNER-specific heads and schemas are not provided by this feature. Original
DeBERTa-v1 and masked-LM heads are not implemented. Configurations outside the
supported geometry fail explicitly. This is experimental pending model/task
accuracy qualification; it is not a bitwise replacement for eager attention.

### EmbeddingGemma (Gemma3)

`google/embeddinggemma-300m` uses packed, ragged CUDA inference in BF16.
The automatic dtype is BF16, including when the checkpoint stores FP32 weights.
FP32/FP16 model execution, CPU/Metal, and `USE_FLASH_ATTENTION=false` are rejected;
there is no padded fallback. Internal normalization still accumulates in FP32.
The model requires mean pooling and its two checkpoint-provided Dense projection
modules. Use the checkpoint's named query/document prompts for retrieval.
`ATTN_BACKEND=auto` is the default and retains FA2 for EmbeddingGemma because
its head shape is not supported by the bundled FA4 kernels.

### Qwen3-VL multimodal embeddings

`Qwen/Qwen3-VL-Embedding-2B` supports text and images through the existing
`/embed` and `/v1/embeddings` endpoints, with a Candle CUDA/FlashAttention build,
FP16, and last-token pooling. Send text, a list of independent texts, or one list
of user/assistant messages containing text and user image parts. The text-only
Qwen3 embedding models do not accept images.

Remote images require exact HTTPS hostnames configured with `--image-allowed-hosts`;
inline base64 PNG/JPEG/WebP is also accepted. See the repository README for a complete
request and resource limits. JPEG decoding can differ from Pillow/libjpeg and is
not pixel-identical. Qwen3-VL currently disables RadixMLP and rejects BF16.

## D3 decision model

`vllm-sr/d3` supports text/JSON, images, and video on the LibTorch BF16
backend with the `d3` decision protocol. The `d3` model alias selects that
protocol automatically. Its full-attention layers are noncausal; linear
attention stays causal and the decision readout runs in FP32. See the README's
D3 typed multimodal decisions section for request examples and media limits.
