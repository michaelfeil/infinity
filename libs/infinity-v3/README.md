# Infinity V3 ♾️

Infinity V3 is a high-throughput, low-latency REST API for serving embeddings,
reranking, classification and typed decision models. The server is written in
Rust and uses **LibTorch, the C++ interface of PyTorch**, for model inference.

**Looking for Python / v2?** See the [Python server README](../infinity_emb/README.md)
and [v2 documentation](https://michaelfeil.eu/infinity/main/deploy/).
The `infinity-emb` Python package and `infinity_emb v2` command belong to that server.

## History

- **2026/10:** Native Rust/LibTorch v3 server for embeddings, reranking, classification and typed decisions.
- **2025/07:** Blackwell support in the Python server.
- **2024:** Python / v2 gained multi-model and multimodal serving, Python clients, and broader GPU, CPU, ONNX and TensorRT deployment options.
- **2023/10:** Initial Infinity release.

See the [Python / v2 README](../infinity_emb/README.md) for the full earlier updates.

<p align="center">
  <a href="https://github.com/basetenlabs/truss-examples/tree/7025918c813d08d718b8939f44f10651a0ff2c8c/custom-server/infinity-embedding-server"><img src="https://avatars.githubusercontent.com/u/54861414" alt="Logo Baseten.co" width="50"/></a>
  <a href="https://github.com/runpod-workers/worker-infinity-embedding"><img src="https://github.com/user-attachments/assets/24f1906d-31b8-4e16-a479-1382cbdea046" alt="Logo Runpod" width="50"/></a>
  <a href="https://www.truefoundry.com/cognita"><img src="https://github.com/user-attachments/assets/1b515b0f-2332-4b12-be82-933056bddee4" alt="Logo TrueFoundry" width="50"/></a>
  <a href="https://vast.ai/article/serving-infinity"><img src="https://github.com/user-attachments/assets/8286d620-f403-48f5-bd7f-f471b228ae7b" alt="Logo Vast" width="46"/></a>
  <a href="https://www.dataguard.de"><img src="https://github.com/user-attachments/assets/3fde1ac6-c299-455d-9fc2-ba4012799f9c" alt="Logo DataGuard" width="50"/></a>
  <a href="https://community.sap.com/t5/artificial-intelligence-and-machine-learning-blogs/bring-open-source-llms-into-sap-ai-core/ba-p/13655167"><img src="https://github.com/user-attachments/assets/743e932b-ed5b-4a71-84cb-f28235707a84" alt="Logo SAP" width="47"/></a>
  <a href="https://x.com/StuartReid1929/status/1763434100382163333"><img src="https://github.com/user-attachments/assets/477a4c54-1113-434b-83bc-1985f10981d3" alt="Logo Nosible" width="44"/></a>
  <a href="https://github.com/freshworksinc/freddy-infinity"><img src="https://github.com/user-attachments/assets/a68da78b-d958-464e-aaf6-f39132be68a0" alt="Logo FreshWorks" width="50"/></a>
  <a href="https://github.com/dstackai/dstack/tree/master/examples/deployment/infinity"><img src="https://github.com/user-attachments/assets/9cde2d6b-dc16-4f0a-81ba-535a84321467" alt="Logo Dstack" width="50"/></a>
  <a href="https://embeddedllm.com/blog/"><img src="https://avatars.githubusercontent.com/u/148834374" alt="Logo JamAI" width="50"/></a>
  <a href="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#infinity_emb"><img src="https://avatars.githubusercontent.com/u/1961952" alt="Logo Alibaba Group" width="50"/></a>
  <a href="https://github.com/bentoml/BentoInfinity/"><img src="https://avatars.githubusercontent.com/u/49176046" alt="Logo BentoML" width="50"/></a>
  <a href="https://x.com/bo_wangbo/status/1766371909086724481"><img src="https://avatars.githubusercontent.com/u/60539444" alt="Logo JinaAi" width="50"/></a>
  <a href="https://github.com/dwarvesf/llm-hosting"><img src="https://avatars.githubusercontent.com/u/10388449" alt="Logo Dwarves Foundation" width="50"/></a>
  <a href="https://github.com/huggingface/chat-ui/blob/daf695ea4a6e2d081587d7dbcae3cacd466bf8b2/docs/source/configuration/embeddings.md#openai"><img src="https://avatars.githubusercontent.com/u/25720743" alt="Logo HF" width="50"/></a>
  <a href="https://www.linkedin.com/posts/markhng525_join-me-and-ekin-karabulut-at-the-ai-infra-activity-7163233344875393024-LafB?utm_source=share&utm_medium=member_desktop"><img src="https://avatars.githubusercontent.com/u/86131705" alt="Logo Gradient.ai" width="50"/></a>
</p>

## Performance highlights

* **High throughput**: token-based dynamic batching and dedicated tokenization workers.
* **Memory-efficient batches**: ragged attention avoids padding variable-length inputs, reducing memory waste and OOM risk.
* **Less repeated computation**: from the authors of [RadixMLP](https://arxiv.org/abs/2601.15013), shared-prefix deduplication within a batch for supported decision models and rerankers.
* **Low latency**: BERT-small runs in ~850 µs end-to-end through the Rust/C++ serving path, including ~250 µs HTTP/MessagePack, ~100 µs batching and ~500 µs inference with CUDA graphs.
* **Fast startup**: ModernBERT-base **0.92 s**, Qwen3-8B **2.51 s** on one H100 80 GB.
* **Optimized inference**: native LibTorch kernels, FlashAttention-4 and optional dynamic FP8 for supported models.

Infinity aims to outperform vLLM and SGLang across embedding, reranking and
decision models. Reported October 2026 benchmark highlights versus vLLM:
**~2× throughput on Mammoth**, **~5× on ModernBERT** and **~1.1× on
Qwen3-Embedding-0.6B**. Speedups depend on the workload and configuration.
Startup timings above use local weights; see the
[startup measurements](docs/modernbert-startup.md).

<details>
  <summary>CUDA support and attention kernels</summary>

The NVIDIA image supports Ampere through Blackwell (SM80–SM120), with builds
for the architectures below. FlashAttention-4 is enabled for supported FP16/BF16
shapes; other shapes use existing attention kernels. DeBERTa-v2/v3 uses a novel
CuTeDSL-based algorithm in FA4 to fuse relative-position lookup without
quadratic bucket or attention-bias storage.

| Compute capability | Example GPUs |
| --- | --- |
| SM80 | A100, A30 |
| SM86 | A10 / A10G, GeForce RTX 3090, RTX A6000 |
| SM89 | L4, L40S, RTX 6000 Ada, GeForce RTX 4090 |
| SM90 | H100, H200 |
| SM100 | B200 |
| SM120 | RTX PRO 6000 Blackwell, GeForce RTX 5090 |

FA4 has runtime validation on H100 and compile/link coverage on the other
listed targets. See the [backend documentation](docs/libtorch-backend.md#infinityfa4)
for supported attention shapes and the `INFINITY_TORCH_FA4=0` fallback switch.

FP16 standard BERT CLS embeddings with 384 hidden features / 12 attention heads
or 1024 hidden features / 16 attention heads use the existing native attention,
LayerNorm and cached cuBLASLt projection paths on supported CUDA GPUs (SM80 and
newer), subject to the existing GEMM policy and fallback guards. CUDA graphs
remain opt-in with `INFINITY_TORCH_CUDA_GRAPHS=1`.

</details>

## Getting started

Launch a compatible Hugging Face model or a local checkpoint:

```bash
infinity_v3 --model-id sentence-transformers/all-MiniLM-L6-v2
```

Check `infinity_v3 --help` for CLI options and environment variables. Open `/docs`
on the running server for the interactive API, or `/api-doc/openapi.json` for
its OpenAPI schema. See [running the server](docs/cli.md) and
[building Infinity V3](CONTRIBUTING.md#working-in-the-infinity-monorepo).

### Request service tiers

Add `"service_tier": "priority"` to any inference request, including embeddings,
reranking, classification, and SystemOne decisions:

```json
{"input": ["first text", "second text"], "service_tier": "priority"}
```

Use `"service_tier": "flex"` for lower-priority work. Flex inputs receive a virtual
queue offset (ten seconds by default), allowing recent default/priority work to
pass them while preserving aging and FIFO within each tier. This adds no artificial
wait and does not change billing-token counts. `auto` maps to `default`.

Each model's existing queue orders inputs by queue-worker admission time, subtracting
a configurable bonus for `priority` requests (one second by default). With the
default bonus, a priority input passes normal inputs admitted less than one second
earlier, but stays behind older work. Equal virtual enqueue times
retain queue insertion order. Omitted tiers, `default`, and `auto` use
real enqueue time. Every input in a client batch inherits its request tier;
response ordering is unchanged. Vertex instances inherit the envelope tier
unless they specify their own.

This orders inputs ready for model batching, without adding an actual delay.
It does not preempt dispatched batches or reorder image downloads, tokenization,
or admission limits. Later priority arrivals cannot indefinitely overtake older
normal work. `/tokenize` and `/decode` accept the field for compatibility but do
not use the inference queue and report `default` in the service-tier header. Queue latency metrics continue to use real enqueue
times. Three FIFO buffers compare only their heads: enqueue is O(1) amortized and
next-input selection is O(1), without sorted middle insertions.

Set `INFINITY_BILLING_HEADERS=x-baseten` to include `x-baseten-input-tokens`
and `x-baseten-service-tier` response headers. Any valid HTTP header-name prefix
works; unset or empty disables these headers. The service-tier header contains
`priority`, `flex`, or `default`, reflecting the queue policy actually applied, and is
exposed through CORS for browser clients. For example, an effective flex
request returns `x-baseten-service-tier: flex` with the prefix above. A zero
queue bonus reports `default` even if the request asked for priority. This reports
scheduling eligibility, not a guarantee that another request was overtaken.
Vertex reports a tier only when every instance received it; mixed-tier or empty
batches report `default`. Setting either tier's queue adjustment to zero reports
`default` for that disabled tier.

Set `INFINITY_EXPERIMENTAL_MULTIPLIER=2.0` to report twice the input tokens in
`<prefix>-input-tokens` for effective priority requests. This optional float is unset
by default, skipping multiplication. When set, it must be finite and positive;
fractional token counts round up to whole tokens (for example, `7.2` becomes `8`).
The configured decimal is evaluated exactly, so `100 * 1.1` reports `110`.
Default-tier requests (including priority requests with a zero queue bonus) keep
unmodified counts. `x-compute-tokens`, response-body usage and scheduling are
unchanged. The multiplier applies only when billing headers are enabled.

Set `--priority-queue-bonus-ms 250` for a 250 ms head start, or `0` to disable priority scheduling.
The default is `1000`. `INFINITY_PRIORITY_QUEUE_BONUS_MS` is the environment
variable. One value applies to all models; semicolon-separated values align
with `--model-id`.

Set `--flex-queue-offset-ms 250` or `INFINITY_FLEX_QUEUE_OFFSET_MS` for a
250 ms flex offset. The default is `10000`; `0` disables flex scheduling and reports
`default`. As with the priority bonus, semicolon-separated values align with
`--model-id`. Set both adjustments to `0` for FIFO across all requests.

### CLI overview

Serve an embedding model on one NVIDIA GPU:

```bash
infinity_v3 \
  --model-id Qwen/Qwen3-Embedding-0.6B \
  --served-model-name embeddings \
  --torch-device cuda --device-id 0 --dtype bfloat16 \
  --max-sequence-length 8192 --max-batch-tokens 32768 \
  --hostname 0.0.0.0 --port 8000
```

Clients use `"model": "embeddings"` in requests. For CPU inference, use
`--torch-device cpu --dtype float32`. Omit `--device-id` to use all visible CUDA
GPUs as replicas; each GPU loads a complete model, rather than a shard.

<details>
  <summary>Model, precision and device options</summary>

| Option | Purpose |
| --- | --- |
| `--model-id` | Hugging Face model ID or local checkpoint directory. Repeat to serve multiple models. |
| `--revision` | Pin a Hugging Face commit, tag or branch. |
| `--served-model-name` | Public model name accepted by the API; defaults to the checkpoint ID. |
| `--torch-device` | `auto`, `cpu` or `cuda`. |
| `--dtype` | `auto`, `float32`, `float16` or `bfloat16`; CUDA attention requires FP16/BF16. |
| `--device-id` | Run on one visible accelerator ordinal, such as `0`. |
| `--backend-device-ids` | Select CUDA replicas, such as `0,1`, or `auto`. |
| `--enable-fp8-dynamic` | Enable experimental dynamic FP8 for supported models and GPUs. Off by default; validate task accuracy before enabling. |

</details>

<details>
  <summary>Batching, input limits and model behavior</summary>

| Option | Purpose |
| --- | --- |
| `--max-batch-tokens` | Total token budget per batch; defaults to `16384`. Larger batches need more memory. |
| `--max-sequence-length` | Token limit for one sequence, including prompts and image tokens; bounded by the model and batch limits. |
| `--max-batch-requests` | Optional maximum number of individual requests in a batch. |
| `--max-concurrent-requests` | Admission limit per model; defaults to `512`. |
| `--max-client-batch-size` | Maximum number of inputs in one client batch. |
| `--auto-truncate=false` | Reject oversized inputs instead of truncating. Truncation is enabled by default; decision and multimodal requests have additional restrictions. |
| `--pooling` | Override the checkpoint's embedding pooling configuration when the model supports it. Normally leave unset. |
| `--default-prompt-name` | Select a saved query/document prompt by name. |
| `--default-prompt` | Use a literal prefix instead; cannot be combined with `--default-prompt-name`. |
| `--decision-protocol` | Select `onejev`, `rune`, `clef`, `pplx` or `d3` for a compatible decision checkpoint. |

For example, an `8192` sequence limit with a `32768` batch budget allows several
sequences in a batch while keeping each individual input at or below 8192 tokens.
See [sequence limits](docs/cli.md#sequence-length-and-batch-capacity) for details.

</details>

<details>
  <summary>HTTP, media and observability options</summary>

| Option | Purpose |
| --- | --- |
| `--hostname`, `--port` | HTTP bind address and port; defaults to `0.0.0.0:3000`. |
| `--api-key` | Require `Authorization: Bearer …` on API requests. Can also be supplied through `INFINITY_API_KEY`. |
| `--image-allowed-hosts` | Comma-separated host allowlist replacing the defaults; `--image-allowed-hosts=` permits inline media only. |
| `--image-memory-budget-mib` | Memory budget for encoded images, preprocessing and prepared patches. |
| `--json-output` | Write structured JSON logs. |
| `--prometheus-port` | Prometheus metrics port; defaults to `9000`. |
| `--huggingface-hub-cache` | Model download cache directory. |

</details>

<details>
  <summary>Multiple models and environment variables</summary>

Repeat `--model-id` and align per-model settings in the same order:

```bash
infinity_v3 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --model-id BAAI/bge-reranker-base \
  --served-model-name embeddings --served-model-name reranker \
  --torch-device cuda --dtype bfloat16 \
  --device-id 0 --device-id 1 --port 8000
```

A single per-model setting applies to every model. Multiple values must match
the number of models; semicolon-separated values are also accepted. HTTP bind
settings are shared by the server.

Official builds expose `INFINITY_*` environment variables for CLI options:

```bash
INFINITY_MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 \
INFINITY_TORCH_DEVICE=cpu \
INFINITY_DTYPE=float32 \
INFINITY_PORT=8000 \
infinity_v3
```

Use quotes for lists, such as `INFINITY_DEVICE_ID="0;1"`. CLI arguments take
precedence over environment values. Run `infinity_v3 --help` for the complete
option list and the environment names used by your build.

</details>

### Packed embedding output (opt-in)

For buffer-oriented clients, opt in with `"encoding_format": "tensor"` and
`Accept: application/msgpack`. Responses carry explicit shapes and portable
little-endian float32 bytes, avoiding base64 and per-element numeric encoding.
See the [tensor quickstart and Python/Node.js examples](docs/embedding-tensors.md).
The default numeric-array output remains available for existing clients.

### Supported Tasks and Models by Infinity V3

Click on the sections below to find tasks and **example models**. Support depends
on the checkpoint architecture and trained output head. The models below use
the LibTorch backend. **Dense and Mixture-of-Experts (MoE) architectures are
supported, including Qwen3, Qwen3.5 and Gemma4 MoE models.** Use checkpoints
with a supported task head; each GPU replica must fit the full model, including
all experts.

The [supported model catalog](docs/supported-models.md) groups named checkpoints
by task with launch settings, input requirements and request examples.
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
- **Nomic v1.5**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5), mean pooling with `search_query: ` / `search_document: ` retrieval prefixes.
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
- [nvidia/llama-nemotron-rerank-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2), using its saved average pooling and temperature
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
- **Gemma4 dense and MoE**: native text/image backbones with routed experts for MoE checkpoints. Gemma3 support is text-only.

Backbone support requires a checkpoint with an appropriate trained output head;
it does not make every instruction model a retrieval embedding model.

**Nemotron 3 and 3.5 dense embeddings:** the Ministral3/Mistral3 backbones are
implemented. Supported checkpoints include
[Nemotron 3 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16),
[Nemotron 3 8B](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16),
[Nemotron 3.5 1B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-1B-BF16) and
[Nemotron 3.5 8B](https://huggingface.co/nvidia/Nemotron-3.5-Embed-8B-BF16)
with mean pooling and the checkpoint's query/document instructions.
Nemotron 3 is text-only; 3.5 adds document images. See the
[model catalog](docs/supported-models.md) for launch settings.

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
scoring are unsupported for these models. See [embedding formats and client usage](docs/embedding-tensors.md)
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
- **D3**: [vllm-sr/d3](https://huggingface.co/vllm-sr/d3), [vllm-sr/d3-mini](https://huggingface.co/vllm-sr/d3-mini), and [vllm-sr/d3-ultra](https://huggingface.co/vllm-sr/d3-ultra), dense Qwen3.5 decision models with code-readout heads. Their short aliases select `--decision-protocol d3` automatically. D3 Ultra supports choice, yes/no, and score decisions with shared text/images/video; see the [model catalog](docs/supported-models.md#typed-decisions) for its supported inputs.

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

- [Supported models and configuration](docs/supported-models.md)
- [Performance and Radix MLP](docs/performance.md)
- [CLI and deployment options](docs/cli.md)
- [Native LibTorch backend](docs/libtorch-backend.md)
- [Model coverage and limitations](docs/libtorch-model-coverage.md)
- [API and documentation index](docs/README.md)
- [Licensing](LICENSING.md): Infinity Restricted MIT License 1.0, with retained third-party and earlier grants
