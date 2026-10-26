# H100 benchmarks

Infinity V3 alpha.19 on **one H100 80 GB**, with eight CPU threads and 64 GiB RAM.
Measured October 25–26, 2026 through the HTTP API: text embeddings, reranking,
image embeddings and decision requests. These synthetic workloads measure
throughput under load; production latency depends on the traffic mix.

## Text embeddings

FP16 with CUDA graphs. Throughput is **input tokens/s** at concurrency 32.
Latency is the fastest observed 8-token request; startup is the median of three
starts with local weights, excluding downloads.

| Model | 128 tokens | 512 tokens | Latency (ms) | Startup (s) |
|---|---:|---:|---:|---:|
| BGE-small | 959,425 | 2,262,919 | 0.71 | 1.41 |
| BGE-large | 435,406 | 648,951 | 1.25 | 1.42 |
| ModernBERT-base | 578,841 | 1,153,286 | 1.12 | 1.47 |
| Qwen3-Embedding-0.6B | 296,344 | 418,911 | 1.50 | 1.66 |
| Qwen3-Embedding-8B | 39,008 | 42,051 | 6.29 | 3.18 |
| EmbeddingGemma2 | 59,401 | 150,284 | 6.69 | 2.45 |
| pplx-embed-v1-4B | 62,968 | 69,945 | 4.40 | 3.15 |
| UltraBERT-Embed-150M | 498,933 | 1,011,166 | 1.20 | 1.45 |

## Reranking

Input tokens/s, counting the query, document and special tokens together.

| Model | Precision | 128-token pair | 512-token pair |
|---|---|---:|---:|
| DeBERTa-v3-base | FP16 | 732,448 | 592,371 |
| Nemotron 3.5 Rerank 8B | BF16 | 40,915 | 41,564 |

## Image embeddings

Qwen3-VL-Embedding-8B, one square PNG per request. Images are sent inline,
so timings exclude external downloads. Token counts include vision placeholders
and are not directly comparable to text throughput.

| Image size | Input | FP16 images/s | BF16 images/s† | FP16 tokens/s | BF16 tokens/s† |
|---|---|---:|---:|---:|---:|
| 256px | Image | 259.99 | 262.12 | 22,359 | 22,542 |
| 256px | Image + text | 244.62 | 250.13 | 23,239 | 23,762 |
| 1024px | Image | 20.80 | 21.32 | 21,754 | 22,299 |
| 1024px | Image + text | 20.91 | 21.35 | 22,060 | 22,522 |

## Decision requests

Elephant with dynamic FP8 linears and a BF16 backbone. Each request shares one
state across its yes/no questions; state length below is in **words**, not tokens.

| State words | Questions/request | Requests/s |
|---:|---:|---:|
| 8 | 1 | 508.03 |
| 128 | 1 | 460.94 |
| 512 | 3 | 210.15 |
| 512 | 12 | 55.38 |
| 2048 | 3 | 47.58 |

## Measurement notes

Throughput uses two five-second windows after warmup. Reranking, image and
decision rows report the best median across tested concurrency levels.
Text embeddings use JSON float outputs, a 16K batch-token budget and no Radix MLP;
DeBERTa uses 8K. Startup storage caches were uncontrolled.
[Measurement data](h100-alpha19.json) contains mixed-length workloads, individual
runs, checkpoint names, image identity and numerical checks.

**Quality caveat:** these are timing benchmarks, not accuracy evaluations.
Against saved alpha.16 outputs, text embeddings passed the 0.999 cosine screen,
but **† Qwen3-VL BF16 reached 0.998545 on one fixture**. Elephant's sampled choices
were unchanged, with probability shifts up to **1.87 percentage points**.
