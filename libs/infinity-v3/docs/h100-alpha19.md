# H100 performance

Infinity V3 alpha.19 on **one H100 80 GB**, with eight CPU threads and 64 GiB RAM.
End-to-end HTTP throughput, request latency and startup time for text embeddings,
reranking, image embeddings and decision requests. Measured October 25–26, 2026;
production latency depends on the traffic mix.

## Text embeddings

FP16 with CUDA graphs. Throughput is **input tokens/s** at concurrency 32.
Latency is the fastest observed 8-token request; startup is the median of three
starts with local weights, excluding downloads.

| Model | 128 tokens | 512 tokens | Min. latency (ms) | Startup (s) |
|---|---:|---:|---:|---:|
| BGE-small | 959,425 | 2,262,919 | 0.71 | 1.41 |
| BGE-large | 435,406 | 648,951 | 1.25 | 1.42 |
| ModernBERT-base | 578,841 | 1,153,286 | 1.12 | 1.47 |
| Qwen3-Embedding-0.6B | 296,344 | 418,911 | 1.50 | 1.66 |
| Qwen3-Embedding-8B | 39,008 | 42,051 | 6.29 | 3.18 |
| EmbeddingGemma2 | 59,401 | 150,284 | 6.69 | 2.45 |
| pplx-embed-v1-4B | 62,968 | 69,945 | 4.40 | 3.15 |
| UltraBERT-Embed-150M | 498,933 | 1,011,166 | 1.20 | 1.45 |

## Compared with vLLM and TEI

Embedding throughput in **input tokens/s**: 512-token inputs, concurrency 32,
FP16, same H100 and server CPU allocation. Infinity uses alpha.19; vLLM 0.32.0
and TEI 1.9.4 results are from the earlier October 2026 benchmark, not a fresh
rerun. Small differences should not be treated as decisive.

| Model | Infinity | vLLM | TEI |
|---|---:|---:|---:|
| BGE-small | 2,262,919 | 362,949 | 2,292,809 |
| BGE-large | 648,951 | 299,205 | 612,727 |
| ModernBERT-base | 1,153,286 | 339,407 | 551,745 |
| Qwen3-Embedding-0.6B | 418,911 | 305,404 | 211,818 |
| Qwen3-Embedding-8B | 42,051 | 41,555 | 29,747 |
| EmbeddingGemma2 | 150,284 | 204,947 | — |
| pplx-embed-v1-4B | 69,945 | 70,348 | — |

— means no comparable measurement. TEI’s Perplexity run used FP32 and quantized
output; EmbeddingGemma2 was unsupported in the tested TEI image. UltraBERT is
omitted because vLLM failed its numerical check and TEI could not load it.

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

**Accuracy:** not evaluated on a task benchmark. † Qwen3-VL BF16 failed one
numerical parity check (0.998545 cosine; threshold 0.999). Elephant passed the
sampled choice checks but showed probability differences up to 1.87 percentage
points. Validation details and reference outputs are described in the data file.
