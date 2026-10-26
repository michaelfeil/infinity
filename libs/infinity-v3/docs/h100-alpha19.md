# H100 performance

Infinity V3 on **one H100 80 GB**, with eight CPU threads and 64 GiB RAM.
End-to-end HTTP throughput, request latency and startup time for text embeddings,
reranking, image embeddings and decision requests. Measured October 25–26, 2026;
production latency depends on the traffic mix.

## Text embeddings

FP16 unless marked BF16, with the same server CPU allocation. Throughput is
**input tokens/s** at concurrency 32. Latency is the fastest observed 8-token request;
startup is the median of three fresh starts with local weights, excluding downloads.
Gemma and Qwen3 were refreshed on October 26 using a local H100 build of main
`c82fb16`, vLLM 0.32.0 and TEI 1.9.4; other rows retain the alpha.19 comparison.
BF16 rows are separate precision measurements.
Small differences should not be treated as decisive.

| Model | Runtime | 128 tokens | 512 tokens | Min. latency (ms) | Startup (s) |
|---|---|---:|---:|---:|---:|
| BGE-small | Infinity | 959,425 | 2,262,919 | 0.71 | 1.41 |
| BGE-small | vLLM | 108,689 | 362,949 | 3.87 | 32.48 |
| BGE-small | TEI | 1,058,801 | 2,292,809 | 0.91 | 7.40 |
| BGE-large | Infinity | 435,406 | 648,951 | 1.25 | 1.42 |
| BGE-large | vLLM | 91,839 | 299,205 | 5.02 | 34.69 |
| BGE-large | TEI | 476,558 | 612,727 | 1.70 | 7.80 |
| ModernBERT-base | Infinity | 578,841 | 1,153,286 | 1.12 | 1.47 |
| ModernBERT-base | vLLM | 101,425 | 339,407 | 4.79 | 37.48 |
| ModernBERT-base | TEI | 385,416 | 551,745 | 1.90 | 10.18 |
| Qwen3-Embedding-0.6B | Infinity | 296,030 | 423,359 | 1.49 | 1.64 |
| Qwen3-Embedding-0.6B | Infinity (BF16) | 297,720 | 431,569 | 1.48 | 1.58 |
| Qwen3-Embedding-0.6B | vLLM | 93,012 | 312,847 | 6.13 | 64.66 |
| Qwen3-Embedding-0.6B | TEI | 166,755 | 207,355 | 2.96 | 9.86 |
| Qwen3-Embedding-8B | Infinity | 37,724 | 41,770 | 6.25 | 3.31 |
| Qwen3-Embedding-8B | Infinity (BF16) | 39,424 | 42,936 | 6.89 | 3.00 |
| Qwen3-Embedding-8B | vLLM | 38,442 | 41,387 | 10.80 | 84.40 |
| Qwen3-Embedding-8B | TEI | 28,422 | 30,682 | 8.46 | 13.03 |
| EmbeddingGemma2 | Infinity | 241,031 | 410,507 | 2.17 | 2.52 |
| EmbeddingGemma2 | vLLM | 62,830 | 212,170 | 28.58 | 72.14 |
| pplx-embed-v1-4B | Infinity | 62,968 | 69,945 | 4.40 | 3.15 |
| pplx-embed-v1-4B | vLLM | 57,574 | 70,348 | 7.40 | 68.65 |
| UltraBERT-Embed-150M | Infinity | 498,933 | 1,011,166 | 1.20 | 1.45 |

Only comparable measurements are shown. TEI’s Perplexity run used FP32 and
quantized output; EmbeddingGemma2 and UltraBERT could not load in the tested TEI
image. UltraBERT’s vLLM run failed its numerical check.

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

Throughput uses two five-second windows after warmup (six for Qwen3-8B BF16
at 128 tokens, including confirmation runs). Reranking, image and
decision rows report the best median across tested concurrency levels.
Text embeddings use JSON float outputs, a 16K batch-token budget and no Radix MLP;
DeBERTa uses 8K. Startup storage caches were uncontrolled.
[Measurement data](h100-alpha19.json) contains mixed-length workloads, individual
runs, checkpoint names, image identity and numerical checks.

**Accuracy:** not evaluated on a task benchmark. Refreshed Gemma/Qwen outputs
passed 14-fixture comparisons against their controls (minimum cosine 0.999998). † Qwen3-VL BF16 failed one
numerical parity check (0.998545 cosine; threshold 0.999). Elephant passed the
sampled choice checks but showed probability differences up to 1.87 percentage
points. Validation details and reference outputs are described in the data file.
