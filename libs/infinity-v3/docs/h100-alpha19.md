# H100 performance

Infinity V3 on **one H100 80 GB**, with eight CPU threads and 64 GiB RAM.
End-to-end HTTP throughput, request latency and startup time for text embeddings,
reranking, image embeddings and decision requests. Updated as models are
rebenchmarked; last updated October 30, 2026. Production latency depends on the
traffic mix.

## Text embeddings

FP16 unless marked BF16, with the same server CPU allocation. Throughput is
**input tokens/s** at concurrency 32. Latency is the fastest observed 8-token request;
startup is the median of three fresh starts with local weights (six for Qwen3-8B
FP16), excluding downloads.
Infinity uses the published 3.0.1 image. Comparisons retain the October 25–26
measurements of vLLM 0.32.0 and TEI 1.9.4. Small differences should not be
treated as decisive; build revisions are recorded in the measurement data.

| Model | Runtime | 128 tokens | 512 tokens | Min. latency (ms) | Startup (s) |
|---|---|---:|---:|---:|---:|
| BGE-small | Infinity | 910,550 | 2,124,043 | 0.73 | 1.59 |
| BGE-small | vLLM | 108,689 | 362,949 | 3.87 | 32.48 |
| BGE-small | TEI | 1,058,801 | 2,292,809 | 0.91 | 7.40 |
| BGE-large | Infinity | 428,049 | 644,783 | 1.25 | 1.71 |
| BGE-large | vLLM | 91,839 | 299,205 | 5.02 | 34.69 |
| BGE-large | TEI | 476,558 | 612,727 | 1.70 | 7.80 |
| ModernBERT-base | Infinity | 547,340 | 1,148,797 | 1.13 | 1.69 |
| ModernBERT-base | vLLM | 101,425 | 339,407 | 4.79 | 37.48 |
| ModernBERT-base | TEI | 385,416 | 551,745 | 1.90 | 10.18 |
| Qwen3-Embedding-0.6B | Infinity | 295,543 | 422,132 | 1.51 | 1.79 |
| Qwen3-Embedding-0.6B | Infinity (BF16) | 298,844 | 432,252 | 1.49 | 1.94 |
| Qwen3-Embedding-0.6B | vLLM | 93,012 | 312,847 | 6.13 | 64.66 |
| Qwen3-Embedding-0.6B | TEI | 166,755 | 207,355 | 2.96 | 9.86 |
| Qwen3-Embedding-8B | Infinity | 37,065 | 42,093 | 6.29 | 12.25 |
| Qwen3-Embedding-8B | Infinity (BF16) | 38,604 | 43,548 | 6.94 | 3.51 |
| Qwen3-Embedding-8B | vLLM | 38,442 | 41,387 | 10.80 | 84.40 |
| Qwen3-Embedding-8B | TEI | 28,422 | 30,682 | 8.46 | 13.03 |
| EmbeddingGemma2 | Infinity | 230,180 | 424,694 | 2.21 | 2.49 |
| EmbeddingGemma2 | vLLM | 62,830 | 212,170 | 28.58 | 72.14 |
| pplx-embed-v1-4B | Infinity | 62,239 | 70,066 | 4.59 | 3.53 |
| pplx-embed-v1-4B | vLLM | 57,574 | 70,348 | 7.40 | 68.65 |
| UltraBERT-Embed-150M | Infinity | 424,551 | 1,003,116 | 1.24 | 1.67 |

Only comparable measurements are shown. TEI’s Perplexity run used FP32 and
quantized output; EmbeddingGemma2 and UltraBERT could not load in the tested TEI
image. UltraBERT’s vLLM run failed its numerical check.

## Reranking

Input tokens/s, counting the query, document and special tokens together.

| Model | Precision | 128-token pair | 512-token pair |
|---|---|---:|---:|
| DeBERTa-v3-base | FP16 | 735,960 | 593,844 |
| Nemotron 3.5 Rerank 8B | BF16 | 40,912 | 41,851 |
| Nemotron 3.5 Rerank 8B | BF16 + FP8 linears | 70,290 | 71,965 |

FP8 uses `--enable-fp8-dynamic`. Its numerical check is a two-document smoke
test, not a ranking-quality evaluation.

## Image embeddings

Qwen3-VL-Embedding-8B, one square PNG per request. Images are sent inline,
so timings exclude external downloads. Token counts include vision placeholders
and are not directly comparable to text throughput.

| Image size | Input | FP16 images/s | BF16 images/s† | FP16 tokens/s | BF16 tokens/s† |
|---|---|---:|---:|---:|---:|
| 256px | Image | 255.88 | 260.93 | 22,006 | 22,440 |
| 256px | Image + text | 245.27 | 249.65 | 23,301 | 23,717 |
| 1024px | Image | 20.92 | 21.37 | 21,884 | 22,353 |
| 1024px | Image + text | 20.97 | 21.36 | 22,119 | 22,538 |

## Decision requests

Elephant with dynamic FP8 linears and a BF16 backbone. Each request shares one
state across its yes/no questions; state length below is in **words**, not tokens.

| State words | Questions/request | Requests/s |
|---:|---:|---:|
| 8 | 1 | 553.60 |
| 128 | 1 | 507.82 |
| 512 | 3 | 221.57 |
| 512 | 12 | 58.39 |
| 2048 | 3 | 49.26 |

## Measurement notes

Throughput uses two five-second windows after warmup. Reranking, image and
decision rows report the best median across tested concurrency levels.
Text embeddings use CUDA graphs, JSON float outputs, a 16K batch-token budget
and no Radix MLP; DeBERTa uses 8K. Startup storage caches were uncontrolled.
Qwen3-8B FP16 starts ranged from 3.4 to 24.6 seconds. GPU placement is recorded in the measurement data.
[Measurement data](h100-alpha19.json) contains mixed-length workloads, individual
runs, checkpoint names, commit SHAs and numerical checks.

**Accuracy:** these are timing benchmarks, not task-accuracy evaluations.
† Qwen3-VL BF16's minimum cosine against the original numerical reference was
0.998545 (screen threshold 0.999). Elephant’s sampled choices and probabilities
matched the previous benchmark. Numerical checks are included in the data file.
