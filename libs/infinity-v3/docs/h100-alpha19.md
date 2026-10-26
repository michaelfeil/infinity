# H100 benchmark: alpha.19

Measured October 25–26, 2026, using the LibTorch backend on one H100 80 GB.

[Measurement data](h100-alpha19.json) includes every timed window, startup
sample, client-capacity check and the numerical audit. The original HTTP
fixtures, logs and runners remain in the benchmark workspace; this report
and data file are not a standalone reproduction harness.

## Scope and controls

- Candidate: ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0-alpha.19,
  commit cd02c79019fc8c74dc85a8699f0ce952f15426cc.
  Registry digest: sha256:201f8582d498a9cc7fee51aaef4120ae715e0c6c589120e1305f19cfb63ba066.
  Subsequent loading fixes in c2a0e69 are not included.
- Other serving engines were not benchmarked in this run.
- GPU 0, H100 80 GB HBM3, 700 W limit, driver 580.126.20.
  Server: CPU threads 24–31, eight-CPU quota, 64 GiB RAM, 1 GiB shared memory.
  Those eight SMT threads cover four physical cores.
  Runs are sequential; clients use separate cores.
- HTTP JSON float outputs. FP16 text embeddings, CUDA graphs enabled,
  Radix MLP disabled. Batch-token budget: 16,384; DeBERTa uses 8,192.
  Token lengths include special tokens and are checked against server usage.
- Two five-second windows per workload after warmup. Fixed-length embedding
  throughput uses concurrency 32; mixed traffic alternates 8/128/512 tokens and
  selects the best median at concurrency 8/32/128. Multi-process client checks
  are used when the single client saturates. Eight-input requests are also
  recorded in the data.
- Startup: three fresh containers with local weights. Pulls/downloads are
  excluded; filesystem caches are uncontrolled.

These are synthetic throughput measurements, not production SLO capacity.

## Text embeddings

Input tokens/s on alpha.19.

| Model | 128 tokens | 512 tokens | Mixed 8/128/512 |
|---|---:|---:|---:|
| BGE-small | 959,425 | 2,262,919 | 2,482,358 |
| BGE-large | 435,406 | 648,951 | 687,629 |
| ModernBERT-base | 578,841 | 1,153,286 | 1,237,667 |
| Qwen3-Embedding-0.6B | 296,344 | 418,911 | 439,094 |
| Qwen3-Embedding-8B | 39,008 | 42,051 | 42,439 |
| EmbeddingGemma2 | 59,401 | 150,284 | 175,473 |
| pplx-embed-v1-4B | 62,968 | 69,945 | 71,753 |
| UltraBERT-Embed-150M | 498,933 | 1,011,166 | 1,102,613 |

Alpha.19 minimum latency for an 8-token request and median startup:

| Model | Minimum latency (ms) | Startup (s) |
|---|---:|---:|
| BGE-small | 0.71 | 1.41 |
| BGE-large | 1.25 | 1.42 |
| ModernBERT-base | 1.12 | 1.47 |
| Qwen3-Embedding-0.6B | 1.50 | 1.66 |
| Qwen3-Embedding-8B | 6.29 | 3.18 |
| EmbeddingGemma2 | 6.69 | 2.45 |
| pplx-embed-v1-4B | 4.40 | 3.15 |
| UltraBERT-Embed-150M | 1.20 | 1.45 |

## Reranking

Input tokens/s include query, document and special tokens. Each row selects
the best median across concurrency 1/32/128 with one document or concurrency
8 with eight documents.

| Model | Precision | Pair tokens | Input tokens/s |
|---|---|---:|---:|
| DeBERTa-v3-base | FP16 | 128 | 732,448 |
| DeBERTa-v3-base | FP16 | 512 | 592,371 |
| Nemotron 3.5 Rerank 8B | BF16 | 128 | 40,915 |
| Nemotron 3.5 Rerank 8B | BF16 | 512 | 41,564 |

## Qwen3-VL-Embedding-8B with images

One synthetic PNG per request, with or without text. Data URIs remove download
variability. Image clients use CPU cores 16/18/20/22; initial unpinned attempts
are excluded. Concurrency is 1/8/32 for 256px and 1/8 for 1024px: the latter
keeps the default 512 MiB image budget.

| Precision | Input | Images/s | Input tokens/s |
|---|---|---:|---:|
| FP16 | 1024-image | 20.80 | 21,754 |
| FP16 | 1024-text-image | 20.91 | 22,060 |
| FP16 | 256-image | 259.99 | 22,359 |
| FP16 | 256-text-image | 244.62 | 23,239 |
| BF16† | 1024-image | 21.32 | 22,299 |
| BF16† | 1024-text-image | 21.35 | 22,522 |
| BF16† | 256-image | 262.12 | 22,542 |
| BF16† | 256-text-image | 250.13 | 23,762 |

Image token counts include vision placeholders and template tokens; they are
not FLOP-equivalent to text tokens. **† BF16 timings do not establish equal
quality:** one fixture failed the cross-version numerical screen below.

## Elephant decision requests

BF16 backbone with dynamic FP8 linears, /v1/systemone, a 16K batch-token budget
and 8K sequence limit. Synthetic shared states have the indicated word counts;
these are **not total input-token counts**. Each question has yes/no choices.
Throughput selects the best median across concurrency 1/8/32, using two sweeps.

| State words | Questions/request | Requests/s |
|---:|---:|---:|
| 8 | 1 | 508.03 |
| 128 | 1 | 460.94 |
| 512 | 3 | 210.15 |
| 512 | 12 | 55.38 |
| 2048 | 3 | 47.58 |

## Numerical checks and limitations

Numerical checks compare against saved alpha.16 outputs; historical performance
measurements are not included.

- Text embedding comparisons passed the 0.999 cosine screen on the captured
  fixtures. BERT comparisons include fresh-control 128/512-token outputs;
  UltraBERT includes its 512-token local-attention fixture.
- DeBERTa's two-document score sample is identical. Nemotron's maximum sampled
  raw-score difference is 0.002494, with unchanged ranking.
- Qwen3-VL FP16 minimum cosine against alpha.16 is **0.999974**.
  BF16 minimum cosine is **0.998545**, below the **0.999** comparison threshold
  on one of eight fixtures. Its new outputs are closer to the saved vLLM FP16
  outputs, consistent with precision differences, but this does not establish
  task-quality parity. The committed audit intentionally reports passed: false.
- Elephant's sampled choices are unchanged, but probabilities shift by up to
  **1.87 percentage points**.

These synthetic checks do not certify retrieval accuracy or decision-index
quality. No deployment or quality signoff is implied by the timing results.
