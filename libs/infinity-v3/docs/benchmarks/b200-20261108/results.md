# Infinity embedding benchmark export — COMPLETE

No aggregate winner or ranking. Failed/diagnostic/missing cells are not ranked.

Data: [results.json](results.json) · [results.csv](results.csv) · [evidence.jsonl.gz](evidence.jsonl.gz) · [artifacts.json](artifacts.json).

## Provenance

- Campaign `b200-20261108-upstream-common-verified`; source `53dd6f2cb3083e72b65fbe2406913bc09ab75952` clean; manifest SHA256 `9e9954cd5383d6458648adfe6daab92103d0ef1a0a3c9e9c27e0a10d5fb7ab7b`.
- Recorded campaign date `2026-11-08` local timezone `America/Los_Angeles` (recorded campaign date, not generation clock).
- Infinity `3.0.2-beta.2` build `3.0.2-beta.2`, binary `f93f747f925ef20f15a7fa8e5d9841d07308c071020d5968bbc1567a59814155`, native libraries `5`.
- Stock runtimes: vLLM `0.33.0`, SGLang `0.5.23`.
- Recorded packages: vLLM record `269fc7884be04288e9bd1e0d01cf9a4a5a8fd98b0e0f29744f611355d7ee8d01` mismatches `0` unrecorded `0` missing `0`; SGLang record `ac395f370c1621b07e6c5436ccdcb67d076a85520ccf315a240b2a7e1169d779` mismatches `0` unrecorded `0` missing `0`.
- Hardware: `NVIDIA B200` 183359 MiB; `AMD EPYC 9575F 64-Core Processor`; 8 server cores/lane; 8 client cores/lane; shared host `True`.
- Method: 3×10s after 3s warm-up, batch 1, dtype bfloat16.

## Completeness

- Expected pairs **90**, observed **90**, missing **0**, complete **True**.
- Terminal statuses `{'client_failed': 1, 'load_failed': 1, 'measured': 88}`.

| Runtime | measured | client_failed | load_failed | checkpoint_unavailable | missing | clean |
|---|---|---|---|---|---|---|
| infinity | 29 | 1 | 0 | 0 | 0 | 68/69 |
| vllm | 30 | 0 | 0 | 0 | 0 | 69/69 |
| sglang | 29 | 0 | 1 | 0 | 0 | 67/69 |

## Full model × length matrix (larger parameters first)

Cells: recomputed server input tok/s / successful req/s and explicit tag. Latency is per-window p50/p95 medians in results.csv/results.json.
C is HTTP concurrency. `rel` compares a runtime with Infinity; `fp32` is the independent reference check. `ELIGIBLE` requires both applicable checks and a clean workload. Infinity's `baseline` flag records its FP32 eligibility. `timing=variable` marks more than twofold variation in window requests/s; numerical eligibility does not establish timing stability. Missing parameter counts and unavailable token rates are labeled n/a.

| Model | Parameters (B) | Tokens | C | Infinity | vLLM | SGLang |
|---|---|---|---|---|---|---|
| `microsoft/harrier-oss-v1-27b` embeddings · diagnostic | 27.01 | 128 | 32 | 18,858.48 tok/s / 147.33 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 23,266.00 tok/s / 181.77 req/s [ELIGIBLE rel=pass fp32=pass] | 14,334.87 tok/s / 112.87 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `microsoft/harrier-oss-v1-27b` embeddings · diagnostic | 27.01 | 512 | 32 | 20,562.67 tok/s / 40.16 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 24,453.61 tok/s / 47.76 req/s [ELIGIBLE rel=pass fp32=pass] | 21,442.82 tok/s / 41.96 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `microsoft/harrier-oss-v1-27b` embeddings · diagnostic | 27.01 | 2048 | 8 | 19,894.06 tok/s / 9.71 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 24,585.39 tok/s / 12.00 req/s [ELIGIBLE rel=pass fp32=pass] | 21,070.52 tok/s / 10.29 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` embeddings · comparison_candidate | 11.77 | 128 | 32 | 40,090.07 tok/s / 313.20 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 49,230.74 tok/s / 384.62 req/s [ELIGIBLE rel=pass fp32=pass] | 25,550.09 tok/s / 199.61 req/s [ELIGIBLE rel=pass fp32=pass] |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` embeddings · comparison_candidate | 11.77 | 512 | 32 | 43,964.89 tok/s / 85.87 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 54,738.79 tok/s / 106.91 req/s [ELIGIBLE rel=pass fp32=pass] | 43,418.44 tok/s / 84.80 req/s [ELIGIBLE rel=pass fp32=pass] |
| `tencent/KaLM-Embedding-Gemma3-12B-2511` embeddings · comparison_candidate | 11.77 | 2048 | 8 | 40,982.06 tok/s / 20.01 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 53,816.73 tok/s / 26.28 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] | 43,102.21 tok/s / 21.05 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` embeddings · diagnostic | 7.61 | 128 | 32 | 49,725.74 tok/s / 385.47 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False timing=variable] | 88,628.35 tok/s / 692.41 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] | 73,527.40 tok/s / 574.43 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` embeddings · diagnostic | 7.61 | 512 | 32 | 58,337.47 tok/s / 113.72 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False timing=variable] | 96,429.04 tok/s / 188.34 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] | 97,817.91 tok/s / 191.05 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` embeddings · diagnostic | 7.61 | 2048 | 8 | tok/s n/a / req/s n/a [DIAGNOSTIC rel=diagnostic fp32=reference_failure baseline=False] | 92,615.79 tok/s / 45.22 req/s [DIAGNOSTIC rel=reference_not_eligible fp32=reference_failure] | 95,678.99 tok/s / 46.72 req/s [DIAGNOSTIC rel=reference_not_eligible fp32=reference_failure] |
| `Qwen/Qwen3-Embedding-8B` embeddings · comparison_candidate | 7.57 | 128 | 32 | 75,381.96 tok/s / 588.92 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 78,533.66 tok/s / 613.54 req/s [ELIGIBLE rel=pass fp32=pass] | 66,528.49 tok/s / 519.75 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-8B` embeddings · comparison_candidate | 7.57 | 512 | 32 | 86,223.23 tok/s / 168.40 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 87,107.66 tok/s / 170.13 req/s [ELIGIBLE rel=pass fp32=pass] | 88,610.94 tok/s / 173.07 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-8B` embeddings · comparison_candidate | 7.57 | 2048 | 8 | 86,164.50 tok/s / 42.07 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 85,001.24 tok/s / 41.50 req/s [ELIGIBLE rel=pass fp32=pass] | 87,016.23 tok/s / 42.49 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Salesforce/SFR-Embedding-2_R` embeddings · comparison_candidate | 7.11 | 128 | 32 | 71,692.56 tok/s / 560.10 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 77,326.65 tok/s / 604.11 req/s [ELIGIBLE rel=pass fp32=pass] | 65,905.19 tok/s / 518.94 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Salesforce/SFR-Embedding-2_R` embeddings · comparison_candidate | 7.11 | 512 | 32 | 85,455.73 tok/s / 166.91 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 84,135.29 tok/s / 164.33 req/s [ELIGIBLE rel=pass fp32=pass] | 87,691.81 tok/s / 171.61 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Salesforce/SFR-Embedding-2_R` embeddings · comparison_candidate | 7.11 | 2048 | 8 | 24,604.04 tok/s / 12.01 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True timing=variable] | 84,827.32 tok/s / 41.42 req/s [ELIGIBLE rel=pass fp32=pass] | 86,722.56 tok/s / 42.37 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Salesforce/SFR-Embedding-Mistral` embeddings · comparison_candidate | 7.11 | 128 | 32 | 33,321.60 tok/s / 260.33 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True timing=variable] | 79,897.06 tok/s / 624.20 req/s [ELIGIBLE rel=pass fp32=pass] | 66,640.81 tok/s / 524.73 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Salesforce/SFR-Embedding-Mistral` embeddings · comparison_candidate | 7.11 | 512 | 32 | 85,966.15 tok/s / 167.90 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 89,061.32 tok/s / 173.95 req/s [ELIGIBLE rel=pass fp32=pass] | 89,412.66 tok/s / 174.98 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `Salesforce/SFR-Embedding-Mistral` embeddings · comparison_candidate | 7.11 | 2048 | 8 | 85,043.45 tok/s / 41.53 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 85,503.36 tok/s / 41.75 req/s [ELIGIBLE rel=pass fp32=pass] | 88,211.84 tok/s / 43.09 req/s [DIAGNOSTIC rel=fail_cosine fp32=fail_cosine] |
| `intfloat/e5-mistral-7b-instruct` embeddings · comparison_candidate | 7.11 | 128 | 32 | 73,708.96 tok/s / 575.85 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 79,642.00 tok/s / 622.20 req/s [ELIGIBLE rel=pass fp32=pass] | 66,074.72 tok/s / 516.21 req/s [ELIGIBLE rel=pass fp32=pass] |
| `intfloat/e5-mistral-7b-instruct` embeddings · comparison_candidate | 7.11 | 512 | 32 | 86,428.93 tok/s / 168.81 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 88,774.63 tok/s / 173.39 req/s [ELIGIBLE rel=pass fp32=pass] | 89,981.66 tok/s / 175.75 req/s [ELIGIBLE rel=pass fp32=pass] |
| `intfloat/e5-mistral-7b-instruct` embeddings · comparison_candidate | 7.11 | 2048 | 8 | 84,988.50 tok/s / 41.50 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 85,849.17 tok/s / 41.92 req/s [ELIGIBLE rel=pass fp32=pass] | 88,182.40 tok/s / 43.06 req/s [ELIGIBLE rel=pass fp32=pass] |
| `perplexity-ai/pplx-embed-v1-4b` embeddings · diagnostic | 4.02 | 128 | 32 | 57,405.33 tok/s / 448.48 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False timing=variable] | 136,920.15 tok/s / 1,069.69 req/s [DIAGNOSTIC rel=pass fp32=reference_failure] | 42,570.87 tok/s / 332.58 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `perplexity-ai/pplx-embed-v1-4b` embeddings · diagnostic | 4.02 | 512 | 32 | 143,694.77 tok/s / 280.65 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False] | 148,271.54 tok/s / 289.59 req/s [DIAGNOSTIC rel=pass fp32=reference_failure] | 105,295.85 tok/s / 205.66 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `perplexity-ai/pplx-embed-v1-4b` embeddings · diagnostic | 4.02 | 2048 | 8 | 132,708.10 tok/s / 64.80 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False] | 134,023.57 tok/s / 65.44 req/s [DIAGNOSTIC rel=pass fp32=reference_failure] | 100,049.50 tok/s / 48.85 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `Qwen/Qwen3-Embedding-4B` embeddings · comparison_candidate | 4.02 | 128 | 32 | 122,194.55 tok/s / 954.64 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 125,644.13 tok/s / 981.59 req/s [ELIGIBLE rel=pass fp32=pass] | 92,523.08 tok/s / 722.84 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-4B` embeddings · comparison_candidate | 4.02 | 512 | 32 | 148,325.11 tok/s / 289.70 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 144,926.79 tok/s / 283.06 req/s [ELIGIBLE rel=pass fp32=pass] | 150,472.28 tok/s / 293.89 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-4B` embeddings · comparison_candidate | 4.02 | 2048 | 8 | 142,431.11 tok/s / 69.55 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 138,580.40 tok/s / 67.67 req/s [ELIGIBLE rel=pass fp32=pass] | 143,244.60 tok/s / 69.94 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` embeddings · diagnostic | 1.78 | 128 | 32 | 266,954.02 tok/s / 2,069.41 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False] | 272,810.06 tok/s / 2,131.33 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] | 121,830.90 tok/s / 951.80 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` embeddings · diagnostic | 1.78 | 512 | 32 | 351,237.38 tok/s / 684.67 req/s [DIAGNOSTIC rel=baseline fp32=reference_failure baseline=False] | 372,693.64 tok/s / 727.92 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] | 344,711.97 tok/s / 673.27 req/s [DIAGNOSTIC rel=fail_cosine fp32=reference_failure] |
| `jinaai/jina-code-embeddings-1.5b` embeddings · diagnostic | 1.54 | 128 | 32 | 268,977.52 tok/s / 2,101.39 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 268,045.02 tok/s / 2,094.10 req/s [ELIGIBLE rel=pass fp32=pass] | load_failed |
| `jinaai/jina-code-embeddings-1.5b` embeddings · diagnostic | 1.54 | 512 | 32 | 340,328.71 tok/s / 664.70 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 363,943.82 tok/s / 710.83 req/s [ELIGIBLE rel=pass fp32=pass] | load_failed |
| `microsoft/harrier-oss-v1-0.6b` embeddings · comparison_candidate | 0.60 | 128 | 32 | 455,670.03 tok/s / 3,559.92 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 312,091.94 tok/s / 2,438.22 req/s [ELIGIBLE rel=pass fp32=pass] | 147,568.22 tok/s / 1,152.88 req/s [ELIGIBLE rel=pass fp32=pass] |
| `microsoft/harrier-oss-v1-0.6b` embeddings · comparison_candidate | 0.60 | 512 | 32 | 788,784.67 tok/s / 1,540.60 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 706,453.24 tok/s / 1,379.79 req/s [ELIGIBLE rel=pass fp32=pass] | 374,868.72 tok/s / 732.17 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-0.6B` embeddings · comparison_candidate | 0.60 | 128 | 32 | 458,657.47 tok/s / 3,583.26 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 315,079.04 tok/s / 2,461.56 req/s [ELIGIBLE rel=pass fp32=pass] | 145,529.34 tok/s / 1,136.95 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Qwen/Qwen3-Embedding-0.6B` embeddings · comparison_candidate | 0.60 | 512 | 32 | 783,658.75 tok/s / 1,530.58 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 709,309.74 tok/s / 1,385.37 req/s [ELIGIBLE rel=pass fp32=pass] | 375,938.41 tok/s / 734.25 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-reranker-v2-m3` rerank · comparison_candidate | 0.57 | 128 | 32 | tok/s n/a / 5,192.30 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 296,467.61 tok/s / 2,316.15 req/s [ELIGIBLE rel=pass fp32=pass] | tok/s n/a / 1,372.56 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-reranker-v2-m3` rerank · comparison_candidate | 0.57 | 512 | 32 | tok/s n/a / 2,322.07 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 648,995.69 tok/s / 1,267.57 req/s [ELIGIBLE rel=pass fp32=pass] | tok/s n/a / 1,103.62 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-l-v2.0` embeddings · comparison_candidate | 0.57 | 128 | 32 | 566,633.30 tok/s / 4,426.82 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 311,753.12 tok/s / 2,435.57 req/s [ELIGIBLE rel=pass fp32=pass] | 137,759.97 tok/s / 1,076.25 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-l-v2.0` embeddings · comparison_candidate | 0.57 | 512 | 32 | 1,176,441.95 tok/s / 2,297.74 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 800,265.15 tok/s / 1,563.02 req/s [ELIGIBLE rel=pass fp32=pass] | 446,064.33 tok/s / 871.22 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-large-en-v1.5` embeddings · comparison_candidate | 0.34 | 128 | 32 | 569,262.85 tok/s / 4,447.37 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 312,505.65 tok/s / 2,441.45 req/s [ELIGIBLE rel=pass fp32=pass] | 146,582.87 tok/s / 1,145.18 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-large-en-v1.5` embeddings · comparison_candidate | 0.34 | 512 | 32 | 1,155,861.70 tok/s / 2,257.54 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 724,467.66 tok/s / 1,414.98 req/s [ELIGIBLE rel=pass fp32=pass] | 360,989.70 tok/s / 705.06 req/s [ELIGIBLE rel=pass fp32=pass] |
| `mixedbread-ai/mxbai-embed-large-v1` embeddings · comparison_candidate | 0.34 | 128 | 32 | 554,582.45 tok/s / 4,332.68 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 311,558.15 tok/s / 2,434.05 req/s [ELIGIBLE rel=pass fp32=pass] | 142,879.03 tok/s / 1,116.24 req/s [ELIGIBLE rel=pass fp32=pass] |
| `mixedbread-ai/mxbai-embed-large-v1` embeddings · comparison_candidate | 0.34 | 512 | 32 | 1,184,403.67 tok/s / 2,313.29 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 708,120.83 tok/s / 1,383.05 req/s [ELIGIBLE rel=pass fp32=pass] | 346,735.09 tok/s / 677.22 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-l` embeddings · comparison_candidate | 0.33 | 128 | 32 | 563,110.64 tok/s / 4,399.30 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 310,238.54 tok/s / 2,423.74 req/s [ELIGIBLE rel=pass fp32=pass] | 143,279.21 tok/s / 1,119.37 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-l` embeddings · comparison_candidate | 0.33 | 512 | 32 | 1,163,652.07 tok/s / 2,272.76 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 725,578.13 tok/s / 1,417.14 req/s [ELIGIBLE rel=pass fp32=pass] | 362,197.85 tok/s / 707.42 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-base-en-v1.5` embeddings · comparison_candidate | 0.11 | 128 | 32 | 707,932.92 tok/s / 5,530.73 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 307,520.11 tok/s / 2,402.50 req/s [ELIGIBLE rel=pass fp32=pass] | 157,881.81 tok/s / 1,233.45 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-base-en-v1.5` embeddings · comparison_candidate | 0.11 | 512 | 32 | 2,783,914.13 tok/s / 5,437.33 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 722,297.47 tok/s / 1,410.74 req/s [ELIGIBLE rel=pass fp32=pass] | 378,365.89 tok/s / 739.00 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-m` embeddings · comparison_candidate | 0.11 | 128 | 32 | 708,169.14 tok/s / 5,532.57 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 311,332.72 tok/s / 2,432.29 req/s [ELIGIBLE rel=pass fp32=pass] | 157,051.51 tok/s / 1,226.96 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-m` embeddings · comparison_candidate | 0.11 | 512 | 32 | 2,801,310.40 tok/s / 5,471.31 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 734,828.32 tok/s / 1,435.21 req/s [ELIGIBLE rel=pass fp32=pass] | 377,954.70 tok/s / 738.19 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-m-v1.5` embeddings · comparison_candidate | 0.11 | 128 | 32 | 707,887.31 tok/s / 5,530.37 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 306,257.05 tok/s / 2,392.63 req/s [ELIGIBLE rel=pass fp32=pass] | 153,236.19 tok/s / 1,197.16 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-m-v1.5` embeddings · comparison_candidate | 0.11 | 512 | 32 | 2,813,764.34 tok/s / 5,495.63 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 179,432.49 tok/s / 350.45 req/s [ELIGIBLE rel=pass fp32=pass timing=variable] | 354,276.33 tok/s / 691.95 req/s [ELIGIBLE rel=pass fp32=pass] |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` rerank · comparison_candidate | 0.03 | 128 | 32 | tok/s n/a / 14,189.44 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 258,769.26 tok/s / 2,021.63 req/s [ELIGIBLE rel=pass fp32=pass] | tok/s n/a / 1,721.47 req/s [DIAGNOSTIC rel=fail_values fp32=fail_values] |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` rerank · comparison_candidate | 0.03 | 512 | 32 | tok/s n/a / 8,684.85 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 462,228.49 tok/s / 902.79 req/s [DIAGNOSTIC rel=fail_values fp32=fail_values] | tok/s n/a / 885.17 req/s [DIAGNOSTIC rel=fail_values fp32=fail_values] |
| `BAAI/bge-small-en-v1.5` embeddings · comparison_candidate | 0.03 | 128 | 32 | 1,071,220.81 tok/s / 8,368.91 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 319,257.44 tok/s / 2,494.20 req/s [ELIGIBLE rel=pass fp32=pass] | 188,574.90 tok/s / 1,473.24 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-small-en-v1.5` embeddings · comparison_candidate | 0.03 | 512 | 32 | 4,214,463.37 tok/s / 8,231.37 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 725,546.40 tok/s / 1,417.08 req/s [ELIGIBLE rel=pass fp32=pass] | 423,972.85 tok/s / 828.07 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-s` embeddings · comparison_candidate | 0.03 | 128 | 32 | 1,041,718.28 tok/s / 8,138.42 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 317,240.13 tok/s / 2,478.44 req/s [ELIGIBLE rel=pass fp32=pass] | 186,634.45 tok/s / 1,458.08 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-s` embeddings · comparison_candidate | 0.03 | 512 | 32 | 4,201,830.51 tok/s / 8,206.70 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 722,479.48 tok/s / 1,411.09 req/s [ELIGIBLE rel=pass fp32=pass] | 419,288.14 tok/s / 818.92 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-small-zh-v1.5` embeddings · comparison_candidate | 0.02 | 128 | 32 | 915,806.12 tok/s / 7,154.74 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 299,674.46 tok/s / 2,341.21 req/s [ELIGIBLE rel=pass fp32=pass] | 173,164.93 tok/s / 1,352.85 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-small-zh-v1.5` embeddings · comparison_candidate | 0.02 | 512 | 32 | 3,672,825.54 tok/s / 7,173.49 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 905,182.26 tok/s / 1,767.93 req/s [ELIGIBLE rel=pass fp32=pass] | 461,821.70 tok/s / 902.00 req/s [ELIGIBLE rel=pass fp32=pass] |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank · comparison_candidate | 0.02 | 128 | 32 | tok/s n/a / 17,468.33 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 251,225.68 tok/s / 1,962.70 req/s [ELIGIBLE rel=pass fp32=pass] | tok/s n/a / 1,645.93 req/s [ELIGIBLE rel=pass fp32=pass] |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank · comparison_candidate | 0.02 | 512 | 32 | tok/s n/a / 11,011.90 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 455,994.80 tok/s / 890.61 req/s [ELIGIBLE rel=pass fp32=pass] | tok/s n/a / 860.78 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-xs` embeddings · comparison_candidate | 0.02 | 128 | 32 | 1,082,639.03 tok/s / 8,458.12 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 311,722.07 tok/s / 2,435.33 req/s [ELIGIBLE rel=pass fp32=pass] | 172,700.85 tok/s / 1,349.23 req/s [ELIGIBLE rel=pass fp32=pass] |
| `Snowflake/snowflake-arctic-embed-xs` embeddings · comparison_candidate | 0.02 | 512 | 32 | 4,292,978.53 tok/s / 8,384.72 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 728,997.62 tok/s / 1,423.82 req/s [ELIGIBLE rel=pass fp32=pass] | 412,273.63 tok/s / 805.22 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-base-zh-v1.5` embeddings · comparison_candidate | n/a | 128 | 32 | 704,208.33 tok/s / 5,501.63 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 311,278.61 tok/s / 2,431.86 req/s [ELIGIBLE rel=pass fp32=pass] | 163,639.07 tok/s / 1,278.43 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-base-zh-v1.5` embeddings · comparison_candidate | n/a | 512 | 32 | 2,804,617.31 tok/s / 5,477.77 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 827,635.50 tok/s / 1,616.48 req/s [ELIGIBLE rel=pass fp32=pass] | 406,428.20 tok/s / 793.81 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-large-zh-v1.5` embeddings · comparison_candidate | n/a | 128 | 32 | 561,307.19 tok/s / 4,385.21 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 314,413.20 tok/s / 2,456.35 req/s [ELIGIBLE rel=pass fp32=pass] | 150,966.37 tok/s / 1,179.42 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-large-zh-v1.5` embeddings · comparison_candidate | n/a | 512 | 32 | 1,179,703.95 tok/s / 2,304.11 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 814,730.34 tok/s / 1,591.27 req/s [ELIGIBLE rel=pass fp32=pass] | 389,453.76 tok/s / 760.65 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-m3` embeddings · comparison_candidate | n/a | 128 | 32 | 564,211.77 tok/s / 4,407.90 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 314,808.16 tok/s / 2,459.44 req/s [ELIGIBLE rel=pass fp32=pass] | 145,113.14 tok/s / 1,133.70 req/s [ELIGIBLE rel=pass fp32=pass] |
| `BAAI/bge-m3` embeddings · comparison_candidate | n/a | 512 | 32 | 1,164,359.02 tok/s / 2,274.14 req/s [ELIGIBLE rel=baseline fp32=pass baseline=True] | 798,959.48 tok/s / 1,560.47 req/s [ELIGIBLE rel=pass fp32=pass] | 444,892.46 tok/s / 868.93 req/s [ELIGIBLE rel=pass fp32=pass] |

## Independent FP32 smoke coverage

- Reference SHA `3f4b7461aabcf868452fc100f11c39c89a8bdd9f5994905a62f09271923d624b`; helper `1319c6f97f99f8bebdabd4dbebeb9817acb4446fd467c0b28d294289dbe338b0`; client `191ab22cd0805a095c2c88a92712f5d4f081ad7a6f6bcc42e52cb2f52a3543e4`; manifest `9e9954cd5383d6458648adfe6daab92103d0ef1a0a3c9e9c27e0a10d5fb7ab7b`; device `cuda:0`, dtype `float32`.
- References `61`, issues `0`, failures `8`, runtime cases `207`, FP32 passes `167`, fully eligible `167`; statuses `{'case_missing': 2, 'fail_cosine': 11, 'fail_values': 3, 'pass': 167, 'reference_failure': 24}`.
- Independent HF float32 is an additional numerical smoke gate, not task-quality, retrieval-quality, or ranking evidence.


## Observed timing variability

The following cases have more than twofold variation in requests/s across their three windows. This descriptive flag does not change numerical eligibility. Their aggregates retain the recorded slowdown and should not be used to claim a controlled speedup. The cause is not established; host activity was uncontrolled.

| Model | Runtime | Tokens | Window requests/s (in order) |
|---|---|---:|---|
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | infinity | 128 | 626.53 / 558.82 / 13.23 |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | infinity | 512 | 11.49 / 179.22 / 170.66 |
| `Salesforce/SFR-Embedding-2_R` | infinity | 2048 | 23.12 / 4.90 / 6.02 |
| `Salesforce/SFR-Embedding-Mistral` | infinity | 128 | 566.62 / 206.62 / 57.92 |
| `Snowflake/snowflake-arctic-embed-m-v1.5` | vllm | 512 | 991.48 / 8.75 / 63.43 |
| `perplexity-ai/pplx-embed-v1-4b` | infinity | 128 | 952.10 / 307.15 / 160.73 |

## Diagnostics

- Comparison statuses `{'candidate_not_eligible': 2, 'fail_cosine': 22, 'fail_values': 3, 'pass': 109, 'reference_not_eligible': 2}`.
- Full per-case status, token rates and latency medians are in results.csv; normalized records are in results.json; probe evidence is in evidence.jsonl.gz.
- Evidence carries its own checksums and full available finite probes. Raw run SHA256 is provenance only; full raw runs/logs are not published.
- BF16 agreement is not FP32 validation or task-quality accuracy.
