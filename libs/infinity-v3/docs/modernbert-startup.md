# ModernBERT startup draft measurements

The draft now uses one **32 MiB** default staging budget for every shard, with the existing explicit override. Earlier measurements below used the provisional 16/256 MiB size policy; the final comparison is in the uniform-pool section.

Goal: less than one second from process launch to successful HTTP health, with weights already local. Full configured warmup remains required before readiness; no warmup limits or execution were changed.

## Method

Baseline: main `286871abd0f454ef4ce3f45f4304b85eb7973432`. Candidate: the startup overlap and small-shard staging changes in this PR.

Ten alternating fresh-process runs per binary on one H100 80 GB, CUDA 13.0, LibTorch 2.14.1, release Rust/native builds. Both used the same native CUDA bridge containing the preflight; subsequent main changes were formatting/documentation. Timing-only process probes were present in both Rust executables.

Checkpoint: `answerdotai/ModernBERT-base`, revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, original FP32 safetensors (598,620,928 tensor bytes), BF16 execution, mean pooling. No checkpoint conversion or download is included.

Identical settings: `--dtype bfloat16 --torch-device cuda --device-id 0 --pooling mean --max-batch-tokens 16384 --tokenization-workers 8 --json-output`. CUDA graphs and dynamic FP8 disabled. `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=8`, `TOKENIZERS_PARALLELISM=false`, `CUDA_MODULE_LOADING=LAZY`. Reader overrides unset: baseline uses 256 MiB; candidate uses 16 MiB. Default direct I/O requested in both. Each process reaches Ready, passes HTTP health, produces embeddings, then shuts down before the next starts.

The node/storage had already been exercised. These are fresh-process results, not controlled first starts on newly provisioned nodes. Other services shared the host. Two failure-path checks ran near the beginning of the series and performed device discovery without loading weights; this is another source of noise.

## HTTP readiness results

| Run | Baseline seconds | Candidate seconds |
|---|---:|---:|
| 1 | 1.213 | 0.937 |
| 2 | 1.108 | 1.027 |
| 3 | 1.064 | 1.078 |
| 4 | 1.101 | 0.958 |
| 5 | 1.072 | 0.962 |
| 6 | 1.080 | 0.934 |
| 7 | 1.062 | 0.960 |
| 8 | 1.072 | 1.106 |
| 9 | 1.074 | 3.001 |
| 10 | 2.111 | 0.973 |

Baseline median: **1.077 s**, range 1.062–2.111 s; 0/10 below one second.
Candidate median: **0.967 s**, range 0.934–3.001 s; 6/10 below one second.
Median reduction: about **10.2%**. The target is met at the median on this node; a consistent sub-second bound and new-node performance are not established. Tail latency remains unresolved.

Median reader time fell from **148 ms to 74 ms**. Candidate configuration validation took **274 ms** and frontend setup **54 ms**; those phases run concurrently and must not be added. CUDA preflight remained about **151 ms** in both variants. These per-phase medians are not one representative run.

All compared embeddings were exactly equal (maximum absolute difference 0). Both an invalid second model device and a missing tokenizer returned startup errors without starting checkpoint loading; validation was drained in both cases.

An earlier exploratory process took 49 seconds to start, and isolated first CUDA device discovery took 22 seconds while cached calls took microseconds. Those observations used a different probe series and are not included in the table. They remain evidence that runtime/driver initialization can dominate first-start latency.

## Remaining work

- Repeat on a newly provisioned node, separating image availability, process startup, and runtime discovery.
- Explain discovery/preflight and host-load outliers before claiming a tail-latency target.
- Broaden small-shard measurements across storage and GPUs before promoting the new default.

## Larger-checkpoint regression smoke

Three alternating starts per binary also loaded Qwen3-8B BF16, revision `b968826d9c46dd6066d109eabc6255188de91218`, on the same GPU, with last-token pooling and the same 16K token budget. Every shard retained the 256 MiB default in both binaries. HTTP health and finite 4096-dimensional embeddings passed; all compared outputs were exactly equal.

| Run | Baseline HTTP seconds | Candidate HTTP seconds |
|---|---:|---:|
| 1 | 21.465 | 12.909 |
| 2 | 3.324 | 3.221 |
| 3 | 3.138 | 3.007 |

The large first-start storage variation makes this a correctness/regression smoke, not evidence of a storage-independent speedup.

## Qwen3-8B: explicit 16 MiB versus 256 MiB pools

The same candidate binary was tested with explicit reader budgets on an idle H100 (physical GPU 1). Six fresh processes per budget, alternating order each round, BF16 checkpoint/execution, 16K token budget and full warmup unchanged. The checkpoint is 16.38 GB across five shards. All embeddings were finite and exactly equal across both budgets.

| Pinned budget | Median checkpoint loading | Median HTTP readiness |
|---|---:|---:|
| 256 MiB | 1.414 s | 2.782 s |
| 16 MiB | 1.235 s | 2.596 s |

The smaller pool was about 13% faster in checkpoint loading and 7% faster overall on this node. The first 256 MiB start took 17.089 s (15.599 s loading weights), with uncontrolled underlying storage caches. Retaining all runs gives the medians above. Excluding the entire initial pair gives 1.384 vs 1.229 s loading and 2.772 vs 2.583 s HTTP readiness, so the repeat-start direction is unchanged. This does not establish which budget is faster on a genuinely cold new node.

A 16 MiB pool is therefore not inherently a large-model throughput regression. This comparison covers one 8B model, one storage setup, and H100; larger checkpoints, dtype conversion, slower PCIe GPUs, and different storage still need coverage before changing the large-shard default.

## Uniform 32 MiB default

A follow-up sweep used explicit 16, 32, and 256 MiB budgets on the same candidate executable, five fresh processes per setting per model. Settings and full 16K warmup were unchanged. Budgets were interleaved and their order reversed each round; 32 MiB remained in the middle. This is still an exercised-node comparison, not a new-node cold-start experiment. Explicit-budget measurements exercise the same reader path as the selected default.

| Model | Budget | Median checkpoint loading | Median HTTP readiness | HTTP range |
|---|---:|---:|---:|---:|
| ModernBERT-base | 16 MiB | 0.070 s | 0.949 s | 0.922–1.880 s |
| ModernBERT-base | **32 MiB** | **0.069 s** | **0.945 s** | 0.930–1.264 s |
| ModernBERT-base | 256 MiB | 0.150 s | 1.025 s | 1.006–1.105 s |
| Qwen3-8B | 16 MiB | 1.146 s | 2.519 s | 2.501–4.323 s |
| Qwen3-8B | **32 MiB** | **1.090 s** | **2.486 s** | 2.406–2.613 s |
| Qwen3-8B | 256 MiB | 1.294 s | 2.665 s | 2.642–3.877 s |

All embeddings were finite and exactly equal within each model across all 15 runs. Four of five 32 MiB ModernBERT runs completed below one second; tail latency remains unresolved.

These results support choosing one 32 MiB default instead of a shard-size cutoff. They do not establish an optimum across storage/GPU types. Explicit 16/64/256 MiB overrides remain available and are covered by the existing options test. Tiny files still cap allocations by their data size in the native reader. The final default passed 20 LibTorch tests (five manual tests ignored); the unchanged router implementation previously passed 78 tests.
