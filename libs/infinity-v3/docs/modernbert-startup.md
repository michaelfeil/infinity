# ModernBERT startup draft measurements

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
