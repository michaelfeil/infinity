# LibTorch startup profiling and staging defaults

## Decision

Use two fixed staging budgets, selected once from the total on-disk size of all checkpoint weight files (including shards and selected sidecars):

- Up to and including **32 GiB**: **32 MiB**.
- Above **32 GiB**: **256 MiB**.
- `INFINITY_TORCH_WEIGHT_READER_BUFFER_BYTES` overrides either choice.

This is one size comparison per model, independent of shard layout, model name, or architecture. It uses filesystem metadata without reading payloads. The cutoff is a practical heuristic between the measured 16.38 GB and 61.07 GB checkpoints, not an experimentally established optimum. Tiny files still cap their allocations in the native reader. The setting bounds the host buffer pool and, separately, transient GPU staging; this experiment does not isolate pinning cost from batching/upload effects.

CUDA discovery/configuration validation overlaps frontend initialization on a blocking worker. All models must pass validation before checkpoint loading starts, and validation is drained on frontend errors. CUDA preflight, health inference, token limits, full configured warmup, and readiness ordering are unchanged.

## Setup

Main baseline: `286871abd0f454ef4ce3f45f4304b85eb7973432`. Release Rust/native, LibTorch 2.14.1, CUDA 13.0, H100 80 GB physical GPU 1. Native bridge includes CUDA preflight; no C++ changes are required by this PR. Weights already local; downloads excluded. BF16 execution, no dynamic FP8 or CUDA graphs, `--max-batch-tokens 16384 --tokenization-workers 8 --json-output`. `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=8`, `TOKENIZERS_PARALLELISM=false`, `CUDA_MODULE_LOADING=LAZY`. Direct I/O requested in every run.

Each fresh process loads, completes the full configured warmup, reaches Ready, passes HTTP health, produces embeddings, and shuts down before the next starts. Explicit 16/32/256 MiB overrides compare the same reader path used by the defaults. The node had been exercised, underlying storage caches were uncontrolled, and other services shared the host. These are not controlled new-node cold-start results.

Pinned checkpoints:

| Model | Revision | Checkpoint |
|---|---|---|
| answerdotai/ModernBERT-base | `8949b909ec900327062f0ebf497f51aef5e6f0c8` | Original FP32 safetensors, 598,620,928 tensor bytes |
| Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` | Original BF16 safetensors, 16,381,516,776 file bytes, 5 shards |
| Qwen/Qwen3-30B-A3B | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` | Original BF16 safetensors, 61,066,575,648 file bytes, 16 shards |

## Final profiling comparison

ModernBERT and Qwen3-8B were rerun after the large-MoE sweep, five fresh processes per budget/model, with rotated/reversed budget order. The large-MoE sweep used three complete rounds (`16,32,256`; `16,256,32`; `256,16,32`); the planned five rounds were stopped once the regression was established. An interrupted fourth-round startup is not a measured result.

| Model | Budget | Median checkpoint loading | Median HTTP readiness | HTTP range |
|---|---:|---:|---:|---:|
| ModernBERT-base | 16 MiB | 0.068 s | 0.952 s | 0.925–1.757 s |
| ModernBERT-base | **32 MiB** | **0.065 s** | **0.921 s** | 0.910–0.954 s |
| ModernBERT-base | 256 MiB | 0.158 s | 1.063 s | 1.005–1.115 s |
| Qwen3-8B | 16 MiB | 1.193 s | 2.543 s | 2.488–21.497 s |
| Qwen3-8B | **32 MiB** | **1.147 s** | **2.505 s** | 2.481–3.687 s |
| Qwen3-8B | 256 MiB | 1.374 s | 2.741 s | 2.705–2.767 s |
| Qwen3-30B-A3B | 16 MiB | 74.080 s | 75.889 s | 75.849–112.772 s |
| Qwen3-30B-A3B | 32 MiB | 66.232 s | 68.117 s | 60.420–80.284 s |
| Qwen3-30B-A3B | **256 MiB** | **50.179 s** | **51.963 s** | 47.081–52.875 s |

All compared embeddings were finite and exactly equal within each model. All five post-MoE explicit 32 MiB ModernBERT runs completed below one second. ModernBERT process shutdown status was recorded and was zero in all 15 post-MoE profiling runs. Automatic-default verification below retains additional tail outliers.

Large-MoE loading samples, retaining every completed run:

| Budget | Checkpoint loading seconds | HTTP readiness seconds |
|---|---|---|
| 16 MiB | 110.685, 74.080, 74.029 | 112.772, 75.849, 75.889 |
| 32 MiB | 78.497, 66.232, 58.656 | 80.284, 68.117, 60.420 |
| 256 MiB | 50.179, 45.242, 51.097 | 51.963, 47.081, 52.875 |

The MoE checkpoint has 18,867 tensors, including 18,432 expert tensors. Larger staging groups plausibly reduce scheduling/upload overhead here, but that effect was not separately profiled. A native test briefly shared the GPU during the first MoE start; later rounds still favored 256 MiB. This is why a uniform 32 MiB default was rejected.

## Baseline and cold-start limitations

The earlier ten-run alternating ModernBERT comparison gave 1.077 s median HTTP readiness for main and 0.967 s for startup overlap with a 16 MiB pool. Ranges were 1.062–2.111 s and 0.934–3.001 s respectively. Reader medians were 148 vs 74 ms. Final profiling above supports 32 MiB for small checkpoints while retaining 256 MiB for large checkpoints.

First-use effects remain material: the first MoE start took 112.772 s, the first 8B start after the MoE sweep took 21.497 s, and an isolated first CUDA device-count probe previously took 22 s. Native-library loading, CUDA discovery/preflight, GEMM first-use state, and underlying storage caches are distinct costs. These observations do not establish a universal new-node latency bound or identify one cache as their cause.

The adaptive policy preserves explicit overrides for deployments whose storage, GPU, dtype conversion, or tensor layout favors a different budget.

## Final adaptive-build validation

With reader overrides unset, the final release binary selected 32 MiB for ModernBERT and all five Qwen3-8B shards, and 256 MiB for all 16 Qwen3-30B-A3B shards. Logged aggregate checkpoint sizes matched 598,635,032, 16,381,516,776 and 61,066,575,648 file bytes respectively. All three completed full 16K warmup, HTTP health and embeddings; outputs exactly matched the explicit-budget profiles and shutdown exit codes were zero. Individual verification starts took 1.079, 2.695 and 49.045 seconds respectively; these are smoke results, not new medians.

Five further ModernBERT starts on the final automatic default took **1.736, 0.934, 0.942, 0.922 and 1.043 seconds**, median **0.942 s**. Three of five completed below one second; all embeddings matched and every shutdown exited zero. These tails reinforce that sub-second median readiness is not a guaranteed cold-start bound.

Invalid second GPU, missing tokenizer, corrupt checkpoint and missing checkpoint weights all returned startup errors with exit code 1; none segfaulted. The first two and missing-weight case did not start checkpoint loading; the corrupt checkpoint failed during loading. Validation was drained in all four cases.

Final CPU suites: 21 LibTorch tests plus 78 router tests passed, with ten manual tests ignored. Boundary/override tests cover the 32 GiB cutoff and explicit buffer sizes; strict library Clippy, rustfmt and diff checks passed. No C++ code, warmup implementation or dependencies changed.
