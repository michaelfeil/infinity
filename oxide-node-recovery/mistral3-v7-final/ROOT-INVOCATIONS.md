# Root invocations (Mistral3 composer v4, final staged controller)

Packet: `/tmp/oxide-all-model-feature-matrix-v1/mistral3-v5`
Base: current `/tmp/infinity-plain-cuda` at public `55380b7` plus the
V5/Qwen35/MeanV6 six-file overlay in `BASE-PINS.json`.
Patch: `selective.patch` (source-only composer; does not write Root).

## 1. Apply the composer

```sh
cd /tmp/infinity-plain-cuda
git apply --check /tmp/oxide-all-model-feature-matrix-v1/mistral3-v5/selective.patch
git apply         /tmp/oxide-all-model-feature-matrix-v1/mistral3-v5/selective.patch
```

## 2. CPU debug checks only

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v5
bash run_cpu_checks.sh
```

This is debug/default only: no release/fat-LTO all-target build, no GPU,
Docker execution, host Torch, network or device compiler. It covers oxide
library tests, Mistral3, cold factory hook, shaped rope, shared metadata,
404-only downloader policy, router `oxide,http` check/test, host consumer debug
build, native metadata controls and controller source tests.

## 3. Final staged controller

`native/root_staged_controller.py` stages only the allowlisted runner/header/
scripts into the existing `infinity-torch214-compare-sdk` container. It uses
`docker --host=unix:///var/run/docker.sock`, clears only the inherited
`DOCKER_HOST/CONTEXT/TLS/TLS_VERIFY/CERT_PATH` values, preserves
`DOCKER_CONFIG`/proxy/CA/auth, and uses `/opt/torch-sdk/bin/python3.12` inside
the container. Every container command runs under the inner owned
`remote_guard.py` with exact PID/argv/start-time cleanup.

CPU stage (CVD empty; actual 531 metadata + full SDK load + 19-arg CAPI/header
+ vendor GOT 130200 before/after; no engine):

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v5/native
CUDA_VISIBLE_DEVICES='' python3 root_staged_controller.py --stage cpu --run \
  --host-out /tmp/mistral3-v5-native-cpu-fresh
```

Quality/perf stages (same controller, actual b66 CAPI, GPU0 only, independent
review clearance required):

```sh
CUDA_VISIBLE_DEVICES=0 python3 root_staged_controller.py --stage quality --run \
  --review-clearance /path/to/independent-review.json \
  --host-out /tmp/mistral3-v5-native-quality-fresh

CUDA_VISIBLE_DEVICES=0 python3 root_staged_controller.py --stage perf --run \
  --review-clearance /path/to/independent-review.json \
  --iterations 20 --warmup 3 \
  --host-out /tmp/mistral3-v5-native-perf-fresh
```

Pinned container facts: SDK `/tmp/native214-fa4-mode20/install/lib/libtei_torch.so`
sha256 `b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e`;
Torch `2.14.1+cu130`; BLAS GOT owner `/tmp/oxide-host-cublas-130200`; read-only
checkpoint `/bench/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16`;
Native graph defaults 8 captures / 4 entries / 3 reuse / max 4096 tokens /
512 MiB, with actual `tei_graph_count`/`tei_graph_stats` before and after.
Quality/perf copy vectors and manifests with `docker cp` into the fresh host
out directory and record `gpu_execution: true` only after the actual run.
No weights are copied and the SDK/production weights are never modified.

## 4. frozen optimized host consumer

Only the previously granted incremental Rust consumer artifact is used:

```sh
/tmp/oxide-all-model-feature-matrix-v1/mistral3-v5/.work/host-consumer-opt3-v4/release/mistral3-host-consumer
sha256 f0653ddec06f0667f3c36dedf65b8026f85154c3629e224af4d766d4340c3067
```

Do not rebuild it for controller-only fixes. Run on GPU0 only after independent
review, with `CUDA_VISIBLE_DEVICES=0`, `INFINITY_MISTRAL3_RUN_GPU=1`,
`INFINITY_MISTRAL3_MODEL`, `INFINITY_OXIDE_MISTRAL3_OUT_DIR`,
`INFINITY_FA4_ROOT` and `INFINITY_OXIDE_KERNEL_MODULE` set. Use the Root
device-union124 all3 ready path
`/tmp/infinity-oxide-parity-20261101/device-union124-all3-v1/kernels` when
available; canonical26 FA4 is sufficient for the actual H4096 noncausal Mistral3.

## 5. Router oxide-only source gate

```sh
cd /tmp/oxide-all-model-feature-matrix-v1/mistral3-v5/libs/infinity-v3
cargo check --offline -p infinity_v3-router --no-default-features --features oxide,http
cargo test  --offline -p infinity_v3-router --no-default-features --features oxide,http \
  --lib mistral3_params_only_metadata_reaches_the_normal_router_config
```

`params.json` fallback is 404-only for the downloader and NotFound-only for the
local config reader. Malformed/advertised configs, 403/transient errors and
unreadable paths propagate. Missing vision metadata is rejected; no Pixtral
vision is invented. Raw Mistral-common weight prefixes remain an explicit typed
HF-conversion gap.

## 6. Remaining real gaps

- Root pending: actual SDK/container CPU stage, independent review, GPU quality/
  perf, paired Native gold.
- Router is source-checked under `oxide,http`; no LibTorch build in the author
  environment.
- 3B H3072, 14B H5120 and older 2410 trained checkpoints are not local.
- Raw Mistral-common params-only weights require HF conversion.
- No public push/commit, GPU, Docker execution, host Torch, network or device
  compiler was used while authoring.
