# Minimum Root commands

## 1. CPU preflight (host only)

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v2
taskset -c 96-103 python3 controller/run_paired_v2.py --cpu-only \
  --output /tmp/oxide-qwen3-moe-paired-perf-v2-cpu
```

## 2. Native SDK/Torch/vendor CPU-real preflight in Docker

Runs inside the existing container before any GPU timing. It stages support
code/helper/cases via docker cp and removes the remote directory in `finally`.

```bash
taskset -c 112-119 python3 controller/run_paired_v2.py --docker-cpu-only \
  --output /tmp/oxide-qwen3-moe-paired-perf-v2-docker-cpu \
  --native-sdk /tmp/native214-fa4-mode20/install/lib/libtei_torch.so \
  --native-python /opt/torch-sdk/bin/python3.12
```

This must pass before the GPU command.

## 3. One frozen optimized Oxide perf build candidate (when granted)

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v2/oxide-perf
taskset -c 112-119 env CUDA_VISIBLE_DEVICES= CARGO_NET_OFFLINE=true \
  CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/oxide-qwen3-moe-paired-perf-v2-target \
  cargo build --release --locked --offline -j4 --bin oxide_perf
```

Write actual binary/profile/source pins to `BUILD-PERF.json` and pass it below.

## 4. Paired GPU run

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v2
taskset -c 112-119 python3 controller/run_paired_v2.py \
  --output /tmp/oxide-qwen3-moe-paired-perf-v2-root \
  --oxide-binary /tmp/oxide-qwen3-moe-paired-perf-v2-target/release/oxide_perf \
  --oxide-build /path/to/BUILD-PERF.json \
  --native-sdk /tmp/native214-fa4-mode20/install/lib/libtei_torch.so \
  --native-python /opt/torch-sdk/bin/python3.12 \
  --kernel-root /tmp/oxide-moe-root629-v8-public126-bundle-v1 \
  --provider-root /tmp/oxide-fa4-canonical26-derived-v4/bundle
```

Add `--skip-stable2k` only if 2k does not fit. Add `--iterations` beyond 200
only with an explicit `--justify-large` note.
