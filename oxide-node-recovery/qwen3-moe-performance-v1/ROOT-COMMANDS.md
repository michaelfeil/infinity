# Minimum Root commands

## 1. CPU preflight

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v1
taskset -c 96-103 python3 controller/run_paired.py --cpu-only \
  --output /tmp/oxide-qwen3-moe-paired-perf-cpu-v1
```

This verifies the public 252789b commit, model/config, cases, Native gold,
helper/header pins, and reports missing Native SDK if it is not staged at the
exact path. Author preflight is in `evidence/cpu-preflight/`.

## 2. Root optimized Oxide perf binary (one build, when granted)

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v1/oxide-perf
taskset -c 112-119 env CUDA_VISIBLE_DEVICES= CARGO_NET_OFFLINE=true \
  CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/tmp/oxide-qwen3-moe-paired-perf-target \
  cargo build --release --locked --offline -j4 --bin oxide_perf
```

Write the actual binary/profile/source pins to a `BUILD-PERF.json` and pass it
with `--oxide-build`. This must target public 252789b.

## 3. Paired GPU run

```bash
cd /tmp/oxide-qwen3-moe-paired-performance-v1
taskset -c 112-119 python3 controller/run_paired.py \
  --output /tmp/oxide-qwen3-moe-paired-perf-root-v1 \
  --oxide-binary /tmp/oxide-qwen3-moe-paired-perf-target/release/oxide_perf \
  --oxide-build /path/to/BUILD-PERF.json \
  --native-sdk /tmp/native214-fa4-mode20/install/lib/libtei_torch.so \
  --kernel-root /tmp/oxide-moe-root629-v8-public126-bundle-v1 \
  --provider-root /tmp/oxide-fa4-canonical26-derived-v4/bundle
```

The controller enforces window order `N,O,O,N`, one engine at a time, GPU0,
Native graphs/defaults and Oxide no graphs. Drop `stable2k` with
`--skip-stable2k` if it does not fit. `--iterations` above 200 requires an
explicit `--justify-large` note.
