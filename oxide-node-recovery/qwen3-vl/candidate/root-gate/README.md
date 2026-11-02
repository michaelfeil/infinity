# Qwen3-VL fresh e582 Root gate

This gate is for the concrete Qwen3-VL composition in this worktree. It does
**not** run Qwen3.5-VL; that author uses `../qwen-vl-shared-plan-v1`.

## Pinned identities

- Public base: `e582ce90eb1544fcff081a495cb8fd4f3c17b112`
- Native b66: `/tmp/native214-fa4-mode20/install/lib/libtei_torch.so`
  - SHA-256 `b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e`
  - cuBLASLt `130200`, Torch `2.14.1+cu130`
- Checkpoint: `Qwen/Qwen3-VL-Embedding-8B` at the cached path in `BUILD.json`
  - 749 tensors, 16,289,587,680 bytes, config SHA
    `1eaa88d8d8824a47304e864c14101fe9efe3286607a7573f27eb723637f0d73d`
- Real fixtures: 8 processor fixtures, including three-frame video, ragged
  three-sequence, empty images and varying shapes.

## 1. Rebuild the two release consumers from this source

```sh
# Oxide consumer
cd libs/infinity-v3
CARGO_NET_OFFLINE=true CUDA_VISIBLE_DEVICES= \
  cargo build --offline --release -p infinity-backend-oxide \
  --example qwen3_vl_oxide_consumer

# Native b66 consumer
cd ../../rust-consumer
CARGO_NET_OFFLINE=true cargo build --offline --release
```

Update the two `release_consumer.sha256` values in `BUILD.json` only after the
builds are reproducible.

## 2. CPU preflight inside the existing container

No `docker run`, image, bind mount, GPU context or Torch CUDA.

```sh
CUDA_VISIBLE_DEVICES='' python3 root-gate/preflight_actual.py \
  --build root-gate/BUILD.json \
  --packet root-gate/native/Root-control-packet.native.json \
  --fixtures root-gate/native/fixtures.json \
  --checkpoint /workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B \
  --b66 /tmp/native214-fa4-mode20/install/lib/libtei_torch.so \
  --torch-python /opt/torch-sdk/bin/python3.12 \
  --vendor /tmp/oxide-host-cublas-130200 \
  --checkpoint-hashes root-gate/checkpoint-sha256.json \
  --native-consumer <native-release> \
  --oxide-consumer <oxide-release> \
  --out root-gate/container-preflight.json
```

The host-only subset (checkpoint hashes/tensors/config/fixtures/consumer
identities) already passed and is recorded in `host-preflight.json`.

The delivered `native/fixtures.json` uses host-absolute pixel paths. When
running inside the container, stage `native-baseline/fixtures/` and rewrite the
paths first:

```sh
python3 root-gate/rewrite_fixtures.py   root-gate/native/fixtures.json /remote/fixtures   /remote/fixtures.json
```


## 3. Coupled quality

Native production defaults keep `_cuda_graphs=true` and
`_media_cuda_graphs=false`. The graph-evidence packet is a separate arm.

```sh
NATIVE=<native-release>
OXIDE=<oxide-release>
B66=/tmp/native214-fa4-mode20/install/lib/libtei_torch.so
CKPT=/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B
OUT=/tmp/qwen-vl-gate
mkdir -p "$OUT"/oxide/{pool0,pool1,pool2,raw}

"$NATIVE" root-gate/native/Root-control-packet.native.json \
  root-gate/native/fixtures.json "$B66" "$OUT/native" \
  --library-sha b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e

"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/oxide/pool0" --pool cls
"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/oxide/pool1" --pool mean
"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/oxide/pool2" --pool last_token
"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/oxide/raw" --raw

python3 root-gate/compare_quality.py \
  --native "$OUT/native/results.json" \
  --oxide "$OUT/oxide" \
  --fixtures root-gate/native/fixtures.json \
  --allow-raw \
  --out root-gate/quality.json
```

Provisional pooled thresholds are `max_abs<=0.01`, `relative_l2<=0.001`,
`cosine>=0.9999`; raw-token thresholds are `0.02`, `0.002`, `0.999`. These are
written into the result. Root may tighten them only by recording the actual
distribution; they must never be silently relaxed.

For ColQwen3 / Nemotron ColEmbed, the Native packet must set `_multi_vector=true`
and the checkpoint must be the actual wrapper checkpoint with its trained
`modules.json`/`score.weight`. A missing projection fails admission.

## 4. Lifetime

```sh
"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/life-a" --iterations 50
"$OXIDE" "$CKPT" root-gate/native/fixtures.json "$OUT/life-b" --iterations 50
```

Run each under `/usr/bin/time -v`; fail on non-finite output, crash, stuck
stream, RSS growth >256 MiB or device-used growth >512 MiB.

## 5. Performance

Run the release Native production consumer and the release Oxide consumer on
the same fixtures. Use 20 warmups and 100 measured iterations. The Oxide policy
is no CUDA graphs; do not disable Native graphs for the Native arm. Admission is
`Oxide p50 <= 1.05x Native p50`; 0.90-0.95x is a goal, not a claim. The current
ColQwen/Nemotron readout projection is host-side; if that arm misses the target,
implement a device DenseHead readout rather than changing the threshold.

## 6. AOT / device bundle

1. Fast probe: `build.sh <out> --architectures sm_90 --only-symbol-probes`;
   require the six additive exports plus `infinity_decoder_rows_bf16`.
2. All three: `OLD90_BUNDLE=<published-old90> CARGO_OXIDE=<cargo-oxide>
   bash aot/build_all3.sh <out>`.
3. Require every old 90 export's `.text` and driver resources to be
   byte-identical; six additive Qwen3-VL exports give 96 total. Canonical26 FA4
   and profile indices are untouched. If the old bundle is absent, fail; do not
   fabricate a pass.

## Non-claims

No GPU result, device compile, AOT preservation, quality, lifetime or
performance number is claimed by the author. No public/common mutation. No
Qwen3.5-VL implementation in this worktree.
