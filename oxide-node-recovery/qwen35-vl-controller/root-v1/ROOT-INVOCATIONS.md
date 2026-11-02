# Root bounded gate (Qwen3.5-VL owning lifecycle)

Canonical packet: `root-v1/ROOT-GATE-V8.json`. Controller:
`root-v1/qwen35_vl_bounded_controller.py` (executable; CPU pins phase is read-only).
Author did not run Docker, GPU, Torch, NVCC, network or a device compile.

## 0. Host pins (CPU only)

```bash
cd /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1
CUDA_VISIBLE_DEVICES='' taskset -c 88-95 \
  python3 root-v1/qwen35_vl_bounded_controller.py --phase pins \
  --media /tmp/oxide-all-model-feature-matrix-v1/qwen3-vl-v3/native-baseline/fixtures/fixtures.json
```

Pinned runtime paths: union124 kernels, canonical26 FA4 bundle, GDN raw/provider/aux.
The controller stages every host runtime directory to an explicit remote path and
checks every `docker cp` return code.

## 1. Harness binary (authorized build executed once)

Preserve `e8bf0596…` and one-grant binary `3780facf…`. The owning-lifecycle binary
was built from source `0c4b3453…`/compile-only fixes; actual SHA is
`21c5603ee6e9eff06dae5a4344d83fba43ee8bd0ed9a31d462417970c0bf297a`. Exact command:

```bash
cd /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1
PATH=/root/.cargo/bin:$PATH CARGO_TARGET_DIR=target-opt3-noLTO-harness \
  CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES='' \
  CARGO_PROFILE_RELEASE_OPT_LEVEL=3 CARGO_PROFILE_RELEASE_LTO=false \
  taskset -c 88-95 cargo build --release --offline \
  --manifest-path consumer/Cargo.toml
```

## 2. SDK CPU preflight (Root/Docker; no GPU)

```bash
python3 root-v1/qwen35_vl_bounded_controller.py --phase cpu --run \
  --output /tmp/oxide-qwen35vl-sdk-output/cpu-postbuild
```

The CPU stage delegates to `qwen35_vl_sdk_controller_v2.py`; every per-case
retrieval checks its return code and fails the report before PASS.

## 3. Paired Native WHOLE + owning Oxide WHOLE + comparison (Root GPU0)

Uses the existing `3780facf…` binary; stages consumer/media/runtime and retrieves
Native, Oxide, and pair outputs.

```bash
python3 root-v1/qwen35_vl_bounded_controller.py --phase paired --run \
  --consumer /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO/release/qwen35-vl-consumer \
  --media /tmp/oxide-qwen35vl-sdk-output/cpu-postbuild/cases/single-cats.media.json \
  --output /tmp/oxide-qwen35vl-sdk-output/paired-postbuild
```

## 4. Owning Backend lifecycle (Root GPU0; after harness build)

Requires the rebuilt `target-opt3-noLTO-harness/release/qwen35-vl-consumer`.
Sequence: baseline submit → hold A/B → drop A uncalled → complete B → changed
shape → invalid media typed preflight → recovery; compares actual vectors.
Normal blocking route only; no held-third or fake Overloaded claim.

```bash
python3 root-v1/qwen35_vl_bounded_controller.py --phase life --run \
  --consumer /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO-harness/release/qwen35-vl-consumer \
  --media /tmp/oxide-qwen35vl-sdk-output/cpu-postbuild/cases/single-cats.media.json \
  --output /tmp/oxide-qwen35vl-sdk-output/life-postbuild
```

No quality, lifetime, fence, or canary result is claimed until Root executes it.
