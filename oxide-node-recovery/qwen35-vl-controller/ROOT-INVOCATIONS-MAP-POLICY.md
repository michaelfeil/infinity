# Qwen3.5-VL SDK map + Native graph-policy successor

OwnedControllerRoot: `/tmp/oxide-qwen35vl-sdk-map-policy-v1`
SourceRoot (read-only): `/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1`

This packet changes only the bounded path/role items:

1. Five named checkpoint location records. Host paths are read only for
   config/index SHA validation. Container paths under `/bench/michaelfeil/...`
   are used only inside SDK container argv. No recursive JSON path rewrite.
2. An explicit seven-key Native production CUDA-graph policy in the new helper
   `native/native_whole_model_policy.py`. The frozen eager helper
   `native/native_whole_model.py` is preserved untouched. Oxide remains on its
   normal no-graph path.

Native file routing uses the actual `model.safetensors.index.json` when it
exists, otherwise the exact `model.safetensors` (Elephant single). It validates
the expected config/index SHA and never scans for or fabricates an index.
No checkpoint config, math, weight, threshold or Rust production source changed.
No weights are copied. The two existing absolute host binaries are staged by
path; they are not rebuilt.

## Files

- `qwen35_vl_map_policy_controller.py`: new bounded controller.
- `native/sdk/sdk_map_preflight.py`: new inside-SDK model-first preflight.
- `native/native_whole_model_policy.py`: new Native production-policy helper.
- `device-packet/model-locations-v1.json`: five host/container records.
- `device-packet/model-manifest-v1.json`: actual config/index hashes and weight-entry sizes.
- `device-packet/native-assets-v1.json`: explicit fixture/media host-to-remote records.
- `device-packet/runtime-paths-v1.json`: QV35 canonical26 FA4 + GDN + union124 paths.
- `device-packet/successor-pins-v1.json`: code, model, binary, runtime file pins.
- `root-v1/ROOT-GATE-MAP-POLICY-V1.json`: Root commands and pins.
- `tests/test_map_policy_packet.py`: real CPU behavior tests.
- `READY.json`: truthful readiness and exact Root argv.

## Author CPU tests (already run cold, offline)

```bash
cd /tmp/oxide-qwen35vl-sdk-map-policy-v1
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  taskset -c 88-95 python3 tests/test_map_policy_packet.py
```

Result: 25 checks passed. See `evidence/bounded-map-policy-cpu-proof.json`.

## Root CPU preflight (fresh output; Root only; no GPU)

```bash
cd /tmp/oxide-qwen35vl-sdk-map-policy-v1
env CUDA_VISIBLE_DEVICES= taskset -c 88-95 \
  python3 qwen35_vl_map_policy_controller.py --phase cpu --run \
  --model-name Qwen3.5-0.8B --case single-cats \
  --output /tmp/oxide-qwen35vl-sdk-map-policy-output/cpu-0.8B-single-v1
```

This validates all five named model records on the host, stages the pinned
runtime to explicit remote paths, verifies actual remote filenames and hashes,
then runs the inside SDK preflight with container paths and expected hashes
before the plan-only Native pass. It does not create a model engine.

## Root paired quality (fresh output; Root GPU0)

```bash
cd /tmp/oxide-qwen35vl-sdk-map-policy-v1
env CUDA_VISIBLE_DEVICES= taskset -c 88-95 \
  python3 qwen35_vl_map_policy_controller.py --phase paired --run \
  --model-name Qwen3.5-0.8B --case single-cats \
  --consumer /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO/release/qwen35-vl-consumer \
  --output /tmp/oxide-qwen35vl-sdk-map-policy-output/paired-0.8B-single-v1
```

Native uses explicit `tei_create(..., b"cuda:0", BF16)` and
`--production-cuda-graphs` with `_cuda_graphs=true`,
`_cuda_graph_max_tokens=4096`, entries 4, reuse 3, captures 8, max bytes
536870912, `_media_cuda_graphs=false`. Oxide receives no graph option.

## Root owning-life (fresh output; Root GPU0)

```bash
cd /tmp/oxide-qwen35vl-sdk-map-policy-v1
env CUDA_VISIBLE_DEVICES= taskset -c 88-95 \
  python3 qwen35_vl_map_policy_controller.py --phase life --run \
  --model-name Qwen3.5-0.8B --case single-cats \
  --life-consumer /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO-harness/release/qwen35-vl-consumer \
  --output /tmp/oxide-qwen35vl-sdk-map-policy-output/life-0.8B-single-v1
```

## Non-claims

The author did not run Docker, GPU, Torch CUDA, NVCC, network or a device
compile, and did not run a whole-model CPU forward. Root owns the actual CPU
retry, paired quality and owning-life execution. Nothing is claimed closed
before the actual Root CPU retry passes.
