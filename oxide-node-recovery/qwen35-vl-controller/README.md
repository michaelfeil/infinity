# Qwen3.5-VL full private branch (base e582ce9 + serving text snapshot)

This branch is isolated from the Qwen3-VL fork, the official f417 text
integration worktree and the Qwen3.5-MoE engine. It starts from the current
`/tmp/infinity-oxide-qwen35-serving-e582-v1-worktree` snapshot (BF16 text core,
99 additive device exports, 583 text tests) and adds an owning Qwen3.5-VL leaf
without overwriting text-worker files.

## What is new

- `qwen35_vl_config/media/vision/text/inventory/device/tests.rs`: the isolated
  vision tower and media contract.
- `qwen35_vl_engine.rs`: owning engine/Work/HostIo integration. It attaches to
  the public `OxideBackend` text facade, builds merged visual rows, uploads
  per-request external partial-mRoPE tables, copies visual rows into the gathered
  embedding slab before the layer loop, and submits through the existing engine
  completion/retirement path. Pooled/raw and causal Decision routes share the
  same hook.
- Selective text hooks only:
  - `Qwen35Model::enqueue(..., vl)` visual-row copy and rope selection;
  - `PreparedWork::enqueue/enqueue_block` optional `Qwen35RopeOverride`.
  GDN layers never receive external positions; only full-attention layers do.
- Seven additive device exports (`qwen35vl_exports.txt`), taking 99 -> 106.
- Native whole-model gold producer (`native/native_whole_model.py`) using the
  pinned b66 C API with all original checkpoint tensors (no Torch/surrogate).
- Native tower probe and the Root paired controller now cover tower and complete
  whole-model pooled/raw comparison.

## Full checkpoints verified (metadata only, original payload untouched)

| checkpoint | tensors | visual | depth | text hidden | media config |
|---|---:|---:|---:|---:|---|
| Elephant-Qwen3.5-VL | 760 | 333 | 27 | 4096 | combined `processor_config.json` |
| Mammoth-Qwen3.5-VL | 1184 | 333 | 27 | 5120 | combined `processor_config.json` |
| Qwen3.5-0.8B/2B/4B | 488/632/738 | 153/297/297 | 12/24/24 | 1024/2048/2560 | separate preprocessors |
| Qwen3.5-35B-A3B | 1811 | 333 | 27 | 2048 | config-only sidecar |

Elephant/Mammoth pass the text owner's full candidate gate and are the primary
dense full-model pair. The 35B MoE text engine remains with its owning MoE
worker; this leaf's vision/fusion contract applies when that owner exposes the
same embedding hook.

## Dtype coverage (do not conflate)

- Device and owning engine: **BF16 only** (text, vision FA4/Lt, pointwise).
- CPU reference: BF16-rounded recipe.
- F16 patch-split: source structure only, not an F16 device/reference recipe.
- F32 device: not attached.
- Native exceptions: `A_log` and `dt_bias` keep stored F32 via
  `WeightSource::read_f32`; ordinary `norm.weight` still converts to BF16.
- 4B metadata: 690 BF16 + 48 F32 (24 A_log + 24 linear_attn.norm.weight).

## Normal factory and central modes

The ordinary server route is `OxideBackend::new` -> `Backend::embed/decide`:
`new_qwen35_serving` stores the VL runtime, and submit paths convert
`Batch.multimodal` image patches into fused visual rows and external mRoPE
automatically. No manual `Qwen35VlEngine` attachment is required for serving; a
media Batch without a VL runtime is rejected.

Published FA4 bakes ratio + head_dim + dtype + causal/local; absolute q/kv are
runtime tensor shapes. Cold plans therefore match ratio, not absolute heads:
- 0.8B/2B/4B: vision ratio1 d64 global mode 0; text ratio4 d256 causal mode 18.
- Elephant: vision 8 + noncausal 6; Mammoth: vision 8 + noncausal 16.
- 35B-A3B: vision 8 + ratio8 d256 causal new tail 26, plus MoE text owner.
All dense full checkpoints (0.8B, 2B, 4B, Elephant, Mammoth) are normal-factory
ready. Only 35B remains gated. Requests: `device-packet/FA4-REQUEST.json`.
FP16 remains unsupported until the Native exact recipe exists.

## Optimized host consumer (one build)

- Binary: `target-opt3-noLTO/release/qwen35-vl-consumer`
- SHA-256: `3780facf61529822f33a4cf1416bc6c8723acc1fa2cd2c778f7bc54f2140d83b`
- Build: CPU 88-95, jobs 4, offline, CVD empty, `opt-level=3`, `lto=false`,
  unique target `target-opt3-noLTO`; exit 0, 39 s.
- Pins: `device-packet/CONSUMER-PINS.json`, `device-packet/CONSUMER-BUILD.json`.

## CPU evidence

- `cargo test -p infinity-backend-oxide --all-targets --offline`:
  **646 passed, 0 failed, 135 ignored** (lib **614 passed, 75 ignored**).
- Qwen3.5-VL focused: 13 passed, 2 ignored.
- Actual full Elephant/Mammoth metadata + text-candidate admission: 1 passed.
- Real 0.8B CPU tower forward: 1 merged row x 1024.
- Clippy `-D warnings`, fmt, consumer release build, Native plan-only mRoPE
  contract: pass.
- Old export proof: 86 old + 9 foundation + 4 encoder + 7 Qwen3.5-VL = 106.

## Root

`root-v1/qwen35_vl_root_controller.py` now has `host-preflight`,
`native-whole`, `oxide-whole`, `pair-whole`, `native-gold`, `oxide-tower`,
`pair`, `lifetime`, `perf`. Commands print by default; `--execute` is Root-only.
No GPU, Docker, Torch CUDA, NVCC, network or device compile was run here.
