# Qwen3-VL fresh e582 composition + Root gate

**Status:** source-ready, CPU gates pass, Root GPU/lifetime/perf/AOT pending.
This worktree owns **Qwen3-VL only**. Qwen3.5-VL is a separate author using the
shared plan in `../qwen-vl-shared-plan-v1`.

## Base and identity

- Public base: `/tmp/infinity-plain-cuda` @
  `e582ce90eb1544fcff081a495cb8fd4f3c17b112`
- Patch: `selective.patch` (applies clean to `e582ce9`)
- Native b66: `/tmp/native214-fa4-mode20/install/lib/libtei_torch.so`
  - SHA-256 `b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e`
  - cuBLASLt `130200`, Torch `2.14.1+cu130`
- Checkpoint: cached `Qwen/Qwen3-VL-Embedding-8B`
  - 749 tensors, 16,289,587,680 bytes
  - config SHA `1eaa88d8d8824a47304e864c14101fe9efe3286607a7573f27eb723637f0d73d`

## Implemented

### Concrete Qwen3-VL execution

- Cold BF16 `model.visual.*` upload, reusable vision/decoder/position workspaces.
- Patch projection, four-corner learned-position interpolation, split-half 2-D
  vision rotary, per-frame non-causal FA4, merger and deepstack mergers.
- Additive kernels:
  `infinity_decoder_rows_bf16`, `infinity_qwen3vl_add_bf16`,
  `infinity_qwen3vl_pos_interp`, `infinity_qwen3vl_rope2d_bf16`,
  `infinity_qwen3vl_gelu_tanh_bf16`, `infinity_qwen3vl_silu_bf16`.
- Visual row **replacement** (`add=0`) at marker spans; deepstack residual
  consolidation then additive rows (`add=1`); selected-readout shortcut disabled
  whenever external media is present.
- External interleaved mRoPE cos/sin with identity row remap; text-only explicit
  media positions now flow through the same path.
- `Pool::Cls/Mean/LastToken`, raw token rows, finite-output checks.
- Image and multi-frame video grids share the same real frame isolation.

### Native aliases / trained wrappers

- `qwen3_vl` plain embedding.
- `colqwen3`: multi-vector, trained Dense token projection loaded from the
  checkpoint's actual `modules.json`, exact-erf merger GELU, L2 token rows.
- `qwen3_vl_nemotron_embed`: bidirectional text, pre-final-norm token rows
  (`_skip_final_norm`), L2 token rows.
- `HF_ColBERT`: main-checkpoint `linear.weight` token projection + L2 rows.
- Explicit opt-in rerank score head (`score.weight`); **not** auto-selected by
  any Native Qwen3-VL `model_type`.
- Both `rope_scaling` and `rope_parameters` spellings are accepted; config.json
  is never rewritten.

### Owning deferred engine

`Qwen3VlEngine` is the production route behind the normal factory. It loads the
model once, owns one CUDA stream and exactly two `HostIo` leases, issues a
same-stream completion ticket before the first GPU upload, stages request-specific
pinned metadata, writes the output into the request's pinned `HostIo`, and returns
`Inference::Deferred`. Per-request workspaces are held A/B and reused only after
their ticket completes; `DeferredVl::Drop` waits for the fence even when the
caller drops the result. Invalid media is rejected before a lease or ticket and
is recoverable. There is no model reupload and no per-op context bind.

### Host consumer freeze

`examples/qwen3_vl_host_consumer.rs` is the single unified host consumer:
manifest, normal-factory alias, all trained projection heads, all 8 fixtures,
typed media errors, and lifecycle A/B/growth/drop. Successor built once with
`host-consumer/build_e7baa.sh` at opt-level 3 / LTO false / unique target.
Historical e7baa consumer SHA `01516290d5dc97960121a6da086d81e1ad6be0a800eed2c941fe1a3b4e51107f` (opt3, no LTO, debug=1, CGU16; wall 41s; ELF is stripped). The new quality-output consumer build is held pending independent PASS.
The pre-P1 source and binary `6e4f...` are preserved in `history/`. Changed-shape cache eviction still takes a conservative full-stream Drop fence; this is a documented limitation, not a per-image sync.
`--mode gpu-quality` writes f32 pool0/1/2/raw vectors on the actual Backend path; `--mode gpu-forward`/`gpu-lifecycle` are additional Root paths. None was run by the author.

### Native b66 paired CPU stage

`native-b66-cpu-stage/STAGE.json` pins the four exact cached checkpoints, the 8
real media fixtures, the Native b66 identity/packet/consumer, the Oxide host
consumer, typed media errors, the inner `remote_guard`, the b66 13.2.0.0 GOT
proof and the `/bench` read-only, no-weight-copy controller.

### Product route (normal factory)

`backends/src/lib.rs::init_backend` detects `qwen3_vl`/`colqwen3`/
`qwen3_vl_nemotron_embed` before dense admission, builds
`Qwen3VlBackend::new_with_pool` inside the normal startup worker, and returns it
to the normal `BackendThread`. `submit_embed(Batch)` then flows through
`pack_batch` (`Batch.multimodal` + position ids + prepared pixels) into the
concrete vision/decoder/readout path. There is no isolated facade the normal
factory cannot reach. A CPU factory test proves the alias dispatch. The literal
dense `OxideBackend::new` constructor remains dense-only; changing its return
type is a separate refactor and is not claimed.

### Actual checkpoint coverage

- Cached **Qwen3-VL-8B**.
- Cached **TomoroAI ColQwen3-4B**: `vlm.model.*` namespace, `embedding_proj_layer`
  320x2560 in the main checkpoint, 715-tensor inventory, BF16 despite
  `text_config.dtype=float32`.
- Cached **Nemotron ColEmbed 4B and 8B**: `_skip_final_norm`, no projection,
  BF16, full inventories.
- Derived public 2B and 32B tuples are structurally ready. FA4 selection uses
  the grouping ratio, not absolute heads: 32B Q64/KV8 d128 reuses modes 22/5,
  and vision 12/12 d64 global reuses mode0 (`bert`). The only genuine new ratio
  is Q40/KV8 d128 ratio 5 (14B/other), reported to the central FA4 owner.
  30B-A3B MoE text is outside the dense foundation.

Geometry report: `qwen-vl-geometry-v1/CONFIG-TUPLES.json`. Central FA4 request:
`qwen-vl-geometry-v1/FA4-REQUEST.json`.

### Composition / factory / plumbing

- `qwen3_vl_batch.rs`: engine `Batch` + `MultimodalEncoding` -> concrete
  `Qwen3VlForward`, including per-sequence media positions and text rows in a
  partially multimodal batch.
- `qwen3_vl_readout.rs`: host trained readout chain and L2 normalization.
- `qwen3_vl_service.rs`: `Qwen3VlModel`, `Qwen3VlBackend` (`Backend` impl),
  `new_with_pool`, explicit readout override.
- `qwen3_vl_fixtures.rs`: real 8-fixture manifest -> engine `Batch`.
- `examples/qwen3_vl_oxide_consumer.rs`: release Oxide consumer for Root.
- `rust-consumer/`: release Native b66 consumer, all pools for every sequence.

### Qwen3.5-VL split

Early shared plan:
`/tmp/oxide-all-model-feature-matrix-v1/qwen-vl-shared-plan-v1`.
It names the actual vision modules, row-replacement ABI, mRoPE functions and
engine hooks, plus the cached Qwen3.5-VL differences (no deepstack,
`partial_rotary_factor`, `rope_parameters`, hybrid text foundation). No Qwen3.5
code is present here and no shared private path was written.

## CPU evidence (author)

- Patch applies clean to `e582ce9`.
- Clean-tree `cargo test -p infinity-backend-oxide --lib`:
  **486 passed, 0 failed, 56 ignored**.
- Normal factory tests (`infinity_v3-backend --no-default-features --features
  oxide --lib`): **22 passed**.
- Actual ColQwen3 and Nemotron 4B/8B config/inventory/projection tests: pass.
- Geometry coverage test: cached tuples ready; 32B missing-FA4 and 30B-A3B
  missing-decoder reported.
- New composition tests: 15 passed, including all 8 real fixtures packed into
  engine batches.
- Real fixture test: **1 passed**.
- `cargo fmt --all -- --check`: pass.
- `cargo clippy -p infinity-backend-oxide --all-targets -- -D warnings`: pass.
- Oxide consumer debug + release build: pass.
- Native b66 consumer debug + release build: pass.
- Host preflight subset (hashes/tensors/config/fixtures/consumer SHAs): pass,
  see `root-gate/host-preflight.json`.
- Isolation: CPU 72-79, jobs 4, `CUDA_VISIBLE_DEVICES=''`, offline, no Docker /
  GPU / CUDA context / NVCC / Torch / network / device compile.

## Root gate

See `review/INDEPENDENT-REVIEW.json`, `review/PACKAGING-V4-FIXES.json`, `root-gate/GATE.json` and
`root-gate/README.md`.

- Preflight: actual checkpoint hashes, 749 tensors, config equality, b66 SHA,
  20 exports, CUDA linkage, C ABI 72/120/48, Torch identity, cuBLASLt 130200,
  8 real fixtures, consumer SHAs.
- Quality: Native production (default Native graphs) vs Oxide no-graphs over all
  8 fixtures and all pools; raw rows separately; provisional thresholds are
  written into the result.
- Lifetime: 50 iterations per fixture plus a second reversed process; no
  non-finite, crash, stuck stream or unbounded growth.
- Performance: 20 warmups + 100 measured iterations; admission `p50 <= 1.05x`
  Native, optimization goal 0.90-0.95x. No number is claimed.
- AOT: fast `sm_90 --only-symbol-probes`, then all3 old90 preservation;
  90 -> 96 exports, canonical26 untouched.

## Non-claims / blockers

- Root GPU quality, lifetime and performance were not run.
- AOT old90 preservation was not executed; `OLD90_BUNDLE` is required.
- The ColQwen3/Nemotron readout projection is host-side after token readback; if
  the performance arm misses target, implement a device DenseHead readout.
- `_transformers_math` rounding is represented as a wrapper flag but the Oxide
  decoder keeps its existing reference kernel rounding; Root quality decides.
- No Qwen3.5-VL implementation, no public/common tree mutation, no push.
