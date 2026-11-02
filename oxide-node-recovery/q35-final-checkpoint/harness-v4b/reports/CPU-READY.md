# Q35MoE unified runtime v4b source update

Product source (unsealed): `/tmp/oxide-all-model-feature-matrix-v1/qwen35-moe-factory-service-blocking-v2`
Product freeze: `UNBOUND` (explicit placeholder; Codex binds the final product sha256 before any build)
Product read or compiled by this update: no.
Harness source updated only inside this root: `/tmp/oxide-q35moe-unified-runtime-v4b`.
Native controller and comparator were not edited.

Build/run status for this source update:
- No CPU build or test run was performed.
- No optimized/release build was performed.
- No GPU, CUDA context, device query, Docker, Torch, network, device compile, or model weights were used.
- `reports/validate-only.json`, `reports/metadata-cpu.json`, and the `reports/dev-*.log` files are historical evidence from the prior v2/v3 snapshots; they are not v4b execution evidence.
- `HARNESS-FREEZE-v2.json` is retained as a historical v2 file map; it was not regenerated and does not freeze the v4b source.

Source changes:
- Cargo retargets the three product path dependencies to `qwen35-moe-factory-service-blocking-v2`; release profile remains opt-level 3, no LTO, debug 1, codegen-units 16.
- Product freeze reported by the harness is explicitly `UNBOUND`.
- `control/generate_fixtures.py` makes `duplicate_t7` and `duplicate_raw_t7` valid normal metadata, keeps bounded duplicates and raw override, and adds synthetic `t9` with exact input ids `20+13*i` for i=0..8, positions 0..8, zero token types, cumulative `[0,9]`, max length 9, pooled `[0]`, raw `[]`.
- The old harness selected-row capacity guard was removed; production canonicalization handles duplicate pooled/raw and raw override without an arbitrary `selected <= tokens + sequences` limit.
- Core lifecycle uses `Arc<Qwen35MoeBackend>`. The third held submit runs in an owned standard worker thread with a start channel, blocks on the common two-slot pool while A/B are held, then becomes Deferred after B completes. The caller polls for 50ms without yielding, receives within 5s, joins, and completes C then A.
- Held T7 plus T9 different geometry uses the two independent Work keys and must be Deferred under the small slot budget; T9 completes first, then A, and T9 retry is exact. No healthy Busy/Overloaded is expected.
- Service lifecycle submits a three-burst A/B/C=A through the normal `Backend::submit_embed`; all three must be Deferred because the CompletionThread is independent, then C/B/A complete against exact baselines.
- Metadata before/after, actual typed errors, finite/index/shape checks, no-selection Ready, and phase checkpoint persistence remain.
- Invalid media, compact, token-id, position, cumulative, and selection cases remain tested, including no-selection variants.
- The CPU ResultPool budget test remains pure CPU with no fake fault injection.

T9 manifest sha256 is generator-owned: `control/generate_fixtures.py` replaces the checked-in `GENERATE-WITH-control/generate_fixtures.py` sentinel before freeze and before any build. `control/freeze.py` refuses to write a freeze while that sentinel remains.

This file states source status only. It is not a CPU build or GPU qualification.
