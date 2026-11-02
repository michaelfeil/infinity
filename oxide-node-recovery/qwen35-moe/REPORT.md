# Qwen3.5-MoE factory service-blocking v3: warm checkout is plan-free

Preserves v2 `qwen35-moe-factory-service-blocking-v2` and immutable product 2d.
Only changed source file:
- `libs/infinity-v3/backends/oxide/src/qwen35_moe_factory.rs`

## Warm-path correction

`checkout_work` now accepts a lazy `FnOnce` projected-byte plan:

- matching Ready Work is checked out first and returns without invoking the plan
  closure, build closure, or eviction closure;
- cache miss invokes the projected-byte plan exactly once before allocation or
  budget admission;
- the engine `work_bytes` closure now lives inside the actual checkout helper,
  after the existing common blocking `engine.lease()` acquisition;
- all prior behavior is preserved: independent geometry slots, common
  `cache::BYTE_BUDGET`, checked accounting, one oversized active, proof-idle
  eviction only, retained/quarantined bytes charged, pre-enqueue RAII, fatal
  proof/fence handling, and duplicate canonicalization.

No engine/provider/context plan is executed on a warm Ready cache hit.

## CPU evidence

Focused `qwen35_moe`: 74 passed, 0 failed.
Clippy `-D warnings` and fmt pass.

Added tests:
- cached Ready hit with panic closures for plan/build/evict, proving none run;
- cache miss plans exactly once and over-budget rejects with build count zero.

No optimized build, GPU, Docker, Torch, network, device compile, harness edit,
product edit, or v2 edit was performed. Fresh independent source review of this
complete v3 freeze is required before any optimized build.
