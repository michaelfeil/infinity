# Qwen3-30B-A3B paired LibTorch vs public Oxide performance harness

Source-only harness against public commit `252789b05ea2e09c14971c04dfcce458f863fead`.
No GPU, Torch, Docker, device compile, AOT or optimized build was run by the author.

## Files

- `oxide-perf/` — standalone Rust normal-backend timing consumer. It loads one
  `OxideBackend::new`, records first call, runs ABA variable-shape warm, then 20
  warmups + 200 timed calls per case. Timed section includes submit + complete +
  canonical result pack/hash. Oxide graphs are off.
- `native-perf/native_perf_process.py` — one-engine LibTorch process using the
  exact V8-qualified SDK/helper and preserved graphs/defaults. Same cases,
  first-call/ABA/warmup/timed contract.
- `controller/run_paired.py` — balanced `N,O,O,N`, one engine at a time, GPU0,
  quality exact checks, artifact pins, timing aggregation.
- `PERF-MANIFEST.json`, `SOURCE-FREEZE.json`, `CASES-SOURCE.json`.
- `evidence/` — CPU preflight, Rust tests, clippy/fmt logs.

## Workloads

1. `t7` — exact Native `ragged7-pooled` batch.
2. `ragged129` — exact Native `ragged129-mixed` batch.
3. `stable2k` — 2048-token single sequence, mean pooled; fit-dependent.

T7/ragged129 are first. Stable2k runs only if the engine fits.

## Quality

- T7/ragged129: every Native and Oxide window output compared to the existing
  mean Native gold, exact vectors only.
- Stable2k: first Native window becomes the reference; all later Native/Oxide
  windows must match exact bits.
- Each process checks first/warm/timed output hash identity and ABA hash identity.

## Required Root inputs

- Exact Native SDK `/tmp/native214-fa4-mode20/install/lib/libtei_torch.so`
  SHA-256 `b66d70b6…` (absent in author environment).
- Exact helper/header/gold pins in `PERF-MANIFEST.json`.
- One new `oxide_perf` optimized production binary + build pin. No build was run
  by the author.

See `ROOT-COMMANDS.md`.
