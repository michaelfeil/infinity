# Qwen3.5-VL isolated private leaf (base e582ce9)

This packet is the only Qwen3.5 vision-language work owned here. It does not
edit or overwrite the Qwen3-VL private fork (`qwen3-vl-v3`/`qwen-vl-current-v1`)
and does not modify the official Qwen3.5 text integration worktree.

## Source

- Worktree: `./` (full source snapshot of public `e582ce9`).
- Qualified text foundation: `/tmp/infinity-oxide-parity-20261101/qwen35-public-efbd-foundation-v1/source`
  (base `efbd735`), applied with the one `device/build.sh` conflict resolved by
  merging its Qwen3.5 leaves into the e582 target-subset build script.
- Qwen3.5-VL additions: `qwen35_vl_{config,media,vision,text,inventory,device,tests}.rs`.
- Device leaf: `device/src/qwen35vl.rs` + `device/qwen35vl_exports.txt`.
- Consumer: `consumer/` (`inventory|plan|tower-cpu|tower-device|lifetime|perf`).
- Native gold producer: `native/dump_vision_tensors.py`,
  `native/tower_probe.cpp`.
- Root controller: `root-v1/qwen35_vl_root_controller.py`.
- Device build plan/pins/old-export proof: `device-packet/`.

## CPU evidence

- `cargo test -p infinity-backend-oxide --all-targets --offline`:
  586 passed, 0 failed, 71 ignored.
- Qwen3.5-VL focused tests: 12 passed, 1 ignored (actual-checkpoint metadata).
- Actual checkpoint metadata: 0.8B (153), 4B (297), 35B (333) vision tensors pass.
- Real 0.8B CPU tower forward over a 2x2 patch grid produces 1 merged row of
  1024 values.
- 35B-A3B cached weight sidecar validates 333 vision tensors (depth 27,
  H1152/16 heads/d72, out_hidden 2048) from config+index metadata only.
- `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` pass.
- Old export proof: 86 old + 9 foundation + 4 encoder + 7 Qwen3.5-VL = 106.

## Root

Run `python3 root-v1/qwen35_vl_root_controller.py host-preflight` first. Every
stage is printed by default; add `--execute` only on Root GPU 0 inside the pinned
SDK container. Full text pairing remains blocked on the official text hook.
