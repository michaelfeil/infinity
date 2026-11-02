#!/usr/bin/env bash
# ONE incremental host build after the try_acquire/async-audit freeze.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR=/tmp/opencode/qwen-vl-host-consumer-successor2-target
cd "${ROOT}/libs/infinity-v3"
taskset -c 72-79 cargo build --offline --release \
  --config 'profile.release.lto=false' \
  --config 'profile.release.opt-level=3' \
  -p infinity-backend-oxide --example qwen3_vl_host_consumer \
  2>&1 | tee "${ROOT}/logs/host-consumer-successor2-build.log"
sha256sum "${CARGO_TARGET_DIR}/release/examples/qwen3_vl_host_consumer" \
  | tee "${ROOT}/host-consumer/successor2.sha256"
