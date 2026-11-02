#!/usr/bin/env bash
# ONE incremental optimized host-consumer build AFTER independent PASS.
# No optimized compile may run before OXIDE_INDEPENDENT_PASS=1 is granted.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${OXIDE_BUILD_ROOT:-${BASE}/candidate}"
TARGET="${OXIDE_TARGET_DIR:-/tmp/opencode/qwen3vl-rope-budget-fix-opt3-target}"

if [ "${OXIDE_INDEPENDENT_PASS:-0}" != "1" ]; then
  echo "Refusing optimized build: independent PASS not granted." >&2
  echo "Set OXIDE_INDEPENDENT_PASS=1 only after the reviewer's PASS." >&2
  exit 2
fi

export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="${TARGET}"
mkdir -p "${BASE}/proof"
cd "${ROOT}/libs/infinity-v3"
start=$(date +%s)
taskset -c 72-79 cargo build --offline --release \
  --config 'profile.release.opt-level=3' \
  --config 'profile.release.lto=false' \
  --config 'profile.release.debug=1' \
  --config 'profile.release.codegen-units=16' \
  --config 'profile.release.debug-assertions=false' \
  --config 'profile.release.strip="none"' \
  -p infinity-backend-oxide --example qwen3_vl_host_consumer \
  2>&1 | tee "${BASE}/proof/optimized-host-build.log"
end=$(date +%s)
echo "wall_seconds=$((end - start))" | tee "${BASE}/proof/optimized-host-walltime.txt"
sha256sum "${CARGO_TARGET_DIR}/release/examples/qwen3_vl_host_consumer" \
  | tee "${BASE}/proof/optimized-host-binary.sha256"
