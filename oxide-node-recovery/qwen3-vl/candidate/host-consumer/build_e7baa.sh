#!/usr/bin/env bash
# ONE incremental optimized host single-consumer build for frozen e7baa453.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR=/tmp/opencode/qwen-vl-host-consumer-e7baa-target
cd "${ROOT}/libs/infinity-v3"
start=$(date +%s)
taskset -c 72-79 cargo build --offline --release \
  --config 'profile.release.opt-level=3' \
  --config 'profile.release.lto=false' \
  --config 'profile.release.debug=1' \
  --config 'profile.release.codegen-units=16' \
  --config 'profile.release.debug-assertions=false' \
  -p infinity-backend-oxide --example qwen3_vl_host_consumer \
  2>&1 | tee "${ROOT}/logs/host-consumer-e7baa-build.log"
end=$(date +%s)
echo "wall_seconds=$((end - start))" | tee "${ROOT}/host-consumer/e7baa-walltime.txt"
sha256sum "${CARGO_TARGET_DIR}/release/examples/qwen3_vl_host_consumer" \
  | tee "${ROOT}/host-consumer/e7baa-binary.sha256"
