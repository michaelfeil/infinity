#!/usr/bin/env bash
# Root-only device build for the fresh e582 Qwen3-VL composition.
# The author does not run this.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${QWEN_VL_SOURCE:-${ROOT}/libs/infinity-v3}"
: "${CARGO_OXIDE:?set CARGO_OXIDE to the pinned cargo-oxide compiler}"
: "${OLD90_BUNDLE:?set OLD90_BUNDLE to the published 90-export kernels directory}"
OUT="${1:?usage: build_all3.sh OUTPUT_DIRECTORY}"
mkdir -p "${ROOT}/logs"
bash "${SRC}/backends/oxide/device/build.sh" "${OUT}" > "${ROOT}/logs/aot-build.log" 2>&1
python3 "${ROOT}/aot/preserve_old90.py" \
  --old "${OLD90_BUNDLE}" --new "${OUT}" \
  --expected-old "${ROOT}/aot/expected-old90.txt" \
  --expected-new "${ROOT}/aot/expected-new96.txt" \
  --cuobjdump "${CUOBJDUMP:-/usr/local/cuda-13.1/bin/cuobjdump}" \
  --out "${OUT}/old90-preservation.json" \
  | tee "${ROOT}/logs/aot-preservation.log"
