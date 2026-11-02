#!/bin/bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 FRESH_OUTPUT_DIR" >&2
  exit 2
fi
OUT=$1
if [ -e "$OUT" ]; then
  echo "fresh output required: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$(dirname "$0")"
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  taskset -c 88-95 python3 tests/test_map_policy_packet.py | tee "$OUT/tests.stdout.json"
cp evidence/bounded-map-policy-cpu-proof.json "$OUT/"
env CUDA_VISIBLE_DEVICES= taskset -c 88-95 \
  python3 qwen35_vl_map_policy_controller.py --phase pins \
  --model-name Qwen3.5-0.8B > "$OUT/pins-0.8B.json"
env CUDA_VISIBLE_DEVICES= taskset -c 88-95 \
  python3 qwen35_vl_map_policy_controller.py --phase plan > "$OUT/plan.json"
echo "cold CPU proof written to $OUT"
