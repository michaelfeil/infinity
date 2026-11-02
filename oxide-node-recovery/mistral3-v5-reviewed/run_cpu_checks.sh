#!/usr/bin/env bash
# CPU-only Mistral3 v5 composer evidence on the current Root working tree
# (public 55380b7 + 625DenseV5/Qwen35/MeanV6 six-file overlay).
# Debug/default tests only; no release/fat-LTO all-target build.
# CPU 136-143, four jobs, empty CUDA_VISIBLE_DEVICES, offline.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_HEAD="55380b7"
BASE_TREE="$(git -C /tmp/infinity-plain-cuda rev-parse HEAD^{tree})"
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_NET_OFFLINE=true
export CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1
export INFINITY_MISTRAL3_MODEL=/workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16
export INFINITY_MISTRAL3_FIXTURE="${ROOT}/fixtures/processor-cases.json"
export CARGO_TARGET_DIR="${ROOT}/.work/target-cpu-debug"
LOG="${ROOT}/logs/debug"
mkdir -p "${LOG}" "${ROOT}/evidence" "${ROOT}/.work/tmp" "${CARGO_TARGET_DIR}"

if [[ ! -s "${ROOT}/fixtures/processor-cases.json" ]]; then
  (cd "${ROOT}/fixtures" && python3 make_fixtures.py > make-fixtures.log)
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cpu_affinity_requested=136-143"
  echo "cpu_affinity_effective=$(taskset -c 136-143 sh -c 'taskset -pc $$' | tr -d '\n')"
  echo "job_limit=${CARGO_BUILD_JOBS}"
  echo "cuda_visible_devices='${CUDA_VISIBLE_DEVICES}'"
  echo "cargo_net_offline=${CARGO_NET_OFFLINE}"
  echo "base_head=${BASE_HEAD}"
  echo "base_tree=${BASE_TREE}"
  echo "base_overlay=V5/Qwen35/MeanV6 six files from /tmp/infinity-plain-cuda working tree"
  cargo +1.93.1 --version
  rustc +1.93.1 --version
  python3 --version
} > "${ROOT}/evidence/environment.txt"

cd "${ROOT}/native"
CUDA_VISIBLE_DEVICES= python3 native_b66_control.py > "${LOG}/native-b66-cpu-control.log" 2>&1
CUDA_VISIBLE_DEVICES= python3 root_paired_controller.py --preflight-only \
  --checkpoint "${INFINITY_MISTRAL3_MODEL}" > "${LOG}/paired-preflight.json" 2>&1
CUDA_VISIBLE_DEVICES= python3 root_native_cpu_stage.py --preflight-only \
  --checkpoint "${INFINITY_MISTRAL3_MODEL}" > "${LOG}/native-cpu-stage-preflight.json" 2>&1
cd "${ROOT}"

cd "${ROOT}/libs/infinity-v3"
cargo +1.93.1 fmt --all -- --check > "${LOG}/rustfmt-check.log" 2>&1
taskset -c 136-143 cargo +1.93.1 clippy -p infinity-backend-oxide --all-targets -- -D warnings \
  > "${LOG}/clippy-all-targets.log" 2>&1
taskset -c 136-143 cargo +1.93.1 test -p infinity-backend-oxide --lib -- --test-threads=4 \
  > "${LOG}/oxide-lib.log" 2>&1
taskset -c 136-143 cargo +1.93.1 test -p infinity-backend-oxide --lib mistral3::tests -- --test-threads=4 \
  > "${LOG}/mistral3-tests.log" 2>&1
taskset -c 136-143 cargo +1.93.1 test -p infinity-backend-oxide --lib cold_mistral3 -- --test-threads=4 \
  > "${LOG}/cold-hook.log" 2>&1
taskset -c 136-143 cargo +1.93.1 test -p infinity-backend-oxide --lib native_rope_tables -- --test-threads=4 \
  > "${LOG}/rope-tests.log" 2>&1
taskset -c 136-143 cargo +1.93.1 run -p infinity-backend-oxide --example mistral3_admission -- \
  "${INFINITY_MISTRAL3_MODEL}" > "${LOG}/mistral3-cli-admission.log" 2>&1
set +e
taskset -c 136-143 cargo +1.93.1 run -p infinity-backend-oxide --example mistral3_admission -- \
  /tmp/mistral3-does-not-exist > "${LOG}/mistral3-cli-error.log" 2>&1
CLI_ERROR_STATUS=$?
set -e
test "${CLI_ERROR_STATUS}" -eq 2
taskset -c 136-143 cargo +1.93.1 test -p infinity_v3-backend-core --lib -- --test-threads=4 \
  > "${LOG}/core-lib.log" 2>&1
taskset -c 136-143 env CARGO_TARGET_DIR="${ROOT}/.work/host-consumer-debug" \
  cargo +1.93.1 build --offline --manifest-path "${ROOT}/native/host-consumer/Cargo.toml" \
  > "${LOG}/host-consumer-debug.log" 2>&1
taskset -c 136-143 cargo +1.93.1 check --offline -p infinity_v3-router \
  --no-default-features --features oxide,http \
  > "${LOG}/router-oxide-only-check.log" 2>&1
cd "${ROOT}/native"
CUDA_VISIBLE_DEVICES= python3 root_staged_controller.py --preflight-only \
  --host-model "${INFINITY_MISTRAL3_MODEL}" \
  > "${LOG}/staged-controller-preflight.json" 2>&1
python3 -m unittest -q "${ROOT}/native/test_staged_controller.py" \
  > "${LOG}/staged-controller-source-tests.log" 2>&1
cd "${ROOT}"

# Snapshot the current Root working tree read-only: public HEAD plus the six
# V5/Qwen35/MeanV6 overlay files, then dry-run the composer.
rm -rf "${ROOT}/.work/current-base-check"
mkdir -p "${ROOT}/.work/current-base-check"
git -C /tmp/infinity-remove-vertex archive "${BASE_HEAD}" | tar -x -C "${ROOT}/.work/current-base-check"
for f in libs/infinity-v3/backends/oxide/src/cublas.rs \
         libs/infinity-v3/backends/oxide/src/decoder.rs \
         libs/infinity-v3/backends/oxide/src/decoder_model.rs \
         libs/infinity-v3/backends/oxide/src/lib.rs \
         libs/infinity-v3/backends/oxide/src/native_rope_tables.rs \
         libs/infinity-v3/backends/oxide/src/decoder_classifier.rs; do
  mkdir -p "${ROOT}/.work/current-base-check/$(dirname "$f")"
  cp -a "/tmp/infinity-plain-cuda/$f" "${ROOT}/.work/current-base-check/$f"
done
(
  cd "${ROOT}/.work/current-base-check"
  patch -p1 --dry-run -i "${ROOT}/selective.patch"
) > "${LOG}/patch-dry-run.log" 2>&1

echo "CPU-only Mistral3 v5 evidence run complete" >&2
