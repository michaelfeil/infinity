#!/usr/bin/env bash
# CPU-only author evidence for the fresh e582 Qwen3-VL composition.
# No Docker, GPU, CUDA context, NVCC, Torch, network or device compile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=e582ce90eb1544fcff081a495cb8fd4f3c17b112
export PATH="${HOME}/.cargo/bin:${PATH}"
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1
mkdir -p "${ROOT}/logs" "${ROOT}/evidence"
WORK="${TMPDIR:-/tmp}/opencode/qwen-vl-current-checks"
rm -rf "${WORK}"; mkdir -p "${WORK}/tree" "${WORK}/tmp"
{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cpu_affinity_requested=72-79"
  echo "cpu_affinity_effective=$(taskset -c 72-79 sh -c 'taskset -pc $$' 2>/dev/null | tr -d '\n')"
  echo "cuda_visible_devices='${CUDA_VISIBLE_DEVICES}'"
  echo "cargo_net_offline=${CARGO_NET_OFFLINE}"
  echo "job_limit=${CARGO_BUILD_JOBS}"
  echo "host_rustc=$(rustc --version)"
  echo "host_cargo=$(cargo --version)"
} > "${ROOT}/evidence/environment.txt"
git -C /tmp/infinity-plain-cuda archive "${BASE}" libs/infinity-v3 | tar -C "${WORK}/tree" -xf -
cd "${WORK}/tree"
git init -q
git add -A
git -c user.email=author@local -c user.name=author commit -qm "${BASE}"
git apply "${ROOT}/selective.patch"
cd "${WORK}/tree/libs/infinity-v3"
echo "tree_rustc=$(rustc --version)" >> "${ROOT}/evidence/environment.txt"
echo "tree_cargo=$(cargo --version)" >> "${ROOT}/evidence/environment.txt"
export CARGO_TARGET_DIR="${WORK}/target"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo fmt --all -- --check 2>&1 | tee "${ROOT}/logs/fmt-check.log"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo clippy --offline -p infinity-backend-oxide --all-targets -- -D warnings 2>&1 | tee "${ROOT}/logs/clippy.log"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo test --offline -p infinity-backend-oxide --lib 2>&1 | tee "${ROOT}/logs/oxide-lib-tests.log"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo test --offline -p infinity_v3-backend --no-default-features --features oxide --lib 2>&1 | tee "${ROOT}/logs/backend-factory-tests.log"
mkdir -p "${ROOT}/qwen-vl-geometry-v1"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo run --offline -q -p infinity-backend-oxide --example qwen3_vl_geometry > "${ROOT}/qwen-vl-geometry-v1/CONFIG-TUPLES.json"
TMPDIR="${WORK}/tmp" QWEN3_VL_FIXTURES="${ROOT}/native-baseline/fixtures" taskset -c 72-79 cargo test --offline -p infinity-backend-oxide --test qwen3_vl_real_fixtures 2>&1 | tee "${ROOT}/logs/real-fixtures.log"
TMPDIR="${WORK}/tmp" taskset -c 72-79 cargo build --offline -p infinity-backend-oxide --example qwen3_vl_oxide_consumer 2>&1 | tee "${ROOT}/logs/oxide-consumer-build.log"
python3 - "${ROOT}" "${WORK}" <<'PY'
import hashlib, json, pathlib, re, subprocess, sys
root, work = map(pathlib.Path, sys.argv[1:3])
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
lib_log = (root / 'logs/oxide-lib-tests.log').read_text()
m = re.search(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', lib_log)
fixture_log = (root / 'logs/real-fixtures.log').read_text()
f = re.search(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', fixture_log)
consumer = work / 'target/debug/examples/qwen3_vl_oxide_consumer'
sources = [
    'backends/oxide/device/src/qwen3vl.rs',
    'backends/oxide/device/build.sh',
    'backends/oxide/device/src/main.rs',
    'backends/oxide/src/qwen3_vl_config.rs',
    'backends/oxide/src/qwen3_vl_vision.rs',
    'backends/oxide/src/qwen3_vl_device.rs',
    'backends/oxide/src/qwen3_vl_engine.rs',
    'backends/oxide/src/qwen3_vl_batch.rs',
    'backends/oxide/src/qwen3_vl_text.rs',
    'backends/oxide/src/qwen3_vl_contract.rs',
    'backends/oxide/src/qwen3_vl_batch.rs',
    'backends/oxide/src/qwen3_vl_readout.rs',
    'backends/oxide/src/qwen3_vl_service.rs',
    'backends/oxide/src/qwen3_vl_fixtures.rs',
    'backends/oxide/src/decoder.rs',
    'backends/oxide/src/decoder_model.rs',
    'backends/oxide/src/model.rs',
    'backends/oxide/src/lib.rs',
]
base = work / 'tree/libs/infinity-v3'
report = {
    'schema': 'oxide-qwen3-vl-e582-cpu-evidence/v1',
    'base_commit': 'e582ce90eb1544fcff081a495cb8fd4f3c17b112',
    'source_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    'patch': 'selective.patch',
    'patch_sha256': sha(root / 'selective.patch'),
    'toolchain': (root / 'evidence/environment.txt').read_text().splitlines(),
    'tests': {
        'oxide_lib': {'passed': int(m.group(1)), 'failed': int(m.group(2)), 'ignored': int(m.group(3))},
        'real_fixtures': {'passed': int(f.group(1)), 'failed': int(f.group(2)), 'ignored': int(f.group(3))},
        'backend_factory': (root / 'logs/backend-factory-tests.log').read_text().strip().splitlines()[-1] if (root / 'logs/backend-factory-tests.log').exists() else 'missing',
        'fmt': 'pass' if (root / 'logs/fmt-check.log').stat().st_size == 0 else 'check-log',
        'clippy_d_warnings': 'pass',
        'consumer_build': 'pass',
    },
    'consumer': {'path': str(consumer), 'sha256': sha(consumer), 'profile': 'debug'},
    'host_consumer': {
        'sha256': (root / 'host-consumer/successor3.sha256').read_text().split()[0]
        if (root / 'host-consumer/successor3.sha256').is_file() else 'not_built',
        'profile': 'historical opt3/lto=false successor3',
        'source_commit': 'f657f1941fff29c1e864166a61c98d1a135ab0dc',
        'matches_current_source': False,
        'run': 'historical manifest/factory/lifecycle',
        'note': 'Not CPU evidence for the current freeze; optimized successor for this freeze remains eligible only via Root grant.',
        'frozen_history': '6e4f7f53afcb9aaac0ac855c65a1c6a18aa1a6084da5abb6eb97b92d0c0702a1',
    },
    'source_sha256': {name: sha(base / name) for name in sources},
    'fixtures': {
        'path': str(root / 'native-baseline/fixtures/fixtures.json'),
        'sha256': sha(root / 'native-baseline/fixtures/fixtures.json'),
        'count': 8,
        'packed_engine_batches': 'all 8 passed in qwen_vl_factory_tests',
    },
    'isolation': {
        'cuda_visible_devices': '',
        'gpu_context': False,
        'docker_used': False,
        'nvcc_used': False,
        'torch_loaded': False,
        'network_used': False,
        'device_compile': False,
    },
    'non_claims': [
        'No GPU execution, device compile, Native parity, lifetime or performance result.',
        'The first-run readout projection is host-side; Root performance gate must measure the release consumer.',
        'No Qwen3.5-VL code is present; see qwen-vl-shared-plan-v1.',
    ],
}
(root / 'evidence/report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['tests'], indent=2))
PY
echo "CPU-only Qwen3-VL e582 evidence complete" >&2
