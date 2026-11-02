#!/usr/bin/env python3
"""Bounded Qwen3-VL Rust-only cold quality + owning-life controller (v1).

One Rust GPU0 job runs the reviewed unified host consumer in
`--mode gpu-quality-life` (single cold Qwen3VlBackend load, all 8 fixtures,
quality pools/raw plus the full owning-life contract). Two host CPU checks then
run: the existing strict Native comparator against the already-retrieved Native
V6 gold, and a strict owning-life report/artifact check. No Native job is run.

The controller preserves the existing QV3 V6 stage behavior and fixed container
libraries. It stages only an explicit manifest of files plus the new unified
binary; the 24 fixed runtime paths are verified, not copied. It never kills
broad processes: the inner remote_guard owns deadline/cleanup, and the caller
owns the host subprocess deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen-vl-current-v1")
CANDIDATE = Path("/tmp/oxide-qwen3vl-rope-budget-fix-v1/candidate")
CONTROLLER = Path(__file__).resolve()
WORKSPACE = CONTROLLER.parent

CONTAINER = "infinity-torch214-compare-sdk"
DOCKER = ("docker", "--host=unix:///var/run/docker.sock")
SDK_PY = "/opt/torch-sdk/bin/python3.12"
B66 = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
B66_SHA = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
VENDOR = "/tmp/oxide-host-cublas-130200"
VENDOR_VERSION = 130200
UNION124 = "/tmp/infinity-oxide-parity-20261101/device-union124-all3-v1/kernels"
UNION_MANIFEST_SHA = "56863d52270f04e2a72c07f0213295cb78cd003ec111239bef1f78e75ea720c5"
FA4_ROOT = "/tmp/fa4-root"
FA4_SM90_MANIFEST_SHA = "6d4c580ea4965ef94c6379d88c62eb7ee09462409f21dc4fd81f73ee080e566d"
RUNTIME_PROOF = Path("/tmp/oxide-qwen3vl-runtime-staging-v1/ROOT-STAGING-PROOF.json")
REMOTE_BASE = "/tmp/qwen3-vl-native-stage-v2"
CPU_CPUS = "160-167"
GPU_CPUS = "184-191"
GUARD_SECONDS = 1800
HOST_DEADLINE = 1860
DOCKER_SELECTORS = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_TLS",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
)

UNIFIED_CONSUMER = Path(
    "/tmp/oxide-qwen3vl-rope-budget-fix-v1/artifacts/"
    "qwen3_vl_host_consumer-a2c02adb"
)
UNIFIED_CONSUMER_SHA = "a2c02adb32445d6bb36fd5d2056ce34d63af629ce83f83598cd64829d63de100"
UNIFIED_CONSUMER_SIZE = 24599072
UNIFIED_CONSUMER_BUILD_ID = "ee2161d7e08268867506db0337303f39bc1b7310"

NATIVE_GOLD = Path(
    "/tmp/oxide-qwen3vl-root-quality-v6/native-quality/qwen3-vl-8b/results.json"
)
NATIVE_GOLD_SHA = "776c66cf0d0bccc6e6f1679ed3a28e197da8dac61135f5b329587558abd67dc2"
NATIVE_GOLD_DIR = NATIVE_GOLD.parent
NATIVE_GOLD_OBSERVATIONS = 8

FIXTURES_MANIFEST = ROOT / "native-baseline/fixtures/fixtures.json"
FIXTURES_MANIFEST_SHA = "0190631e9389cf65a398534bdce0f4f457b4522dbcd3d192ef7c456321f24374"
FIXTURE_ROOT = ROOT / "native-baseline/fixtures"
PACKET_8B = ROOT / "native-b66-cpu-stage/Root-control-packet.native-v5.json"
PACKET_8B_SHA = "ac90c550bb0acd068c18dbd5ed35b0b5de577ca663f528100c3a3c2f0ff81244"
REMOTE_GUARD = ROOT / "native-b66-cpu-stage/remote_guard.py"
REMOTE_GUARD_SHA = "a11cf9ee05b971c8a427bf671ae2b16b776c06a447644d6d94b7cf1754f627b1"
COMPARE = ROOT / "root-gate/compare_quality.py"
COMPARE_SHA = "8e57c14db58cd9ed41a0cac017afdddd239b27fc3eafc94351ee12e4f4c3506c"
COMPARE_TEST = ROOT / "root-gate/test_compare_quality.py"
COMPARE_TEST_SHA = "e25bba9df6bdfae57eede50be88a1817c4d4d10bc7980f0d3b00e20f0f3f862f"

E7_COMMIT = "027b359c6e0c5378cd07398b07b4435af3ee492d"
CASE = {
    "name": "qwen3-vl-8b",
    "host": "/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B",
    "container": "/bench/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B",
}
SOURCE_CANDIDATE = {
    "decoder.rs": CANDIDATE / "libs/infinity-v3/backends/oxide/src/decoder.rs",
    "rotary.rs": CANDIDATE / "libs/infinity-v3/backends/oxide/src/rotary.rs",
    "qwen3_vl_host_consumer.rs": CANDIDATE
    / "libs/infinity-v3/backends/oxide/examples/qwen3_vl_host_consumer.rs",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(document):
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def clean_env():
    env = dict(os.environ)
    for key in DOCKER_SELECTORS:
        env.pop(key, None)
    return env


def container_env(phase):
    if phase != "quality":
        raise ValueError(f"unsupported phase {phase}")
    values = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LD_BIND_NOW": "1",
        "LD_PRELOAD": f"{VENDOR}/libcublas.so.13:{VENDOR}/libcublasLt.so.13",
        "LD_LIBRARY_PATH": (
            f"{VENDOR}:/tmp/native214-fa4-mode20/install/lib:"
            "/opt/torch-sdk/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64"
        ),
        "INFINITY_FA4_ROOT": FA4_ROOT,
        "INFINITY_OXIDE_KERNEL_MODULE": UNION124,
    }
    out = []
    for key, value in values.items():
        out += ["-e", f"{key}={value}"]
    return out


def docker(*args, capture=False, timeout=300):
    return subprocess.run(
        [*DOCKER, *map(str, args)],
        env=clean_env(),
        capture_output=capture,
        text=capture,
        timeout=timeout,
        check=False,
    )


def guard_command(remote, label, inner):
    state = str(remote / f"state-{label}.json")
    return [
        *DOCKER,
        "exec",
        *container_env("quality"),
        CONTAINER,
        "setsid",
        "--wait",
        SDK_PY,
        str(remote / "remote_guard.py"),
        "--state",
        state,
        "--seconds",
        str(GUARD_SECONDS),
        "--",
        "taskset",
        "-c",
        GPU_CPUS,
        *map(str, inner),
    ]


def runtime_paths():
    proof = json.loads(RUNTIME_PROOF.read_text())
    entries = []
    for relative, expected in proof["files"].items():
        if relative.startswith("kernels/"):
            host = Path(UNION124) / relative[len("kernels/") :]
        elif relative.startswith("fa4/"):
            host = Path(FA4_ROOT) / relative[len("fa4/") :]
        else:
            raise RuntimeError(f"unexpected runtime proof path {relative}")
        entries.append(
            {
                "proof_path": relative,
                "host_path": str(host),
                "sha256": expected,
            }
        )
    if len(entries) != 24:
        raise RuntimeError(f"runtime proof must contain 24 paths, got {len(entries)}")
    return entries


def verify_runtime_paths():
    checked = []
    for entry in runtime_paths():
        path = Path(entry["host_path"])
        if not path.is_file():
            raise RuntimeError(f"missing runtime path {path}")
        actual = sha(path)
        if actual != entry["sha256"]:
            raise RuntimeError(f"runtime path hash mismatch {path}: {actual}")
        checked.append({key: entry[key] for key in ("proof_path", "host_path", "sha256")})
    return checked


def fixture_manifest():
    return json.loads(FIXTURES_MANIFEST.read_text())


def fixture_assets():
    fixture = fixture_manifest()
    assets = []
    for item in fixture["fixtures"]:
        for image in item.get("images", []):
            manifest_path = Path(image["pixels"])
            staged_source = FIXTURE_ROOT / manifest_path.name
            if not manifest_path.is_file():
                raise RuntimeError(f"missing manifest fixture {manifest_path}")
            if not staged_source.is_file():
                raise RuntimeError(f"missing frozen fixture {staged_source}")
            manifest_sha = sha(manifest_path)
            frozen_sha = sha(staged_source)
            if manifest_sha != frozen_sha:
                raise RuntimeError(f"fixture bytes disagree for {manifest_path.name}")
            assets.append(
                {
                    "fixture": item["name"],
                    "name": manifest_path.name,
                    "source": str(staged_source),
                    "sha256": frozen_sha,
                }
            )
    if len(assets) != 13:
        raise RuntimeError(f"expected 13 fixture assets, got {len(assets)}")
    return assets


def stage_manifest():
    entries = [
        {
            "role": "unified_consumer",
            "kind": "copy",
            "source": str(UNIFIED_CONSUMER),
            "dest": "unified-consumer",
            "sha256": sha(UNIFIED_CONSUMER),
            "critical": True,
        },
        {
            "role": "remote_guard",
            "kind": "copy",
            "source": str(REMOTE_GUARD),
            "dest": "remote_guard.py",
            "sha256": sha(REMOTE_GUARD),
            "critical": True,
        },
        {
            "role": "packet_metadata",
            "kind": "copy",
            "source": str(PACKET_8B),
            "dest": "packet.native-v5.json",
            "sha256": sha(PACKET_8B),
            "critical": False,
            "used_by_unified_consumer": False,
        },
        {
            "role": "fixtures_json",
            "kind": "generate",
            "source": str(FIXTURES_MANIFEST),
            "dest": "fixtures.json",
            "sha256": None,
            "critical": True,
        },
    ]
    for asset in fixture_assets():
        entries.append(
            {
                "role": "fixture_asset",
                "fixture": asset["fixture"],
                "kind": "copy",
                "source": asset["source"],
                "dest": f"fixtures/{asset['name']}",
                "sha256": asset["sha256"],
                "critical": True,
            }
        )
    for entry in runtime_paths():
        entries.append(
            {
                "role": "runtime_mounted",
                "kind": "runtime",
                "source": None,
                "dest": entry["host_path"],
                "sha256": entry["sha256"],
                "critical": True,
                "copied": False,
            }
        )
    return entries


def native_gold_files():
    report = json.loads(NATIVE_GOLD.read_text())
    files = {}
    for row in report["observations"]:
        name = row["fixture"]
        recorded = [row["output"], row["repeat_output"]]
        recorded += [str(Path(row["output"]).parent / f"{name}.pool{index}.f32") for index in range(3)]
        for record in recorded:
            recorded_path = Path(record)
            local = NATIVE_GOLD_DIR / recorded_path.name
            if recorded_path.is_file():
                actual = sha(recorded_path)
                if not local.is_file() or sha(local) != actual:
                    raise RuntimeError(f"recorded Native gold differs from retrieved {record}")
            if not local.is_file():
                raise RuntimeError(f"missing retrieved Native gold file {local}")
            files[local.name] = sha(local)
    if len(report["observations"]) != NATIVE_GOLD_OBSERVATIONS:
        raise RuntimeError("Native V6 gold observation count mismatch")
    if len(files) != 40:
        raise RuntimeError(f"expected 40 Native gold f32 files, got {len(files)}")
    return files


def pins():
    if sha(UNIFIED_CONSUMER) != UNIFIED_CONSUMER_SHA:
        raise RuntimeError("unified consumer SHA mismatch")
    if UNIFIED_CONSUMER.stat().st_size != UNIFIED_CONSUMER_SIZE:
        raise RuntimeError("unified consumer size mismatch")
    if sha(NATIVE_GOLD) != NATIVE_GOLD_SHA:
        raise RuntimeError("Native V6 gold SHA mismatch")
    if sha(FIXTURES_MANIFEST) != FIXTURES_MANIFEST_SHA:
        raise RuntimeError("fixtures manifest SHA mismatch")
    if sha(PACKET_8B) != PACKET_8B_SHA:
        raise RuntimeError("Native packet SHA mismatch")
    if sha(REMOTE_GUARD) != REMOTE_GUARD_SHA:
        raise RuntimeError("remote_guard SHA mismatch")
    if sha(COMPARE) != COMPARE_SHA:
        raise RuntimeError("compare_quality SHA mismatch")
    if sha(COMPARE_TEST) != COMPARE_TEST_SHA:
        raise RuntimeError("compare_quality test SHA mismatch")
    if sha(Path(UNION124) / "manifest.json") != UNION_MANIFEST_SHA:
        raise RuntimeError("union124 manifest SHA mismatch")
    if sha(Path(FA4_ROOT) / "sm_90/manifest.json") != FA4_SM90_MANIFEST_SHA:
        raise RuntimeError("fa4 sm90 manifest SHA mismatch")
    runtime = verify_runtime_paths()
    assets = fixture_assets()
    gold = native_gold_files()
    source = {name: sha(path) for name, path in SOURCE_CANDIDATE.items()}
    report = {
        "schema": "oxide-qwen3vl-rope-budget-life-controller-pins/v1",
        "controller": {"path": str(CONTROLLER), "sha256": sha(CONTROLLER)},
        "unified_consumer": {
            "path": str(UNIFIED_CONSUMER),
            "sha256": sha(UNIFIED_CONSUMER),
            "expected_sha256": UNIFIED_CONSUMER_SHA,
            "size_bytes": UNIFIED_CONSUMER.stat().st_size,
            "expected_size_bytes": UNIFIED_CONSUMER_SIZE,
            "build_id": UNIFIED_CONSUMER_BUILD_ID,
            "profile": "opt3/lto=false/debug1/cgu16/debug-assertions=false/strip=none",
        },
        "source_candidate": {
            "path": str(CANDIDATE),
            "commit": E7_COMMIT,
            "files": source,
        },
        "native_v6_gold": {
            "results": {"path": str(NATIVE_GOLD), "sha256": sha(NATIVE_GOLD)},
            "dir": str(NATIVE_GOLD_DIR),
            "observations": NATIVE_GOLD_OBSERVATIONS,
            "f32_files": gold,
            "reused": True,
            "native_rerun": False,
        },
        "fixtures": {
            "manifest": {"path": str(FIXTURES_MANIFEST), "sha256": sha(FIXTURES_MANIFEST)},
            "assets": assets,
        },
        "packet_metadata": {"path": str(PACKET_8B), "sha256": sha(PACKET_8B)},
        "remote_guard": {"path": str(REMOTE_GUARD), "sha256": sha(REMOTE_GUARD)},
        "comparator": {
            "path": str(COMPARE),
            "sha256": sha(COMPARE),
            "tests": {"path": str(COMPARE_TEST), "sha256": sha(COMPARE_TEST)},
        },
        "runtime_paths_24": runtime,
        "libraries": {
            "b66": {"path": B66, "sha256": B66_SHA, "container_only": True},
            "vendor": {"path": VENDOR, "cublaslt": VENDOR_VERSION, "container_only": True},
            "union124": {"path": UNION124, "manifest_sha256": UNION_MANIFEST_SHA},
            "fa4": {"root": FA4_ROOT, "sm90_manifest_sha256": FA4_SM90_MANIFEST_SHA},
        },
        "container": {"name": CONTAINER, "sdk_python": SDK_PY},
        "stage_manifest": stage_manifest(),
        "no_weight_copy": True,
        "no_native_job": True,
        "no_broad_process_kill": True,
    }
    report["pins_sha256"] = canonical_hash(report)
    return report


def plan(run_id, output):
    if not run_id:
        raise RuntimeError("run_id is required")
    remote = Path(REMOTE_BASE) / run_id
    out = Path(output)
    oxide_out = remote / "oxide-quality-life" / CASE["name"]
    jobs = [
        {
            "case": CASE["name"],
            "job": "quality-life-vectors",
            "phase": "quality",
            "kind": "container",
            "argv": guard_command(
                remote,
                "quality-life-qwen3-vl-8b",
                [
                    str(remote / "unified-consumer"),
                    "--checkpoint",
                    CASE["container"],
                    "--fixtures",
                    str(remote / "fixtures.json"),
                    "--mode",
                    "gpu-quality-life",
                    "--out",
                    str(oxide_out),
                ],
            ),
        },
        {
            "case": CASE["name"],
            "job": "quality-compare-native-v6-gold",
            "phase": "quality",
            "kind": "host",
            "argv": [
                "taskset",
                "-c",
                CPU_CPUS,
                "python3",
                str(COMPARE),
                "--native",
                str(NATIVE_GOLD),
                "--oxide",
                str(out / "oxide-quality-life" / CASE["name"]),
                "--fixtures",
                str(FIXTURES_MANIFEST),
                "--allow-raw",
                "--out",
                str(out / "quality" / f"{CASE['name']}.json"),
            ],
        },
        {
            "case": CASE["name"],
            "job": "quality-owning-life-contract",
            "phase": "quality",
            "kind": "host",
            "argv": [
                "taskset",
                "-c",
                CPU_CPUS,
                "python3",
                str(CONTROLLER),
                "--check-life-report",
                str(out / "oxide-quality-life" / CASE["name"] / "gpu-owning-life.json"),
                "--check-life-out",
                str(out / "quality" / f"{CASE['name']}-owning-life.json"),
            ],
        },
    ]
    return {
        "schema": "oxide-qwen3vl-rope-budget-life-controller-plan/v1",
        "run_id": run_id,
        "remote": str(remote),
        "output": str(out),
        "container": CONTAINER,
        "case": CASE,
        "no_weight_copy": True,
        "no_native_job": True,
        "one_cold_backend_load": True,
        "mode": "gpu-quality-life",
        "fixtures": 8,
        "native_gold": str(NATIVE_GOLD),
        "pins_sha256": pins()["pins_sha256"],
        "jobs": jobs,
    }


def stage(remote):
    remote = Path(remote)
    if not str(remote).startswith(REMOTE_BASE + "/"):
        raise RuntimeError("remote stage must be a fresh directory under the owned base")
    code = f"from pathlib import Path;Path({str(remote)!r}).mkdir(parents=True)"
    done = docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, SDK_PY, "-c", code, capture=True)
    if done.returncode:
        raise RuntimeError(f"remote mkdir failed: {done.stderr}")
    with tempfile.TemporaryDirectory(prefix="qv3-budget-life-") as tmp_name:
        tmp = Path(tmp_name)
        copied = []
        for entry in stage_manifest():
            if entry["kind"] == "runtime":
                continue
            if entry["kind"] == "generate":
                fixture = fixture_manifest()
                for item in fixture["fixtures"]:
                    for image in item.get("images", []):
                        image["pixels"] = str(remote / "fixtures" / Path(image["pixels"]).name)
                target = tmp / entry["dest"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(fixture, indent=2) + "\n")
                copied.append({"dest": entry["dest"], "sha256": sha(target)})
                continue
            source = Path(entry["source"])
            target = tmp / entry["dest"]
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", str(source), str(target)], check=True)
            if sha(target) != entry["sha256"]:
                raise RuntimeError(f"staged file hash mismatch {entry['dest']}")
            copied.append({"dest": entry["dest"], "sha256": sha(target)})
        done = docker("cp", str(tmp) + "/.", f"{CONTAINER}:{remote}", capture=True)
        if done.returncode:
            raise RuntimeError(f"docker cp failed: {done.stderr}")
    return {"remote": str(remote), "copied": copied, "runtime_verified": verify_runtime_paths()}


def retrieve(remote, output):
    destination = Path(output) / "oxide-quality-life" / CASE["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    done = docker(
        "cp",
        f"{CONTAINER}:{Path(remote) / 'oxide-quality-life' / CASE['name']}",
        str(destination),
        capture=True,
    )
    if done.returncode:
        raise RuntimeError(f"artifact retrieve failed: {done.stderr}")
    fixtures_copy = Path(output) / "staged-fixtures.json"
    done = docker("cp", f"{CONTAINER}:{Path(remote) / 'fixtures.json'}", str(fixtures_copy), capture=True)
    if done.returncode:
        raise RuntimeError(f"fixtures retrieve failed: {done.stderr}")
    return destination


def check_life(report_path, out_path):
    report_path = Path(report_path)
    out_path = Path(out_path)
    report = json.loads(report_path.read_text())
    combined_path = report_path.parent / "gpu-quality-life.json"
    combined = json.loads(combined_path.read_text()) if combined_path.is_file() else None
    manifest = fixture_manifest()
    expected = {item["name"] for item in manifest["fixtures"]}
    failures = []

    if report.get("schema") != "qwen3-vl-host-consumer-gpu-owning-life/v1":
        failures.append("wrong owning-life schema")
    if report.get("route") != "Qwen3VlBackend::submit_embed":
        failures.append("wrong backend route")
    if report.get("backend_constructions") != 1:
        failures.append("owning-life report does not prove one backend construction")
    if report.get("fixture_count") != 8:
        failures.append("owning-life fixture count is not 8")
    for key in (
        "third_submit_while_two_held",
        "drop_without_completion_fenced",
        "complete_b_exact",
        "invalid_media_typed",
        "invalid_media_zero_selectors",
        "health_after_recovery",
    ):
        expected_value = False if key == "third_submit_while_two_held" else True
        if report.get(key) is not expected_value:
            failures.append(f"{key} is not {expected_value}")
    if report.get("changed_shape", {}).get("exact") is not True:
        failures.append("changed-shape exact check failed")
    if report.get("recovered", {}).get("exact") is not True:
        failures.append("recovery exact check failed")
    held = report.get("held", {})
    if not held.get("a_shape") or held.get("a_shape") == held.get("b_shape"):
        failures.append("held A/B shapes are not distinct")
    baselines = report.get("baselines", [])
    names = [row.get("fixture") for row in baselines]
    if len(baselines) != 8 or len(set(names)) != 8 or set(names) != expected:
        failures.append("baseline set is not the exact 8 real fixtures")
    for row in baselines:
        if row.get("values", 0) <= 0:
            failures.append(f"empty baseline {row.get('fixture')}")
        file_info = row.get("file", {})
        if file_info.get("finite") is not True:
            failures.append(f"non-finite baseline {row.get('fixture')}")
        path = report_path.parent / "life-baselines" / Path(file_info.get("path", "")).name
        if not path.is_file():
            failures.append(f"missing baseline artifact {path.name}")
            continue
        if sha(path) != file_info.get("sha256"):
            failures.append(f"baseline artifact hash mismatch {path.name}")
    if combined is None:
        failures.append("missing combined gpu-quality-life report")
    else:
        if combined.get("schema") != "qwen3-vl-host-consumer-gpu-quality-life/v1":
            failures.append("wrong combined quality-life schema")
        if combined.get("backend_constructions") != 1:
            failures.append("combined report does not prove one cold backend load")
        if "quality" not in combined or "owning_life" not in combined:
            failures.append("combined report lacks quality or owning_life")
    rendered = json.dumps(report) + json.dumps(combined or {})
    if "Overloaded" in rendered:
        failures.append("report contains a fake Overloaded label")
    result = {
        "schema": "oxide-qwen3vl-rope-budget-owning-life-contract/v1",
        "passed": not failures,
        "failures": failures,
        "report": str(report_path),
        "combined_report": str(combined_path),
        "fixtures": 8,
        "backend_constructions": 1,
        "native_job_run": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_pins(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = pins()
    (directory / "CONTROLLER-PINS-v1.json").write_text(json.dumps(document, indent=2) + "\n")
    (directory / "STAGE-PATHS-v1.json").write_text(
        json.dumps(document["stage_manifest"], indent=2) + "\n"
    )
    return document


def preflight(run_id, output):
    pins_document = pins()
    plan_document = plan(run_id, output)
    stage_entries = stage_manifest()
    return {
        "schema": "oxide-qwen3vl-rope-budget-life-preflight/v1",
        "pins_sha256": pins_document["pins_sha256"],
        "unified_consumer": pins_document["unified_consumer"],
        "native_gold": pins_document["native_v6_gold"],
        "fixtures": pins_document["fixtures"],
        "runtime_paths_24": pins_document["runtime_paths_24"],
        "stage_entries": len(stage_entries),
        "copied_entries": sum(1 for entry in stage_entries if entry["kind"] != "runtime"),
        "runtime_entries": sum(1 for entry in stage_entries if entry["kind"] == "runtime"),
        "jobs": plan_document["jobs"],
        "no_native_job": True,
        "gpu_used_by_preflight": False,
        "docker_used_by_preflight": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pins", action="store_true")
    parser.add_argument("--write-pins", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-out", type=Path)
    parser.add_argument("--phase", default="quality")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-life-report", type=Path)
    parser.add_argument("--check-life-out", type=Path)
    args = parser.parse_args()

    if args.phase != "quality":
        raise SystemExit("only the quality Rust-first phase is supported")
    if args.check_life_report is not None:
        if args.check_life_out is None:
            raise SystemExit("--check-life-report requires --check-life-out")
        result = check_life(args.check_life_report, args.check_life_out)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.write_pins is not None:
        write_pins(args.write_pins)
        if not (args.pins or args.preflight or args.plan or args.run):
            return
    if args.pins:
        print(json.dumps(pins(), indent=2))
        return
    if args.preflight:
        if args.output is None:
            raise SystemExit("--preflight requires --output")
        result = preflight(args.run_id, args.output)
        if args.preflight_out is not None:
            args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    if args.plan or not args.run:
        print(json.dumps(plan(args.run_id, args.output), indent=2))
        return
    if os.geteuid() != 0:
        raise SystemExit("Docker phases are Root-only")
    if args.output is None or args.output.exists():
        raise SystemExit("--run requires a fresh --output directory outside the source tree")
    pins_document = pins()
    plan_document = plan(args.run_id, args.output)
    remote = Path(plan_document["remote"])
    staged = stage(remote)
    report = {
        "schema": "oxide-qwen3vl-rope-budget-life-controller-results/v1",
        "run_id": args.run_id,
        "pins_sha256": pins_document["pins_sha256"],
        "unified_consumer": pins_document["unified_consumer"],
        "native_gold": pins_document["native_v6_gold"],
        "staged": staged,
        "jobs": [],
    }
    for job in plan_document["jobs"]:
        completed = subprocess.run(
            job["argv"],
            env=clean_env(),
            capture_output=True,
            text=True,
            timeout=HOST_DEADLINE,
        )
        record = {
            "case": job["case"],
            "job": job["job"],
            "phase": job["phase"],
            "kind": job["kind"],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        report["jobs"].append(record)
        if completed.returncode == 0 and job["kind"] == "container":
            retrieve(remote, args.output)
            record["retrieved"] = True
        if completed.returncode:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit(f"job failed: {job['job']}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
