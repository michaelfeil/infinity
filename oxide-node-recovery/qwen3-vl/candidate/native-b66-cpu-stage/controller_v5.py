#!/usr/bin/env python3
"""Qwen3-VL 027b359c real SDK CPU + paired whole-model controller (v5, Native max_tokens 4096).

CPU phase: CPU-hidden, stages only controller/fixtures/consumers, reads /bench
models read-only, proves b66/vendor GOT, runs whole-model Native CPU for the 8B
packet and the Oxide host consumer manifest/factory/lifecycle.
Quality phase: explicit Root GPU0 env and explicit Native/Oxide/lifecycle jobs.

No checkpoint weights are copied. Old controller.py is frozen history.
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

ROOT = Path(__file__).resolve().parent.parent
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
E7_COMMIT = "027b359c6e0c5378cd07398b07b4435af3ee492d"
HOST_CONSUMER = "/tmp/opencode/qwen-vl-host-consumer-e7baa-target/release/examples/qwen3_vl_host_consumer"
HOST_CONSUMER_SHA = "01516290d5dc97960121a6da086d81e1ad6be0a800eed2c941fe1a3b4e51107f"
QUALITY_CONSUMER = "/tmp/opencode/qwen-vl-host-consumer-quality-target/release/examples/qwen3_vl_host_consumer"
def _quality_sha():
    path = ROOT / "host-consumer/quality-binary.sha256"
    return path.read_text().split()[0] if path.is_file() else None
QUALITY_CONSUMER_SHA = _quality_sha()
NATIVE_CONSUMER = "/tmp/opencode/qwen-vl-native-release-target/release/qwen3-vl-b66-consumer"
NATIVE_CONSUMER_SHA = "89bd6274a26c62fa719ad442d8c2dfeb06e195cb0731117cb005437127c1b671"
FIXTURES_MANIFEST = ROOT / "native-baseline/fixtures/fixtures.json"
FIXTURE_ROOT = ROOT / "native-baseline/fixtures"
REMOTE_BASE = "/tmp/qwen3-vl-native-stage-v2"
CPU_CPUS = "160-167"
GPU_CPUS = "184-191"
GUARD_SECONDS = 1800
HOST_DEADLINE = 1860
PACKET_8B = ROOT / "native-b66-cpu-stage/Root-control-packet.native-v5.json"
DOCKER_SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")
CASES = [
    {"name": "qwen3-vl-8b", "host": "/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B", "container": "/bench/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B", "packet": PACKET_8B},
    {"name": "colqwen3-4b", "host": "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b", "container": "/bench/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b", "packet": None},
    {"name": "nemotron-4b", "host": "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2", "container": "/bench/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2", "packet": None},
    {"name": "nemotron-8b-optional", "host": "/tmp/perplexity-embedding-audit/multivector-support/nvidia--nemotron-colembed-vl-8b-v2", "container": None, "packet": None, "skip": "no verified /bench mapping; weights are not copied"},
]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_env():
    env = dict(os.environ)
    for key in DOCKER_SELECTORS:
        env.pop(key, None)
    return env

def container_env(phase):
    if phase == "cpu":
        cuda = ""
    elif phase == "quality":
        cuda = "0"
    else:
        raise ValueError(f"unsupported phase {phase}")
    values = {
        "CUDA_VISIBLE_DEVICES": cuda,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "LD_BIND_NOW": "1",
        "LD_PRELOAD": f"{VENDOR}/libcublas.so.13:{VENDOR}/libcublasLt.so.13",
        "LD_LIBRARY_PATH": f"{VENDOR}:/tmp/native214-fa4-mode20/install/lib:/opt/torch-sdk/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64",
    }
    if phase == "quality":
        values["INFINITY_FA4_ROOT"] = FA4_ROOT
        values["INFINITY_OXIDE_KERNEL_MODULE"] = UNION124
    out = []
    for key, value in values.items():
        out += ["-e", f"{key}={value}"]
    return out

def docker(*args, capture=False, timeout=300):
    return subprocess.run([*DOCKER, *map(str, args)], env=clean_env(), capture_output=capture,
                          text=capture, timeout=timeout, check=False)

def guard_command(remote, phase, label, cpus, inner):
    state = str(remote / f"state-{label}.json")
    return [
        *DOCKER, "exec", *container_env(phase), CONTAINER,
        "setsid", "--wait", SDK_PY, str(remote / "remote_guard.py"),
        "--state", state, "--seconds", str(GUARD_SECONDS),
        "--", "taskset", "-c", cpus, *map(str, inner),
    ]

def source_pins():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if commit != E7_COMMIT:
        raise RuntimeError(f"source commit {commit} != frozen {E7_COMMIT}")
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "libs"], text=True).strip()
    if dirty:
        raise RuntimeError(f"source changed after freeze: {dirty}")
    handshake = Path("/tmp/oxide-vl-lifecycle-review-v1/QWEN3VL-SOURCE-READY.json")
    ready = json.loads(handshake.read_text())
    if ready.get("source_commit") != E7_COMMIT:
        raise RuntimeError("readiness source commit mismatch")
    return {"source_commit": commit, "readiness": str(handshake),
            "file_pins": ready["file_pins"]}

def pins():
    host_checks = {}
    for role, path, expected in [
        ("host_consumer", HOST_CONSUMER, HOST_CONSUMER_SHA),
        ("native_consumer", NATIVE_CONSUMER, NATIVE_CONSUMER_SHA),
        ("quality_consumer", QUALITY_CONSUMER, QUALITY_CONSUMER_SHA),
        ("fixtures_manifest", str(FIXTURES_MANIFEST), None),
        ("union124_manifest", f"{UNION124}/manifest.json", UNION_MANIFEST_SHA),
        ("fa4_sm90_manifest", f"{FA4_ROOT}/sm_90/manifest.json", FA4_SM90_MANIFEST_SHA),
    ]:
        p = Path(path)
        present = p.is_file()
        actual = sha(p) if present else None
        if expected is not None and present and actual != expected:
            raise RuntimeError(f"{role} pin mismatch: {actual} != {expected}")
        if role in ("host_consumer", "native_consumer") and actual != expected:
            raise RuntimeError(f"{role} missing or mismatched: {actual} != {expected}")
        if role == "quality_consumer" and present and expected is not None and actual != expected:
            raise RuntimeError(f"quality consumer pin mismatch: {actual} != {expected}")
        host_checks[role] = {"path": path, "present_on_host": present, "sha256": actual,
                             "expected_sha256": expected,
                             "pending_build": role == "quality_consumer" and not present}
    fixture = json.loads(FIXTURES_MANIFEST.read_text())
    fixture_pins = {}
    for item in fixture["fixtures"]:
        for image in item.get("images", []):
            p = Path(image["pixels"])
            if not p.is_file():
                raise RuntimeError(f"missing fixture pixel {p}")
            fixture_pins[p.name] = sha(p)
        for key in ("ids", "cumulative"):
            if not item.get(key):
                raise RuntimeError(f"fixture {item['name']} lacks {key}")
    controller_files = {}
    for name in ("controller_v4.py", "test_controller_v4.py", "load_probe.py", "test_load_probe.py", "native_vl_cpu.py", "verify_b66.py", "verify_b66_v4.py", "remote_guard.py"):
        controller_files[name] = sha(ROOT / "native-b66-cpu-stage" / name)
    controller_files["root-gate/compare_quality.py"] = sha(ROOT / "root-gate/compare_quality.py")
    controller_files["root-gate/test_compare_quality.py"] = sha(ROOT / "root-gate/test_compare_quality.py")
    report = {
        "schema": "oxide-qwen3-vl-sdk-controller-pins/v5",
        "container": CONTAINER,
        "b66": {"path": B66, "sha256": B66_SHA, "container_only": True},
        "vendor": {"path": VENDOR, "cublaslt": VENDOR_VERSION, "container_only": True},
        "union124": {"path": UNION124, "manifest_sha256": UNION_MANIFEST_SHA},
        "fa4": {"root": FA4_ROOT, "sm90_manifest_sha256": FA4_SM90_MANIFEST_SHA},
        "host_checks": host_checks,
        "fixture_pins": fixture_pins,
        "controller_files": controller_files,
        "source": source_pins(),
        "old_controller_immutable_sha256": sha(ROOT / "native-b66-cpu-stage/controller.py"),
        "controller_v2_immutable_sha256": sha(ROOT / "native-b66-cpu-stage/controller_v2.py"),
        "controller_v3_immutable_sha256": sha(ROOT / "native-b66-cpu-stage/controller_v3.py"),
        "controller_v4_sha256": sha(ROOT / "native-b66-cpu-stage/controller_v4.py"),
        "quality_consumer_pending_build": not Path(QUALITY_CONSUMER).is_file(),
        "native_packet_v5": {"path": str(PACKET_8B.resolve()), "sha256": sha(PACKET_8B)},
        "no_weight_copy": True,
    }
    report["pins_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return report

def cases():
    fixture = json.loads(FIXTURES_MANIFEST.read_text())
    names = [entry["name"] for entry in fixture["fixtures"]]
    return [dict(case, fixtures=names) for case in CASES if case["container"] is not None]

def plan(run_id, output=None):
    remote = Path(REMOTE_BASE) / run_id
    jobs = []
    for case in cases():
        jobs.append({"case": case["name"], "job": "cpu-bindings", "phase": "cpu", "argv": guard_command(
            remote, "cpu", f"bind-{case['name']}", CPU_CPUS,
            [SDK_PY, str(remote / "verify_b66_v4.py"), "--library-sha", B66_SHA])})
        if case["packet"] is not None:
            jobs.append({"case": case["name"], "job": "cpu-native-whole", "phase": "cpu", "argv": guard_command(
                remote, "cpu", f"native-cpu-{case['name']}", CPU_CPUS,
                [SDK_PY, str(remote / "native_vl_cpu.py"), "--library", B66,
                 "--packet", str(remote / f"{case['name']}.native.json"),
                 "--fixtures", str(remote / "fixtures.json"),
                 "--fixture-root", str(remote / "fixtures"),
                 "--library-sha", B66_SHA,
                 "--out", str(remote / "native-cpu" / f"{case['name']}.json")])})
        jobs.append({"case": case["name"], "job": "cpu-oxide-batch", "phase": "cpu", "argv": guard_command(
            remote, "cpu", f"oxide-cpu-{case['name']}", CPU_CPUS,
            [str(remote / "host-consumer"), "--checkpoint", case["container"],
             "--fixtures", str(remote / "fixtures.json"), "--mode", "all",
             "--out", str(remote / "oxide-cpu" / case["name"])])})
        if case["packet"] is not None:
            jobs.append({"case": case["name"], "job": "quality-native", "phase": "quality", "argv": guard_command(
                remote, "quality", f"quality-native-{case['name']}", GPU_CPUS,
                [str(remote / "native-consumer"), str(remote / f"{case['name']}.native.json"),
                 str(remote / "fixtures.json"), B66,
                 str(remote / "native-quality" / case["name"]),
                 "--library-sha", B66_SHA])})
        jobs.append({"case": case["name"], "job": "quality-vectors", "phase": "quality", "argv": guard_command(
            remote, "quality", f"quality-vectors-{case['name']}", GPU_CPUS,
            [str(remote / "quality-consumer"), "--checkpoint", case["container"],
             "--fixtures", str(remote / "fixtures.json"), "--mode", "gpu-quality",
             "--out", str(remote / "oxide-quality-vectors" / case["name"])])})
        jobs.append({"case": case["name"], "job": "quality-lifecycle-smoke", "phase": "quality", "argv": guard_command(
            remote, "quality", f"quality-life-{case['name']}", GPU_CPUS,
            [str(remote / "host-consumer"), "--checkpoint", case["container"],
             "--fixtures", str(remote / "fixtures.json"), "--mode", "gpu-lifecycle",
             "--out", str(remote / "oxide-lifecycle" / case["name"])])})
    if output is not None:
        out = Path(output)
        for case in cases():
            if case["packet"] is not None:
                jobs.append({"case": case["name"], "job": "quality-compare", "phase": "quality",
                             "kind": "host", "argv": ["python3", str(ROOT / "root-gate/compare_quality.py"),
                             "--native", str(out / "native-quality" / case["name"] / "results.json"),
                             "--oxide", str(out / "oxide-quality-vectors" / case["name"]),
                             "--fixtures", str(FIXTURES_MANIFEST), "--allow-raw",
                             "--out", str(out / "quality" / f"{case['name']}.json")]})
    skipped = [case for case in CASES if case["container"] is None]
    for case in CASES:
        if case["container"] is not None and case["packet"] is None:
            skipped.append({"name": case["name"], "status": "native_packet_missing",
                            "reason": "no Native quality packet; Oxide vectors still produced"})
    return {"schema": "oxide-qwen3-vl-sdk-controller-plan/v6", "run_id": run_id,
            "remote": str(remote), "container": CONTAINER, "no_weight_copy": True,
            "read_only_models": [case["container"] for case in cases()],
            "skipped": skipped,
            "quality_consumer_pending_build": not Path(QUALITY_CONSUMER).is_file(),
        "native_packet_v5": {"path": str(PACKET_8B.resolve()), "sha256": sha(PACKET_8B)},
            "pins_sha256": pins()["pins_sha256"], "jobs": jobs}

def stage(remote):
    if not str(remote).startswith(REMOTE_BASE + "/"):
        raise RuntimeError("remote stage must be a fresh directory under the owned base")
    code = f"from pathlib import Path;Path({str(remote)!r}).mkdir(parents=True)"
    done = docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, SDK_PY, "-c", code, capture=True)
    if done.returncode:
        raise RuntimeError(f"remote mkdir failed: {done.stderr}")
    with tempfile.TemporaryDirectory(prefix="qv3-stage-") as tmp:
        tmp = Path(tmp)
        subprocess.run(["cp", "-a", str(ROOT / "native-b66-cpu-stage") + "/.", str(tmp)], check=True)
        subprocess.run(["cp", "-a", str(FIXTURE_ROOT) + "/.", str(tmp / "fixtures")], check=True)
        subprocess.run(["cp", HOST_CONSUMER, str(tmp / "host-consumer")], check=True)
        subprocess.run(["cp", NATIVE_CONSUMER, str(tmp / "native-consumer")], check=True)
        if Path(QUALITY_CONSUMER).is_file():
            subprocess.run(["cp", QUALITY_CONSUMER, str(tmp / "quality-consumer")], check=True)
        fixture = json.loads(FIXTURES_MANIFEST.read_text())
        for item in fixture["fixtures"]:
            for image in item.get("images", []):
                image["pixels"] = str(remote / "fixtures" / Path(image["pixels"]).name)
        (tmp / "fixtures.json").write_text(json.dumps(fixture, indent=2) + "\n")
        for case in cases():
            if case["packet"] is None:
                continue
            packet = json.loads(Path(case["packet"]).read_text())
            packet["checkpoint"] = case["container"]
            packet["fixtures"] = str(remote / "fixtures.json")
            (tmp / f"{case['name']}.native.json").write_text(json.dumps(packet, indent=2) + "\n")
        done = docker("cp", str(tmp) + "/.", f"{CONTAINER}:{remote}", capture=True)
        if done.returncode:
            raise RuntimeError(f"docker cp failed: {done.stderr}")
    return True

def retrieve(remote, output, job):
    mapping = {
        "cpu-bindings": [f"b66-binding-proof.json"],
        "cpu-native-whole": [f"native-cpu/{job['case']}.json"],
        "cpu-oxide-batch": [f"oxide-cpu/{job['case']}"],
        "quality-native": [f"native-quality/{job['case']}"],
        "quality-vectors": [f"oxide-quality-vectors/{job['case']}"],
        "quality-lifecycle-smoke": [f"oxide-lifecycle/{job['case']}"],
    }
    for relative in mapping.get(job["job"], []):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        done = docker("cp", f"{CONTAINER}:{remote}/{relative}", str(destination), capture=True)
        if done.returncode:
            raise RuntimeError(f"artifact retrieve failed {relative}: {done.stderr}")

def select_jobs(plan_doc, phase, jobs):
    if phase != "all":
        plan_doc["jobs"] = [job for job in plan_doc["jobs"] if job["phase"] == phase]
    if phase == "cpu" and jobs == "all":
        # V5 FAST CPU default: never run the whole-model 8B CPU forward.
        jobs = "cpu-bindings,cpu-oxide-batch"
    selected = {x.strip() for x in jobs.split(",") if x.strip()} if jobs != "all" else None
    if selected is not None:
        plan_doc["jobs"] = [job for job in plan_doc["jobs"] if job["job"] in selected]
        if not plan_doc["jobs"]:
            raise SystemExit(f"no jobs matched {sorted(selected)}")
    return plan_doc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cpu", "quality", "all"), default="cpu")
    parser.add_argument("--jobs", default="all", help="all or comma-separated job names")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--pins", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pins:
        print(json.dumps(pins(), indent=2)); return
    run_id = args.run_id or f"{args.phase}-{time.time_ns()}"
    plan_doc = plan(run_id, args.output)
    plan_doc = select_jobs(plan_doc, args.phase, args.jobs)
    if args.plan or not args.run:
        print(json.dumps(plan_doc, indent=2)); return
    if os.geteuid() != 0:
        raise SystemExit("Docker phases are Root-only")
    if args.output is None or args.output.exists():
        raise SystemExit("--run requires a fresh --output directory outside the source tree")
    if any(job["job"] == "quality-vectors" for job in plan_doc["jobs"]) and not Path(QUALITY_CONSUMER).is_file():
        raise SystemExit("quality consumer is not built; optimized quality-output build is pending")
    remote = Path(plan_doc["remote"])
    stage(remote)
    report = {"schema": "oxide-qwen3-vl-sdk-controller-results/v2", "run_id": run_id,
              "pins_sha256": plan_doc["pins_sha256"], "jobs": []}
    for job in plan_doc["jobs"]:
        completed = subprocess.run(job["argv"], env=clean_env(), capture_output=True, text=True, timeout=HOST_DEADLINE)
        record = {"case": job["case"], "job": job["job"], "phase": job["phase"],
                  "returncode": completed.returncode, "stdout": completed.stdout,
                  "stderr": completed.stderr}
        if completed.returncode == 0 and job.get("kind") != "host":
            retrieve(remote, args.output, job)
            record["retrieved"] = True
        report["jobs"].append(record)
        if completed.returncode:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            raise SystemExit(f"job failed: {job['case']} {job['job']}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(args.output / "report.json")}))

if __name__ == "__main__":
    main()
