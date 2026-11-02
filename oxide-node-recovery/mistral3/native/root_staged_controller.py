#!/usr/bin/env python3
"""FINAL source-only HOST controller for Mistral3 Native SDK stages.

Stages only allowlisted source/header/scripts into the existing
`infinity-torch214-compare-sdk` container. It never copies weights, starts a
container, edits Docker config/proxy/CA, or modifies the SDK/weights.

  cpu      actual 531 metadata + full SDK load + 19-arg CAPI/header + vendor
           GOT proof before/after, CVD empty, no engine
  quality  CPU preflight, then actual b66 CAPI quality vectors on GPU0
  perf     CPU preflight, then actual b66 CAPI perf windows on GPU0

Every container command runs under the inner owned remote_guard with exact
PID/argv/start-time cleanup. Quality/perf require an independent review
clearance file. No GPU is executed by the author.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTAINER_DEFAULT = "infinity-torch214-compare-sdk"
CONTAINER_PYTHON = "/opt/torch-sdk/bin/python3.12"
DOCKER_HOST = "unix:///var/run/docker.sock"
SDK_DEFAULT = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
SDK_SHA_DEFAULT = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
VENDOR_DEFAULT = "/tmp/oxide-host-cublas-130200"
HOST_MODEL_DEFAULT = "/workspace/model-performance/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16"
CONTAINER_MODEL_DEFAULT = "/bench/michaelfeil/benchmark-overview/models/nvidia--Nemotron-3.5-Rerank-8B-BF16"
ALLOWLIST = [
    "src/mistral3_runner.cpp",
    "include/mistral3_b66.h",
    "include/bridge.h",
    "native_b66_control.py",
    "container_sdk_got.py",
    "remote_guard.py",
    "root_native_cpu_stage.py",
]
DOCKER_CLEAR = ["DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def docker_env(base: dict | None = None) -> dict:
    env = dict(base or os.environ)
    for key in DOCKER_CLEAR:
        env.pop(key, None)
    return env

def docker_argv(*args: str) -> list[str]:
    return ["docker", f"--host={DOCKER_HOST}", *args]

def run_host(command: list[str], timeout: int = 300, env: dict | None = None) -> str:
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env or docker_env())
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout

def docker(*args: str, timeout: int = 300) -> str:
    return run_host(docker_argv(*args), timeout=timeout)

def runtime_env(sdk: str, vendor: str, cvd: str) -> dict[str, str]:
    torch_lib = "/opt/torch-sdk/lib:/opt/torch-sdk/lib/python3.12/site-packages/torch/lib"
    return {
        "CUDA_VISIBLE_DEVICES": cvd,
        "LD_BIND_NOW": "1",
        "LD_LIBRARY_PATH": f"{vendor}:{Path(sdk).parent}:{torch_lib}",
        "LD_PRELOAD": f"{vendor}/libcublasLt.so.13:{vendor}/libcublas.so.13",
    }

def container_cmd(container: str, work: str, argv: list[str], env: dict[str, str], timeout: int) -> str:
    command = docker_argv("exec", "-w", work)
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += [container, *argv]
    return run_host(command, timeout=timeout)

def guarded(container: str, work: str, name: str, argv: list[str], env: dict[str, str], seconds: int) -> str:
    state = f"{work}/{name}-guard.json"
    command = [
        CONTAINER_PYTHON, f"{work}/remote_guard.py",
        "--state", state, "--seconds", str(seconds),
        "--", "taskset", "-c", "136-143", *argv,
    ]
    try:
        return container_cmd(container, work, command, env, seconds + 60)
    except subprocess.TimeoutExpired:
        try:
            container_cmd(container, work, [CONTAINER_PYTHON, f"{work}/remote_guard.py", "--state", state, "--stop"], env, 60)
        except Exception:
            pass
        raise

def preflight_host(host_model: Path) -> dict:
    script = ROOT / "root_native_cpu_stage.py"
    proc = subprocess.run([sys.executable, str(script), "--preflight-only", "--checkpoint", str(host_model)], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"host metadata preflight failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)

def stage_requires_gpu(stage: str) -> bool:
    return stage in ("quality", "perf")

def build_runner_argv(work: str, sdk: str) -> list[str]:
    return [
        "g++", "-std=c++20", "-O2", f"-I{work}/include",
        f"{work}/src/mistral3_runner.cpp",
        "-L", str(Path(sdk).parent), "-ltei_torch",
        f"-Wl,-rpath,{Path(sdk).parent}",
        "-o", f"{work}/mistral3-b66-runner",
    ]

def got_argv(work: str, sdk: str, sha_value: str, vendor: str) -> list[str]:
    return [
        CONTAINER_PYTHON, f"{work}/container_sdk_got.py",
        "--sdk", sdk, "--runner", f"{work}/mistral3-b66-runner",
        "--expected-sha", sha_value, "--vendor", vendor,
        "--bridge-header", f"{work}/include/bridge.h",
    ]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cpu", "quality", "perf"], default="cpu")
    ap.add_argument("--container", default=CONTAINER_DEFAULT)
    ap.add_argument("--host-model", type=Path, default=Path(HOST_MODEL_DEFAULT))
    ap.add_argument("--container-model", default=CONTAINER_MODEL_DEFAULT)
    ap.add_argument("--sdk", default=SDK_DEFAULT)
    ap.add_argument("--sdk-sha256", default=SDK_SHA_DEFAULT)
    ap.add_argument("--vendor", default=VENDOR_DEFAULT)
    ap.add_argument("--work", default="")
    ap.add_argument("--host-out", type=Path, default=None)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--review-clearance", type=Path)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    preflight = preflight_host(args.host_model)
    if args.preflight_only:
        print(json.dumps({"preflight": preflight, "allowlist": ALLOWLIST, "container_python": CONTAINER_PYTHON, "docker_host": DOCKER_HOST}, indent=2))
        return 0
    if not args.run:
        print(json.dumps({"preflight": preflight, "stage": args.stage, "allowlist": ALLOWLIST, "container_python": CONTAINER_PYTHON, "docker_host": DOCKER_HOST}, indent=2))
        return 0
    if args.stage == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        print("refusing: CPU stage requires CUDA_VISIBLE_DEVICES=''", file=sys.stderr); return 2
    if stage_requires_gpu(args.stage):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            print("refusing: quality/perf requires Root CUDA_VISIBLE_DEVICES=0", file=sys.stderr); return 2
        if not args.review_clearance or not args.review_clearance.is_file():
            print("refusing: quality/perf requires an independent review clearance file", file=sys.stderr); return 2
        clearance = json.loads(args.review_clearance.read_text())
        if clearance.get("independent") is not True or clearance.get("status") != "approved":
            print("refusing: review clearance is not independent/approved", file=sys.stderr); return 2
    if args.host_out is None:
        print("refusing: --host-out is required for --run", file=sys.stderr); return 2
    if args.host_out.exists() and any(args.host_out.iterdir()):
        print(f"refusing non-fresh --host-out {args.host_out}", file=sys.stderr); return 2
    args.host_out.mkdir(parents=True, exist_ok=True)

    work = args.work or f"/tmp/mistral3-v4-stage-{os.getpid()}-{int(time.time())}"
    docker("exec", args.container, "mkdir", "-p", f"{work}/src", f"{work}/include", f"{work}/cpu-control", timeout=60)
    for name in ALLOWLIST:
        host = ROOT / name
        if not host.is_file():
            raise RuntimeError(f"allowlisted staged file is missing: {host}")
        docker("cp", str(host), f"{args.container}:{work}/{name}", timeout=120)
    staged = {name: sha(ROOT / name) for name in ALLOWLIST}
    pins_before = {
        "host": preflight,
        "staged": staged,
        "container_config": docker("exec", args.container, "sha256sum", f"{args.container_model}/config.json", timeout=60).split()[0],
        "container_index": docker("exec", args.container, "sha256sum", f"{args.container_model}/model.safetensors.index.json", timeout=60).split()[0],
        "sdk_expected": args.sdk_sha256,
    }
    (args.host_out / "pins-before.json").write_text(json.dumps(pins_before, indent=2) + "\n")
    env_cpu = runtime_env(args.sdk, args.vendor, "")
    guarded(args.container, work, "build", build_runner_argv(work, args.sdk), env_cpu, 600)
    got_before = json.loads(guarded(args.container, work, "got-before", got_argv(work, args.sdk, args.sdk_sha256, args.vendor), env_cpu, 600))
    cpu_env = {**env_cpu, "INFINITY_MISTRAL3_MODEL": args.container_model, "INFINITY_MISTRAL3_SDK": args.sdk, "INFINITY_MISTRAL3_CONTROL_OUT": f"{work}/cpu-control"}
    metadata = json.loads(guarded(args.container, work, "metadata", [CONTAINER_PYTHON, f"{work}/native_b66_control.py"], cpu_env, args.seconds))
    got_after = json.loads(guarded(args.container, work, "got-after", got_argv(work, args.sdk, args.sdk_sha256, args.vendor), env_cpu, 600))
    docker("cp", f"{args.container}:{work}/cpu-control", str(args.host_out / "cpu-control"), timeout=120)
    result = {
        "schema": "oxide-mistral3-staged-controller/v1",
        "stage": args.stage,
        "container": args.container,
        "docker_host": DOCKER_HOST,
        "container_python": CONTAINER_PYTHON,
        "work": work,
        "host_model": str(args.host_model),
        "container_model": args.container_model,
        "staged_files": staged,
        "sdk": args.sdk,
        "sdk_sha256": args.sdk_sha256,
        "got_before": got_before,
        "metadata": metadata,
        "got_after": got_after,
        "gpu_execution": False,
        "review_clearance": str(args.review_clearance) if args.review_clearance else None,
        "errors": [],
    }
    if stage_requires_gpu(args.stage):
        vectors = f"{work}/native-gpu-vectors" if args.stage == "quality" else ""
        runner_args = [f"{work}/mistral3-b66-runner", "--checkpoint", args.container_model, "--device", "cuda:0", "--iterations", str(args.iterations), "--warmup", str(args.warmup)]
        if vectors:
            runner_args += ["--emit-vectors", vectors]
        gpu = guarded(args.container, work, args.stage, runner_args, runtime_env(args.sdk, args.vendor, "0"), args.seconds)
        native = json.loads(gpu[gpu.find("{"):])
        (args.host_out / "native.json").write_text(json.dumps(native, indent=2) + "\n")
        if vectors:
            docker("cp", f"{args.container}:{vectors}", str(args.host_out / "vectors"), timeout=300)
        if native.get("tensors") != 531:
            result["errors"].append(f"tensors {native.get('tensors')} != 531")
        if native.get("shards") != 4:
            result["errors"].append(f"shards {native.get('shards')} != 4")
        if native.get("graphs") is not True or native.get("graph_policy", {}).get("max_captures") != 8:
            result["errors"].append("Native graph defaults/counters not preserved")
        result.update({"native": native, "vectors": str(args.host_out / "vectors") if vectors else None, "gpu_execution": True})
    pins_after = {
        "container_config": docker("exec", args.container, "sha256sum", f"{args.container_model}/config.json", timeout=60).split()[0],
        "container_index": docker("exec", args.container, "sha256sum", f"{args.container_model}/model.safetensors.index.json", timeout=60).split()[0],
        "sdk_sha256": got_after.get("sdk_sha256"),
        "staged": staged,
    }
    (args.host_out / "pins-after.json").write_text(json.dumps(pins_after, indent=2) + "\n")
    if got_before.get("sdk_sha256") != args.sdk_sha256 or got_after.get("sdk_sha256") != args.sdk_sha256:
        result["errors"].append("SDK sha mismatch before/after")
    if pins_before["container_config"] != pins_after["container_config"] or pins_before["container_index"] != pins_after["container_index"]:
        result["errors"].append("checkpoint config/index changed")
    if not got_before.get("ok") or not got_after.get("ok"):
        result["errors"].append("actual SDK load/GOT proof failed")
    result["ok"] = not result["errors"]
    (args.host_out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
