#!/usr/bin/env python3
"""Root-only actual-SDK controller for Qwen3.5-VL whole-model capture.

CPU phase is executable without a GPU and never creates an engine. It stages the
untouched native_whole_model.py b66 C-API implementation plus actual image/video
fixtures into the existing infinity-torch214-compare-sdk container, validates
Torch/GOT/b66/union124/model/fixture identity, and runs plan-only mRoPE. Quality
phase is prepared for Root GPU0.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "infinity-torch214-compare-sdk"
DOCKER = ("docker", "--host=unix:///var/run/docker.sock")
PYTHON = "/opt/torch-sdk/bin/python3.12"
B66 = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
VENDOR = "/tmp/oxide-host-cublas-130200"
UNION124 = "/tmp/infinity-oxide-parity-20261101/device-union124-all3-v1/kernels"
REMOTE_BASE = "/tmp/oxide-qwen35vl-sdk"
CPU_LIST = "88-95"
GUARD_SECONDS = 1200
HOST_DEADLINE = 1260
FIXTURES = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen3-vl-v3/native-baseline/fixtures/fixtures.json")
SELECTED_FIXTURES = ("single-cats", "two-images", "three-frame-cats")
CHECKPOINTS = {
    "Qwen3.5-0.8B": "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-0.8B",
    "Qwen3.5-2B": "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-2B",
    "Qwen3.5-4B": "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-4B",
    "Elephant-Qwen3.5-VL": "/workspace/model-performance/michaelfeil/elephant-1",
    "Mammoth-Qwen3.5-VL": "/workspace/model-performance/michaelfeil/onejev-perf/models/mammoth-1",
}
DOCKER_SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_", "INFINITY_", "LD_"))}
    for key in DOCKER_SELECTORS: env.pop(key, None)
    return env

def docker(*args, capture=False, timeout=300):
    return subprocess.run([*DOCKER, *map(str, args)], env=clean_env(), capture_output=capture,
                          text=capture, timeout=timeout, check=False)

def container_env(phase):
    values = {"CUDA_VISIBLE_DEVICES": "" if phase == "cpu" else "0",
              "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_BIND_NOW": "1",
              "LD_PRELOAD": f"{VENDOR}/libcublas.so.13:{VENDOR}/libcublasLt.so.13",
              "LD_LIBRARY_PATH": f"{VENDOR}:/tmp/native214-fa4-mode20/install/lib:/opt/torch-sdk/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64"}
    out = []
    for k, v in values.items(): out += ["-e", f"{k}={v}"]
    return out

def cases():
    source = json.loads(FIXTURES.read_text())
    selected = {x["name"]: x for x in source["fixtures"] if x["name"] in SELECTED_FIXTURES}
    result = []
    for name in SELECTED_FIXTURES:
        item = selected[name]
        images = []
        for image in item["images"]:
            grid = image["grid_thw"]
            images.append({"name": image["name"], "pixels": image["pixels"],
                           "rows": image["rows"], "patch_dim": image["patch_dim"],
                           "grid": grid, "merge": image["merge_size"],
                           "token_start": image["token_start"], "token_count": image["token_count"]})
        kind = "video" if images and images[0]["grid"][0] > 1 else "image"
        media = {"text": {"input_ids": item["ids"], "token_type_ids": item["types"],
                          "position_ids": item["positions"],
                          "cumulative_seq_lengths": item["cumulative"],
                          "pooled_indices": [0], "raw_indices": [0]},
                 "fixtures": images}
        result.append({"name": name, "kind": kind, "media": media,
                       "pixels": [image["pixels"] for image in images]})
    return result

def build_cases(directory):
    directory.mkdir(parents=True, exist_ok=True)
    result = cases()
    for case in result:
        path = directory / f"{case['name']}.media.json"
        path.write_text(json.dumps(case["media"], indent=2) + "\n")
        case["manifest"] = path
    return result

def stage(remote, output, staged_cases):
    if not (remote.startswith(REMOTE_BASE + "/") and remote != REMOTE_BASE):
        raise RuntimeError("remote stage must be a fresh run directory under the owned base")
    code = f"from pathlib import Path;p=Path({remote!r});p.mkdir(parents=True)"
    done = docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON, "-c", code, capture=True)
    if done.returncode: raise RuntimeError(f"fresh remote directory failed: {done.stderr}")
    with tempfile.TemporaryDirectory(prefix="oxide-qwen35vl-stage-") as tmp:
        tmp = Path(tmp)
        for name in ("native/native_whole_model.py", "native/sdk/sdk_stage.py", "native/sdk/remote_guard.py"):
            shutil.copy2(ROOT / name, tmp / Path(name).name)
        (tmp / "cases").mkdir()
        pixel_copies = {}
        for case in staged_cases:
            shutil.copy2(case["manifest"], tmp / "cases" / case["manifest"].name)
            for pixel in case["pixels"]:
                if pixel not in pixel_copies:
                    target = tmp / "fixtures" / Path(pixel).name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(pixel, target)
                    pixel_copies[pixel] = target
            data = json.loads((tmp / "cases" / case["manifest"].name).read_text())
            for fixture in data["fixtures"]:
                fixture["pixels"] = f"{remote}/fixtures/{Path(fixture['pixels']).name}"
            (tmp / "cases" / case["manifest"].name).write_text(json.dumps(data, indent=2) + "\n")
        frozen = {str(p.relative_to(tmp)): sha(p) for p in tmp.rglob("*") if p.is_file()}
        (tmp / "stage.json").write_text(json.dumps(frozen, indent=2) + "\n")
        done = docker("cp", str(tmp) + "/.", f"{CONTAINER}:{remote}", capture=True)
        if done.returncode: raise RuntimeError(f"docker cp failed: {done.stderr}")
    return frozen

def retrieve_case(remote, label, out_case):
    done = docker("cp", f"{CONTAINER}:{remote}/out-{label}", str(out_case), capture=True)
    if done.returncode:
        raise RuntimeError(f"retrieval failed for {label} rc={done.returncode}: {done.stderr}")
    return done

def stop_guard(remote, label):
    state = f"{remote}/state-{label}.json"
    return docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON,
                  f"{remote}/remote_guard.py", "--state", state, "--stop", capture=True, timeout=60)

def guarded(remote, phase, label, argv, output):
    state = f"{remote}/state-{label}.json"
    inner = ["setsid", "--wait", PYTHON, f"{remote}/remote_guard.py", "--state", state,
             "--seconds", str(GUARD_SECONDS), "--", "taskset", "-c", CPU_LIST, PYTHON, *map(str, argv)]
    command = [*DOCKER, "exec", *container_env(phase), CONTAINER, *inner]
    log = output / f"{label}.log"
    process = None; primary = None
    try:
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=clean_env(), stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=HOST_DEADLINE)
        if code: primary = RuntimeError(f"{label} failed rc={code}; see {log}")
    except subprocess.TimeoutExpired:
        primary = RuntimeError(f"{label} host deadline exceeded; see {log}")
    except BaseException as error:
        primary = error
    finally:
        try: cleanup = stop_guard(remote, label)
        except BaseException as error: cleanup = {"error": str(error)}
        if process is not None and process.poll() is None:
            try: process.terminate(); process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            except ProcessLookupError: pass
    if primary: raise primary
    if isinstance(cleanup, dict) and "error" in cleanup: raise RuntimeError(cleanup["error"])
    if cleanup.returncode: raise RuntimeError(cleanup.stderr)
    return cleanup

def pins():
    files = {}
    for name in ("native/native_whole_model.py", "native/sdk/sdk_stage.py",
                 "native/sdk/remote_guard.py", "root-v1/qwen35_vl_sdk_controller.py",
                 "device-packet/CONSUMER-PINS.json", "device-packet/FA4-REQUEST.json"):
        files[name] = sha(ROOT / name)
    files["fixture-set/fixtures.json"] = sha(FIXTURES)
    for case in cases():
        for pixel in case["pixels"]: files[f"fixture/{Path(pixel).name}"] = sha(pixel)
    models = {}
    for name, path in CHECKPOINTS.items():
        path = Path(path)
        index = path / "model.safetensors.index.json"
        models[name] = {"config_sha256": sha(path / "config.json"),
                        "index_sha256": sha(index) if index.is_file() else None,
                        "path": str(path)}
    union_manifest = json.loads((Path(UNION124) / "manifest.json").read_text())
    union = {"manifest_sha256": sha(Path(UNION124) / "manifest.json"),
             "architectures": {entry["architecture"]: len(entry["symbols_verified"])
                               for entry in union_manifest["architectures"]}}
    report = {"files": files, "models": models, "union124": union,
              "consumer": json.loads((ROOT / "device-packet/CONSUMER-PINS.json").read_text())}
    return report, hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def run(args):
    if os.geteuid() != 0: raise RuntimeError("Docker phases are Root-only")
    output = Path(args.output).expanduser().resolve()
    if output.exists(): raise RuntimeError(f"fresh output directory required: {output}")
    if output.is_relative_to(ROOT): raise RuntimeError("actual output must be outside the source tree")
    output.mkdir(parents=True)
    run_id = f"{args.phase}-{time.time_ns()}"
    remote = f"{REMOTE_BASE}/{run_id}"
    staged_cases = build_cases(output / "cases")
    report = {"schema": "oxide-qwen35vl-sdk-controller/v1", "phase": args.phase,
              "run_id": run_id, "remote": remote, "container": CONTAINER,
              "GPU_executed": args.phase == "quality", "cases": [], "pins": pins()[1]}
    persist = lambda: (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    persist()
    try:
        report["stage"] = stage(remote, output, staged_cases)
        persist()
        for model_name, model_path in CHECKPOINTS.items():
            for case in staged_cases:
                label = f"{args.phase}-{model_name}-{case['name']}".replace("/", "_")
                out_case = output / f"{model_name}-{case['name']}"
                out_case.mkdir()
                remote_media = f"{remote}/cases/{case['manifest'].name}"
                if args.phase == "cpu":
                    argv = [f"{remote}/sdk_stage.py", "--library", B66, "--torch-python", PYTHON,
                            "--vendor", VENDOR, "--union124", UNION124, "--model", model_path,
                            "--media", remote_media, "--out", f"{remote}/out-{label}"]
                    guarded(remote, "cpu", label + "-sdk", argv, output)
                    argv = [f"{remote}/native_whole_model.py", "--plan-only", "--library", B66,
                            "--model", model_path, "--media", remote_media, "--out", f"{remote}/out-{label}"]
                    guarded(remote, "cpu", label + "-plan", argv, output)
                    retrieve_case(remote, label, out_case)
                    report["cases"].append({"model": model_name, "fixture": case["name"],
                                            "output": str(out_case), "status": "cpu-stage-passed"})
                else:
                    argv = [f"{remote}/native_whole_model.py", "--allow-gpu", "--library", B66,
                            "--model", model_path, "--media", remote_media, "--out", f"{remote}/out-{label}"]
                    if model_name in ("Elephant-Qwen3.5-VL", "Mammoth-Qwen3.5-VL"):
                        argv.append("--decision")
                    guarded(remote, "quality", label, argv, output)
                    retrieve_case(remote, label, out_case)
                    report["cases"].append({"model": model_name, "fixture": case["name"],
                                            "output": str(out_case), "status": "native-quality-cpu-pending"})
                persist()
        report["status"] = ("root-cpu-stage-passed-actual-sdk-quality-pending"
                            if args.phase == "cpu" else
                            "root-native-quality-complete-paired-oxide-pending")
        persist()
        print(json.dumps({"status": report["status"], "output": str(output)}))
    except BaseException as error:
        report.update(status="failed", error=str(error)); persist(); raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cpu", "quality"), default="cpu")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pins", action="store_true")
    args = parser.parse_args()
    if args.pins:
        print(pins()[1]); return
    if not args.run:
        print(json.dumps({"phase": args.phase, "pins": pins()[1],
                          "status": "plan-only; add --run for the actual SDK container"})); return
    if args.output is None: raise SystemExit("--output FRESHDIR is required with --run")
    run(args)

if __name__ == "__main__":
    main()
