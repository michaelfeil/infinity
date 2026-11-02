#!/usr/bin/env python3
"""Native-only Root controller (v4): light -> t7_a gold -> full gold/compare.

Existing container only via docker exec/docker cp; in-container guard owns
process-group cleanup. No starter-rust, no host execution. Native gold source is
preserved v3; v4 carries the corrected comparator, tests, light and controller.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_INPUT = ROOT / "input"
SOURCE_INPUT = LOCAL_INPUT if LOCAL_INPUT.is_dir() else (
    ROOT.parent / "input" if (ROOT.parent / "input").is_dir()
    else Path("/tmp/oxide-q35moe-native-runtime-control-v1/input"))
if SOURCE_INPUT.is_dir():
    sys.path.insert(0, str(SOURCE_INPUT))

import native_batch as nb  # noqa: E402

CONTAINER = "infinity-torch214-compare-sdk"
CONTAINER_PYTHON = "/opt/torch-sdk/bin/python3.12"
CONTAINER_BASE = "/tmp/qwen35-moe-sdk-root-v1"
HOST_BASE = Path(CONTAINER_BASE)
MODEL = "/bench/michaelfeil/onejev-perf/models/qwen35-35b-a3b"
NATIVE_LIBRARY = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
VENDOR = "/tmp/oxide-host-cublas-130200"
PHASES = ("light", "starter-native", "quality")
IMPL_FILES = ("native_light.py", "strict_compare.py", "root_control.py",
              "remote_guard.py", "test_control.py", "CONTRACT.json")
GOLD_SOURCE = ROOT / "native_gold.py"
REQUIRED_INPUT_FILES = ("native_base.py", "native_raw_probe.py", "native_batch.py",
                        "model_checkpoint.py", "verify_bindings.py", "container_sdk_got.py",
                        "bridge.h", "bridge.cpp", "qwen35_models.cpp")
PASS_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy",
            "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE")
LIB_PRELOAD = f"{VENDOR}/libcublasLt.so.13:{VENDOR}/libcublas.so.13"
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "LD_BIND_NOW": "1",
           "LD_PRELOAD": LIB_PRELOAD, "LD_LIBRARY_PATH": VENDOR,
           "INFINITY_BENCH_REQUIRE_HOST_VENDOR": "1"}
GPU_ENV = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "LD_BIND_NOW": "1", "LD_PRELOAD": LIB_PRELOAD, "LD_LIBRARY_PATH": VENDOR,
           "INFINITY_BENCH_REQUIRE_HOST_VENDOR": "1",
           "INFINITY_TORCH_CUDA_GRAPHS": "1",
           "INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES": "8",
           "INFINITY_TORCH_CUDA_GRAPH_ENTRIES": "4",
           "INFINITY_TORCH_CUDA_GRAPH_REUSE": "3",
           "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES": "512MiB",
           "INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS": "4096",
           "INFINITY_TORCH_FP8_SCOPE": "mlp"}


class ControlError(RuntimeError):
    pass


def fresh_runid(runid):
    if runid is None:
        return f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    if not runid or runid in (".", "..") or "/" in runid or "\\" in runid:
        raise ControlError(f"invalid runid: {runid!r}")
    return runid


def docker_run(args, timeout=120, check=True):
    finished = subprocess.run(["docker"] + args, capture_output=True, text=True,
                              timeout=timeout, env=os.environ.copy())
    if check and finished.returncode != 0:
        raise ControlError(f"docker {' '.join(args[:3])} failed: {finished.stderr.strip()}")
    return finished


def docker_exec(container_dir: str, argv: list[str], env: dict) -> list[str]:
    command = ["docker", "exec", "--workdir", container_dir]
    # Only the fixed, non-sensitive native options can become literal arguments.
    allowed = set(CPU_ENV) | set(GPU_ENV)
    for key in env:
        if key not in allowed or env[key] not in {CPU_ENV.get(key), GPU_ENV.get(key)}:
            raise ControlError("non-fixed native environment option rejected")
    # Docker inherits these values by name; never read or serialize their contents.
    for name in PASS_ENV:
        if name in os.environ:
            command += ["-e", name]
    for key in sorted(env):
        command += ["-e", f"{key}={env[key]}"]
    return command + [CONTAINER] + argv


def _owner(runid: str, host_dir: Path) -> dict:
    return {"schema": "oxide-q35moe-owned-run/v4", "runid": runid,
            "host_dir": str(host_dir), "container": CONTAINER,
            "container_base": CONTAINER_BASE}


def preflight_sources() -> None:
    for name in IMPL_FILES:
        if not (ROOT / name).is_file():
            raise ControlError(f"missing implementation file: {name}")
    if not GOLD_SOURCE.is_file():
        raise ControlError("missing native gold implementation")
    for name in REQUIRED_INPUT_FILES:
        if not (SOURCE_INPUT / name).is_file():
            raise ControlError(f"missing required runtime input: {name}")


def stage_host(runid: str) -> tuple[Path, bool]:
    preflight_sources()
    host_dir = HOST_BASE / fresh_runid(runid)
    reuse = host_dir.exists()
    if reuse:
        owner_path = host_dir / ".owner"
        if not owner_path.is_file():
            raise ControlError(f"refusing unowned host run directory: {host_dir}")
        if json.loads(owner_path.read_text()) != _owner(runid, host_dir):
            raise ControlError(f"owned run marker mismatch: {host_dir}")
    else:
        host_dir.mkdir(parents=True)
        if not SOURCE_INPUT.is_dir():
            raise ControlError(f"missing frozen source input directory: {SOURCE_INPUT}")
        for source in sorted(SOURCE_INPUT.iterdir()):
            if source.is_file():
                shutil.copyfile(source, host_dir / source.name)
        for name in IMPL_FILES:
            source = ROOT / name
            if not source.is_file():
                raise ControlError(f"missing implementation file: {source}")
            shutil.copyfile(source, host_dir / name)
        if not GOLD_SOURCE.is_file():
            raise ControlError(f"missing preserved v3 native gold: {GOLD_SOURCE}")
        shutil.copyfile(GOLD_SOURCE, host_dir / "native_gold.py")
        (host_dir / ".owner").write_text(
            json.dumps(_owner(runid, host_dir), indent=2) + "\n")
    return host_dir, reuse


def container_owned(container_dir: str, host_dir: Path) -> bool:
    exists = docker_run(["exec", CONTAINER, "test", "-e", container_dir], check=False)
    if exists.returncode != 0:
        return False
    owner = docker_run(["exec", CONTAINER, "cat", f"{container_dir}/.owner"], check=False)
    if owner.returncode != 0:
        raise ControlError(f"container run namespace is unowned: {container_dir}")
    if owner.stdout.strip() != json.dumps(_owner(host_dir.name, host_dir), indent=2):
        raise ControlError(f"container run marker mismatch: {container_dir}")
    return True


def stage_container(host_dir: Path, container_dir: str, reuse: bool, extras) -> None:
    docker_run(["exec", CONTAINER, "mkdir", "-p", CONTAINER_BASE])
    exists = container_owned(container_dir, host_dir)
    if reuse:
        if not exists:
            raise ControlError("owned host run exists but container namespace is missing")
        for name in extras:
            docker_run(["cp", str(host_dir / name),
                        f"{CONTAINER}:{container_dir}/{name}"], timeout=120)
        return
    if exists:
        raise ControlError(f"container run namespace already exists: {container_dir}")
    docker_run(["cp", f"{host_dir}/.", f"{CONTAINER}:{container_dir}"], timeout=300)


def copy_back(container_dir: str, host_dir: Path, names) -> None:
    for name in names:
        docker_run(["cp", f"{CONTAINER}:{container_dir}/{name}",
                    str(host_dir / name)], timeout=120)


def stop_guard(container_dir: str, label: str) -> dict:
    command = docker_exec(container_dir,
                          [CONTAINER_PYTHON, f"{container_dir}/remote_guard.py",
                           "--state", f"{container_dir}/guard-{label}.json", "--stop"], {})
    finished = subprocess.run(command, capture_output=True, text=True, timeout=30,
                              env=os.environ.copy())
    if finished.returncode != 0:
        raise ControlError(f"owned guard stop failed: {finished.stderr.strip()}")
    return json.loads(finished.stdout)


def merge_record(record: dict, result: dict) -> dict:
    if result.get("argv") != record.get("argv"):
        raise ControlError("execution result argv differs from the phase record")
    for key, value in result.items():
        if key in record and record[key] != value:
            raise ControlError(f"refusing to merge conflicting result key {key!r}")
        record[key] = value
    return record


def guard_argv(step: dict, container_dir: str, timeout_s: int) -> list[str]:
    return docker_exec(container_dir,
                       [CONTAINER_PYTHON, f"{container_dir}/remote_guard.py",
                        "--state", f"{container_dir}/guard-{step['label']}.json",
                        "--seconds", str(timeout_s), "--"] + list(step["argv"]),
                       step["env"])


def run_bounded(step: dict, host_dir: Path, container_dir: str, timeout_s: int) -> dict:
    argv = guard_argv(step, container_dir, timeout_s)
    primary = cleanup_error = None
    timed_out = False
    proc = subprocess.Popen(argv, cwd=host_dir, env=os.environ.copy(),
                            start_new_session=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            timed_out = True
            primary = ControlError(f"{step['label']} exceeded the guarded deadline")
            try:
                stop_guard(container_dir, step["label"])
            except Exception as error:
                cleanup_error = error
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
        if not timed_out and proc.returncode not in (0, None):
            primary = ControlError(f"{step['label']} exited with status {proc.returncode}")
        result = {"argv": argv, "returncode": proc.returncode, "timed_out": timed_out,
                  "stdout": stdout, "stderr": stderr}
    except Exception as error:
        primary = error
        result = {"argv": argv, "returncode": None, "timed_out": False,
                  "stdout": "", "stderr": ""}
    finally:
        if proc.poll() is None:
            try:
                stop_guard(container_dir, step["label"])
            except Exception as error:
                cleanup_error = cleanup_error or error
            try:
                proc.communicate(timeout=10)
            except Exception as error:
                cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        result["cleanup_error"] = str(cleanup_error)
        if primary is not None:
            primary.add_note(f"cleanup also failed: {cleanup_error}")
    (host_dir / f"{step['label']}.stdout").write_text(result["stdout"])
    (host_dir / f"{step['label']}.stderr").write_text(result["stderr"])
    (host_dir / f"{step['label']}.process.json").write_text(json.dumps(result, indent=2) + "\n")
    if primary is not None:
        raise primary
    return result


def _pins_argv(container_dir: str, output: str) -> list[str]:
    return [CONTAINER_PYTHON, f"{container_dir}/native_light.py", "--container",
            "--pins-only", "--model", MODEL, "--sdk", NATIVE_LIBRARY,
            "--vendor", VENDOR, "--bridge-header", f"{container_dir}/bridge.h",
            "--output", output]


def pins_container(container_dir: str, host_dir: Path, label: str) -> dict:
    name = f"pins-{label}.json"
    output = f"{container_dir}/{name}"
    finished = subprocess.run(docker_exec(container_dir, _pins_argv(container_dir, output),
                                          CPU_ENV),
                              capture_output=True, text=True, timeout=300,
                              env=os.environ.copy())
    if finished.returncode != 0:
        raise ControlError(f"container pins {label} failed: {finished.stderr.strip()}")
    docker_run(["cp", f"{CONTAINER}:{output}", str(host_dir / name)], timeout=120)
    return json.loads((host_dir / name).read_text())


def pin_gate(before: dict, after: dict, phase: str) -> dict:
    stable = (before.get("pins_digest") == after.get("pins_digest")
              and before.get("sources") == after.get("sources")
              and before.get("binaries") == after.get("binaries")
              and before.get("providers") == after.get("providers")
              and before.get("model_config") == after.get("model_config")
              and before.get("model_header") == after.get("model_header")
              and before.get("identity_sha256") == after.get("identity_sha256"))
    if not stable:
        raise ControlError(f"{phase}: actual source/binary/provider/module/config/header pins changed")
    return {"pins_stable": stable, "pins_before_digest": before.get("pins_digest"),
            "pins_after_digest": after.get("pins_digest"),
            "pins_before": before, "pins_after": after}


def select_t7_a(path: Path) -> dict:
    data = json.loads(Path(path).read_bytes())
    cases = data["cases"] if isinstance(data, dict) and "cases" in data else (
        data if isinstance(data, list) else [data])
    if not isinstance(cases, list):
        raise ControlError("fixtures must contain a cases list")
    matches = [case for case in cases
               if isinstance(case, dict) and case.get("case_id") == "t7_a"]
    if len(matches) != 1:
        raise ControlError("starter-native requires exactly one t7_a fixture")
    case = matches[0]
    if case.get("pool") != "mean":
        raise ControlError("t7_a fixture must use mean pooling")
    nb.validate_batch(case.get("batch"))
    return {"cases": [case]}


def phase_plan(args, container_dir: str) -> list[dict]:
    if args.phase == "light":
        return [{"label": "light", "env": CPU_ENV, "copy_back": ["light.json"], "argv": [
            CONTAINER_PYTHON, f"{container_dir}/native_light.py", "--container",
            "--model", MODEL, "--sdk", NATIVE_LIBRARY, "--vendor", VENDOR,
            "--bridge-header", f"{container_dir}/bridge.h",
            "--output", f"{container_dir}/light.json"]}]
    if args.phase == "starter-native":
        return [{"label": "starter-native", "env": GPU_ENV,
                 "copy_back": ["native-gold-t7.json"], "argv": [
            CONTAINER_PYTHON, f"{container_dir}/native_gold.py", "--container",
            "--model", MODEL, "--library", NATIVE_LIBRARY, "--vendor", VENDOR,
            "--bridge-header", f"{container_dir}/bridge.h",
            "--sdk-proof", f"{container_dir}/light.json",
            "--cases", f"{container_dir}/cases.json", "--minimal-t7",
            "--output", f"{container_dir}/native-gold-t7.json"]}]
    if args.phase == "quality":
        if args.cases is None or args.rust_contract is None:
            raise ControlError("quality requires --cases and --rust-contract")
        compare_argv = [CONTAINER_PYTHON, f"{container_dir}/strict_compare.py",
                        "--native", f"{container_dir}/native-gold.json",
                        "--rust", f"{container_dir}/rust-contract.json",
                        "--out", f"{container_dir}/compare.json"]
        if args.manifest is not None:
            compare_argv += ["--manifest", f"{container_dir}/manifest.json"]
        return [{"label": "native-gold", "env": GPU_ENV,
                 "copy_back": ["native-gold.json"], "argv": [
            CONTAINER_PYTHON, f"{container_dir}/native_gold.py", "--container",
            "--model", MODEL, "--library", NATIVE_LIBRARY, "--vendor", VENDOR,
            "--bridge-header", f"{container_dir}/bridge.h",
            "--sdk-proof", f"{container_dir}/light.json",
            "--cases", f"{container_dir}/cases.json",
            "--output", f"{container_dir}/native-gold.json"]},
                {"label": "strict-compare", "env": CPU_ENV,
                 "copy_back": ["compare.json"], "argv": compare_argv}]
    raise ControlError(f"native-only controller has no phase {args.phase!r}")


def check_phase(args, host_dir: Path) -> None:
    if args.phase == "light":
        report = json.loads((host_dir / "light.json").read_text())
        if report.get("schema") != "oxide-q35moe-native-light/v1" or not report.get("ok"):
            raise ControlError("light CPU proof did not pass")
        if report.get("engine_created") or report.get("engine_or_forward_called"):
            raise ControlError("light phase reported an engine")
    elif args.phase == "starter-native":
        report = json.loads((host_dir / "native-gold-t7.json").read_text())
        if report.get("status") != "ok" or report.get("minimal_t7") is not True:
            raise ControlError("minimal t7_a native gold did not pass")
        cases = report.get("cases") or []
        if (len(cases) != 1 or cases[0].get("case_id") != "t7_a"
                or cases[0].get("pool") != "mean" or not cases[0].get("outputs")):
            raise ControlError("minimal t7_a gold must return the t7_a mean vectors")
        if not report.get("identity_stable") or not report.get("sources_stable"):
            raise ControlError("minimal t7_a identity repetition failed")
    elif args.phase == "quality":
        gold = json.loads((host_dir / "native-gold.json").read_text())
        if gold.get("status") != "ok":
            raise ControlError("full native gold did not pass")
        compare = json.loads((host_dir / "compare.json").read_text())
        if compare.get("status") != "UNCHANGED" or not compare.get("unchanged"):
            raise ControlError("strict comparator did not return UNCHANGED")
        if compare.get("numeric_qualified") is not True or compare.get("float_count", 0) <= 0:
            raise ControlError("quality lacks an actual nonempty numeric comparison")
        if not gold.get("identity_stable") or not gold.get("sources_stable"):
            raise ControlError("full gold identity repetition failed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--runid")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--rust-contract", type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    preflight_sources()
    if not 1 <= args.timeout <= 3600:
        raise SystemExit("--timeout must be in 1..3600 seconds")
    runid = fresh_runid(args.runid)
    container_dir = f"{CONTAINER_BASE}/{runid}"
    if not args.run:
        print(json.dumps({"schema": "oxide-q35moe-root-control-plan/v4",
                          "phase": args.phase, "run": False,
                          "mode": "cpu-source-plan-only", "container": CONTAINER,
                          "container_run_dir": container_dir,
                          "commands": phase_plan(args, container_dir),
                          "note": "docker exec existing container only; --run required"},
                         indent=2))
        return 0
    extras = []
    selected_t7 = None
    if args.phase == "starter-native":
        if args.cases is None:
            raise ControlError("starter-native requires --cases")
        selected_t7 = select_t7_a(args.cases)
    elif args.phase == "quality":
        if args.cases is None or args.rust_contract is None:
            raise ControlError("quality requires --cases and --rust-contract")
        extras += ["cases.json", "rust-contract.json"]
        if args.manifest is not None:
            extras.append("manifest.json")
    host_dir, reuse = stage_host(runid)
    if selected_t7 is not None:
        (host_dir / "cases.json").write_text(json.dumps(selected_t7, indent=2) + "\n")
        extras = ["cases.json"]
    elif args.phase == "quality":
        shutil.copyfile(args.cases, host_dir / "cases.json")
        shutil.copyfile(args.rust_contract, host_dir / "rust-contract.json")
        if args.manifest is not None:
            shutil.copyfile(args.manifest, host_dir / "manifest.json")
    stage_container(host_dir, container_dir, reuse, extras)
    plan = phase_plan(args, container_dir)
    pins_before = pins_container(container_dir, host_dir, "before")
    records = []
    for step in plan:
        argv = guard_argv(step, container_dir, args.timeout)
        try:
            result = run_bounded(step, host_dir, container_dir, args.timeout)
        except Exception as primary:
            try:
                copy_back(container_dir, host_dir, step["copy_back"])
            except Exception as cleanup:
                primary.add_note(f"failure evidence copy also failed: {cleanup}")
            raise
        records.append(merge_record({
            "phase": args.phase, "label": step["label"], "argv": argv,
            "container_run_dir": container_dir}, result))
        copy_back(container_dir, host_dir, step["copy_back"])
    pins_after = pins_container(container_dir, host_dir, "after")
    pin_record = pin_gate(pins_before, pins_after, args.phase)
    check_phase(args, host_dir)
    print(json.dumps({"schema": "oxide-q35moe-root-control-result/v4",
                      "phase": args.phase, "run": True, "container": CONTAINER,
                      "container_run_dir": container_dir,
                      "host_staging_dir": str(host_dir), "pins": pin_record,
                      "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
