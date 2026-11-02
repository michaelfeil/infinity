#!/usr/bin/env python3
"""Real CPU behaviour proof for the bounded controller and harness packet.

No Docker/GPU/Torch/network/compiler is used. Controller parser, pins, plan,
comparator, staging/retrieval argv and failure return codes are exercised directly.
"""
from __future__ import annotations
import contextlib, importlib.util, io, json, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
CONTROLLER = ROOT / "root-v1/qwen35_vl_bounded_controller.py"
SDK = ROOT / "root-v1/qwen35_vl_sdk_controller_v2.py"
FIXTURES = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen3-vl-v3/native-baseline/fixtures/fixtures.json")
checks = []

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run(name, func):
    try:
        func()
        checks.append({"name": name, "passed": True})
    except BaseException as error:
        checks.append({"name": name, "passed": False, "error": str(error)})
        raise

def one(name):
    def register(func):
        run(name, func)
        return func
    return register

@one("argparse_help")
def _():
    done = subprocess.run([sys.executable, str(CONTROLLER), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "usage" in done.stdout

@one("argparse_rejects_bad_phase")
def _():
    done = subprocess.run([sys.executable, str(CONTROLLER), "--phase", "bogus"], capture_output=True, text=True)
    assert done.returncode != 0 and "invalid choice" in done.stderr

@one("argparse_requires_run")
def _():
    done = subprocess.run([sys.executable, str(CONTROLLER), "--phase", "cpu"], capture_output=True, text=True)
    assert done.returncode != 0 and "--run is required" in done.stderr

@one("argparse_requires_output")
def _():
    done = subprocess.run([sys.executable, str(CONTROLLER), "--phase", "cpu", "--run"], capture_output=True, text=True)
    assert done.returncode != 0 and "--output FRESHDIR is required" in done.stderr

@one("argparse_requires_media")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run([sys.executable, str(CONTROLLER), "--phase", "paired", "--run", "--output", str(Path(tmp) / "out")], capture_output=True, text=True)
    assert done.returncode != 0 and "--media is required" in done.stderr

@one("pins_real_fixture")
def _():
    bc = load("bc_pins", CONTROLLER)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        bc.phase_pins(SimpleNamespace(consumer=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"), media=FIXTURES))
    report = json.loads(stream.getvalue())
    assert report["missing"] == []
    assert report["manifest"] == "56863d52270f04e2a72c07f0213295cb78cd003ec111239bef1f78e75ea720c5"
    assert report["fa4_manifest"] == "0e4af6e1fd40714300649dd98e6eb31a0c59e9e3cb493b377de07d57d63916a9"
    assert report["gdn_manifest"] == "f0b2a18d14c80eec9e1e881d87bff80c27408c77a07eb9a1039ed57548b74a7b"

@one("plan_matches_gate_argv")
def _():
    bc = load("bc_plan", CONTROLLER)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        bc.phase_plan(SimpleNamespace())
    plan = json.loads(stream.getvalue())
    gate = json.loads((ROOT / "root-v1/ROOT-GATE-V8.json").read_text())
    assert plan["phases"] == {name: body["command"] for name, body in gate["phases"].items()}

@one("remote_env_roles")
def _():
    bc = load("bc_env", CONTROLLER)
    env = bc.remote_env("/tmp/oxide-qwen35vl-bounded/test", 0)
    pairs = dict(env[index + 1].split("=", 1) for index, value in enumerate(env) if value == "-e" and index + 1 < len(env))
    expected = {"INFINITY_OXIDE_KERNEL_MODULE": "/tmp/oxide-qwen35vl-bounded/test/runtime/kernels",
                "INFINITY_FA4_ROOT": "/tmp/oxide-qwen35vl-bounded/test/runtime/fa4",
                "INFINITY_OXIDE_GDN_RAW_KERNELS": "/tmp/oxide-qwen35vl-bounded/test/runtime/gdn-raw",
                "INFINITY_OXIDE_GDN_PROVIDER": "/tmp/oxide-qwen35vl-bounded/test/runtime/gdn-provider",
                "INFINITY_OXIDE_GDN_AUX_KERNELS": "/tmp/oxide-qwen35vl-bounded/test/runtime/gdn-aux"}
    for key, value in expected.items():
        assert pairs[key] == value
    assert pairs["CUDA_VISIBLE_DEVICES"] == "0"

@one("comparator_real_fixture_pixels")
def _():
    bc = load("bc_cmp", CONTROLLER)
    data = json.loads(FIXTURES.read_text())
    item = next(x for x in data["fixtures"] if x["name"] == "single-cats")
    pixel = Path(item["images"][0]["pixels"])
    values = [float.fromhex("0x0p+0")] if False else list(__import__("struct").unpack("<8f", pixel.read_bytes()[:32]))
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / "native.json"; oxide = Path(tmp) / "oxide.json"; out = Path(tmp) / "pair.json"
        native.write_text(json.dumps({"rows": [{"index": 0, "kind": "pooled", "values": values}]}))
        oxide.write_text(json.dumps({"rows": [{"index": 0, "kind": "pooled", "values": values}]}))
        bc.compare_whole(native, oxide, out)
        assert json.loads(out.read_text())["status"] == "passed"

@one("comparator_rejects_invalid_fields")
def _():
    bc = load("bc_cmp_bad", CONTROLLER)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def write(name, rows):
            path = tmp / name; path.write_text(json.dumps({"rows": rows})); return path
        native = write("n.json", [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]}, {"index": 1, "kind": "pooled", "values": [3.0]}])
        cases = {
            "missing": [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]}],
            "extra": [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]}, {"index": 1, "kind": "pooled", "values": [3.0]}, {"index": 2, "kind": "pooled", "values": [4.0]}],
            "empty": [{"index": 0, "kind": "pooled", "values": []}, {"index": 1, "kind": "pooled", "values": [3.0]}],
            "nan": [{"index": 0, "kind": "pooled", "values": [float("nan"), 2.0]}, {"index": 1, "kind": "pooled", "values": [3.0]}],
            "empty_logits": [{"index": 0, "kind": "decision", "logits": []}, {"index": 1, "kind": "decision", "logits": [1.0]}],
        }
        for name, rows in cases.items():
            oxide = write(name + ".json", rows)
            try:
                bc.compare_whole(native, oxide, tmp / (name + "-out.json"))
            except RuntimeError:
                continue
            raise AssertionError(f"comparator accepted {name}")

@one("checked_cp_actual_argv_and_failure")
def _():
    bc = load("bc_cp", CONTROLLER)
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=(1 if len(calls) == 1 else 0), stderr="mock retrieval failed")
    with mock.patch.object(bc, "docker", fake):
        try:
            bc.checked_cp("host-file", "remote-file")
            raise AssertionError("checked_cp accepted rc=1")
        except RuntimeError:
            pass
        bc.checked_cp("host-file", "remote-file")
    assert calls[0] == ("cp", "host-file", f"{bc.CONTAINER}:remote-file")

@one("run_container_actual_argv_and_failure")
def _():
    bc = load("bc_run", CONTROLLER)
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=2, stdout="out", stderr="failure")
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.log"
        with mock.patch.object(bc, "docker", fake):
            try:
                bc.run_container("/tmp/remote", 0, ["bin", "arg"], log)
                raise AssertionError("run_container accepted rc=2")
            except RuntimeError:
                pass
        assert log.exists() and "failure" in log.read_text()
    assert calls[0][0] == "exec" and bc.CONTAINER in calls[0] and "taskset" in calls[0]

@one("retrieve_actual_argv_and_failure")
def _():
    bc = load("bc_retrieve", CONTROLLER)
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stderr="mock retrieval failed")
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(bc, "docker", fake):
            try:
                bc.retrieve("remote/out", Path(tmp) / "out")
                raise AssertionError("retrieve accepted rc=1")
            except RuntimeError:
                pass
    assert calls[0] == ("cp", f"{bc.CONTAINER}:remote/out", str(Path(tmp) / "out"))

@one("phase_cpu_uses_checked_v2")
def _():
    bc = load("bc_cpu", CONTROLLER)
    calls = []
    def fake(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(bc.subprocess, "run", fake):
            bc.phase_cpu(SimpleNamespace(output=Path(tmp) / "out"))
    assert calls[0][1].endswith("qwen35_vl_sdk_controller_v2.py")

@one("sdk_v2_retrieval_nonzero_fails")
def _():
    sdk = load("sdk_v2", SDK)
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=3, stderr="mock copy failed")
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(sdk, "docker", fake):
            try:
                sdk.retrieve_case("remote", "label", Path(tmp) / "out")
                raise AssertionError("retrieve_case accepted rc=3")
            except RuntimeError:
                pass
    assert calls[0] == ("cp", f"{sdk.CONTAINER}:remote/out-label", str(Path(tmp) / "out"))

@one("gate_controller_pin_matches_actual")
def _():
    gate = json.loads((ROOT / "root-v1/ROOT-GATE-V8.json").read_text())
    actual = __import__("hashlib").sha256(CONTROLLER.read_bytes()).hexdigest()
    assert gate["controller"]["sha256"] == actual
    assert gate["controller"]["sdk_controller_v2"]["sha256"] == __import__("hashlib").sha256(SDK.read_bytes()).hexdigest()

@one("harness_metadata")
def _():
    harness = json.loads((ROOT / "device-packet/CONSUMER-HARNESS.json").read_text())
    gate = json.loads((ROOT / "root-v1/ROOT-GATE-V8.json").read_text())
    assert harness["required_binary"]["built"] is True
    built = Path(harness["required_binary"]["path"])
    assert built.is_file() and __import__("hashlib").sha256(built.read_bytes()).hexdigest() == harness["required_binary"]["sha256"]
    assert harness["mode"] == "owning-lifecycle"
    assert gate["phases"]["owning_life"]["command"][-1] == gate["phases"]["owning_life"]["output"]
    assert gate["consumer"]["exact_build_command"] == harness["exact_build_command"]

failed = [row for row in checks if not row["passed"]]
report = {"schema": "oxide-qwen35vl-bounded-cpu-proof/v2", "status": "passed" if not failed else "failed",
          "checks": checks, "check_count": len(checks),
          "controller_sha256": __import__("hashlib").sha256(CONTROLLER.read_bytes()).hexdigest(),
          "sdk_v2_sha256": __import__("hashlib").sha256(SDK.read_bytes()).hexdigest(),
          "fixtures_sha256": __import__("hashlib").sha256(FIXTURES.read_bytes()).hexdigest()}
out = ROOT / "root-v1/out/bounded-packet-cpu-proof.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "checks": len(checks), "out": str(out)}))
if failed:
    raise SystemExit(1)
