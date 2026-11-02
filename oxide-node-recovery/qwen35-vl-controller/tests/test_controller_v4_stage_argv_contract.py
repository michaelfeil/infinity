#!/usr/bin/env python3
"""Fresh V4 stage/argv contract regression test.

Calls the actual V4 stage() with only BC.docker mocked at the subprocess
boundary. Records real docker cp argv, then derives the actual SDK/Native argv
builders and asserts their explicit-Python helper paths are exactly the paths
staged. No source-string/AST proof, no Docker/GPU/Torch/network/compile.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "qwen35_vl_map_policy_controller_v4.py"
V4_SHA = "5ce7b5f184ab2cf4e1fff42fd80bc5477c149d42b39d71cf42630c641210181b"
PAIRED = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO/release/qwen35-vl-consumer")
LIFE = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO-harness/release/qwen35-vl-consumer")
SOURCE = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1")
RUST_PINS = {
    "consumer/src/main.rs": "5d7a75d681a882be7149d4e543b9b861bcd5c674b0e416fd301a7d648f8d2a2b",
    "libs/infinity-v3/backends/oxide/src/lib.rs": "aeea32e693ea2d5120841fe4c28ae8527b8c1adcb09b008beab2815dde950721",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_device.rs": "1910837d81bdb0abc73a34c73c1879219b5baca7fe44bcb81fdb234d904ea5fb",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_engine.rs": "83fd2c8b3e233cbb6b42e2442900f809bacdc86d58b6f52932d363c54bb99b00",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_tests.rs": "c1f31d1622bdc548c512163a54802a47fe2bea8cd2f444090be32fe471ce5d0b",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_vision.rs": "b12143546b1ff7738cce854c060f33d4fb5db86a57cdfa9759c215d346a4b42e",
    "libs/infinity-v3/backends/oxide/device/src/main.rs": "c19ae20ff2ec668f1983b65f32992c7912ccca52d27b2c18d84b72503d71f6ae",
    "libs/infinity-v3/backends/oxide/device/src/qwen35vl.rs": "f7fd1a0165c15a8bc043d2c6010951449b17d2100a147e6fb305a89f034a477e",
}
PAIRED_SHA = "3780facf61529822f33a4cf1416bc6c8723acc1fa2cd2c778f7bc54f2140d83b"
LIFE_SHA = "21c5603ee6e9eff06dae5a4344d83fba43ee8bd0ed9a31d462417970c0bf297a"
checks = []
contract_evidence = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BC = load("controller_v4", V4)


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


def recorded_stage():
    remote = BC.REMOTE_BASE + "/stage-argv-contract"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases = BC.build_cases(tmp / "cases", remote)
        model = BC.validate_model_records(["Qwen3.5-0.8B"])["Qwen3.5-0.8B"]
        calls = []
        def fake_docker(*args, **kwargs):
            calls.append(tuple(str(item) for item in args))
            stdout = '{"status":"passed","files":1,"failures":[]}\n'
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        with mock.patch.object(BC, "docker", fake_docker):
            result = BC.stage(remote, PAIRED, cases, manifest_dir=tmp)
        return remote, model, cases, calls, result


@one("v4_sha_stable_before")
def _():
    assert sha(V4) == V4_SHA


@one("stage_records_real_cp_argv_and_sdk_helper_path_matches_staged")
def _():
    remote, model, cases, calls, result = recorded_stage()
    cp_pairs = [(args[1], args[2]) for args in calls if args and args[0] == "cp"]
    staged_targets = {target.split(BC.CONTAINER + ":", 1)[1]
                      for _, target in cp_pairs if target.startswith(BC.CONTAINER + ":")}
    staged_sources = {source for source, _ in cp_pairs}
    sdk_target = f"{remote}/sdk/{BC.SDK_PREFLIGHT.name}"
    assert str(BC.SDK_PREFLIGHT) in staged_sources
    assert sdk_target in staged_targets
    media = f"{remote}/cases/{cases[0]['manifest'].name}"
    argv = BC.sdk_preflight_argv(remote, "Qwen3.5-0.8B", model, media,
                                 f"{remote}/preflight")
    assert argv[0] == BC.PYTHON
    assert argv[1] == sdk_target
    assert argv[1] in staged_targets
    assert result["verified_files"] > 0
    contract_evidence["sdk"] = {"source": str(BC.SDK_PREFLIGHT), "staged": sdk_target,
                                "argv": argv}


@one("native_plan_and_whole_python_helper_filename_matches_staged")
def _():
    remote, model, cases, calls, _ = recorded_stage()
    cp_pairs = [(args[1], args[2]) for args in calls if args and args[0] == "cp"]
    staged_targets = {target.split(BC.CONTAINER + ":", 1)[1]
                      for _, target in cp_pairs if target.startswith(BC.CONTAINER + ":")}
    native_target = f"{remote}/native/{BC.NATIVE_HELPER.name}"
    assert str(BC.NATIVE_HELPER) in {source for source, _ in cp_pairs}
    assert native_target in staged_targets
    media = f"{remote}/cases/{cases[0]['manifest'].name}"
    plan = BC.native_plan_argv(remote, model, media, f"{remote}/plan")
    whole = BC.native_whole_argv(remote, "Qwen3.5-0.8B", model, media, f"{remote}/native")
    assert plan[0] == BC.PYTHON and whole[0] == BC.PYTHON
    assert plan[1] == native_target
    assert whole[1] == native_target
    assert plan[1] in staged_targets and whole[1] in staged_targets
    contract_evidence["native"] = {"source": str(BC.NATIVE_HELPER),
                                   "staged": native_target,
                                   "plan_argv": plan, "whole_argv": whole}


@one("elf_consumer_staged_path_is_used_by_consumer_argv_without_python")
def _():
    remote, model, cases, calls, _ = recorded_stage()
    cp_pairs = [(args[1], args[2]) for args in calls if args and args[0] == "cp"]
    staged_sources = {source for source, _ in cp_pairs}
    staged_targets = {target.split(BC.CONTAINER + ":", 1)[1]
                      for _, target in cp_pairs if target.startswith(BC.CONTAINER + ":")}
    consumer_target = f"{remote}/bin/qwen35-vl-consumer"
    assert str(PAIRED) in staged_sources
    assert consumer_target in staged_targets
    media = f"{remote}/cases/{cases[0]['manifest'].name}"
    whole = BC.oxide_whole_argv(remote, model, media, f"{remote}/oxide.json")
    life = BC.oxide_life_argv(remote, model, media, f"{remote}/life.json")
    assert whole[0] == consumer_target and life[0] == consumer_target
    assert whole[0] != BC.PYTHON and life[0] != BC.PYTHON
    contract_evidence["consumer"] = {"source": str(PAIRED), "staged": consumer_target,
                                     "whole_argv": whole, "life_argv": life}


@one("v4_sha_stable_after")
def _():
    assert sha(V4) == V4_SHA


@one("source8_and_binaries_stable")
def _():
    for rel, expected in RUST_PINS.items():
        assert sha(SOURCE / rel) == expected, rel
        assert sha(ROOT / rel) == expected, rel
    assert sha(PAIRED) == PAIRED_SHA
    assert sha(LIFE) == LIFE_SHA


failed = [row for row in checks if not row["passed"]]
receipt = {
    "schema": "oxide-qwen35vl-controller-v4-stage-argv-contract/v1",
    "status": "passed" if not failed else "failed",
    "test": "tests/test_controller_v4_stage_argv_contract.py",
    "test_sha256": sha(Path(__file__)),
    "v4_controller": {"path": str(V4), "sha256": sha(V4), "expected": V4_SHA},
    "checks": checks,
    "check_count": len(checks),
    "contract_evidence": contract_evidence,
    "contract": {
        "sdk_preflight_name": BC.SDK_PREFLIGHT.name,
        "native_helper_name": BC.NATIVE_HELPER.name,
        "sdk_python": BC.PYTHON,
        "elf_consumer": "qwen35-vl-consumer",
    },
    "non_claims": ["no Docker/GPU/Torch/network/compile", "no V4/controller/helper/runtime/model/binary edits"],
}
out = ROOT / "evidence/controller-v4-stage-argv-contract-receipt.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"status": receipt["status"], "checks": len(checks),
                  "test": str(Path(__file__)), "receipt": str(out),
                  "receipt_sha256": sha(out), "v4_sha256": sha(V4)}))
if failed:
    raise SystemExit(1)
