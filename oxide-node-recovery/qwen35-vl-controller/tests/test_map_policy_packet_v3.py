#!/usr/bin/env python3
"""Real CPU behaviour proof for the map/policy successor controller.

No Docker/GPU/Torch/network/compiler is used. The tests call the same functions
that the Root phases use: host model validation, explicit staging argv builders,
remote environment, comparator, SDK preflight model validation, and the Native
production-policy helper. Docker/subprocess are mocked only at their boundary.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
CONTROLLER = ROOT / "qwen35_vl_map_policy_controller_v3.py"
PREFLIGHT = ROOT / "native/sdk/sdk_map_preflight_v3.py"
NATIVE = ROOT / "native/native_whole_model_policy.py"
FIXTURES = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen3-vl-v3/native-baseline/fixtures/fixtures.json")
MODELS = ROOT / "device-packet/model-locations-v1.json"
ASSETS = ROOT / "device-packet/native-assets-v1.json"
GATE = ROOT / "root-v1/ROOT-GATE-MAP-POLICY-V3.json"
PAIRED = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO/release/qwen35-vl-consumer")
LIFE = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1/target-opt3-noLTO-harness/release/qwen35-vl-consumer")
checks = []


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BC = load("map_policy_controller", CONTROLLER)
PRE = load("sdk_map_preflight", PREFLIGHT)
NATIVE_MOD = load("native_policy", NATIVE)


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


def model_args(names=None):
    return SimpleNamespace(model_name=names, consumer=str(PAIRED), life_consumer=str(LIFE))


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


@one("pins_real_models_assets_runtime_source_and_native_policy")
def _():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        report = BC.phase_pins(model_args(["Qwen3.5-0.8B"]))
    assert report["models"]["Qwen3.5-0.8B"]["container_path"].startswith("/bench/michaelfeil/")
    assert report["source"]["unchanged"] is True
    assert len(report["source"]["eight_rust_pins"]) == 8
    assert report["runtime"]["fa4"]["file_count"] == 213
    assert report["runtime"]["gdn_provider"]["file_count"] == 74
    assert report["native_policy"]["policy"] == BC.EXPECTED_NATIVE_POLICY
    assert report["oxide_cuda_graphs"] is False


@one("plan_all_five_models_uses_container_paths_only_inside")
def _():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        plan = BC.phase_plan(model_args(None))
    expected = {"Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B",
                "Elephant-Qwen3.5-VL", "Mammoth-Qwen3.5-VL"}
    assert set(plan["models"]) == expected
    for name, row in plan["models"].items():
        assert row["host_path"].startswith("/workspace/model-performance/michaelfeil/")
        assert row["container_path"].startswith("/bench/michaelfeil/")
    assert len(plan["native_policy"]) == 7
    assert plan["native_policy"]["_cuda_graph_max_tokens"] == "4096"
    assert plan["oxide_cuda_graphs"] is False


@one("remote_env_roles_and_no_oxide_graph_env")
def _():
    env = BC.remote_env("/tmp/oxide-qwen35vl-map-policy/test", 0)
    pairs = dict(env[index + 1].split("=", 1)
                 for index, value in enumerate(env) if value == "-e" and index + 1 < len(env))
    expected = {
        "INFINITY_OXIDE_KERNEL_MODULE": "/tmp/oxide-qwen35vl-map-policy/test/runtime/kernels",
        "INFINITY_FA4_ROOT": "/tmp/oxide-qwen35vl-map-policy/test/runtime/fa4",
        "INFINITY_OXIDE_GDN_RAW_KERNELS": "/tmp/oxide-qwen35vl-map-policy/test/runtime/gdn-raw",
        "INFINITY_OXIDE_GDN_PROVIDER": "/tmp/oxide-qwen35vl-map-policy/test/runtime/gdn-provider",
        "INFINITY_OXIDE_GDN_AUX_KERNELS": "/tmp/oxide-qwen35vl-map-policy/test/runtime/gdn-aux",
    }
    for key, value in expected.items():
        assert pairs[key] == value
    assert pairs["CUDA_VISIBLE_DEVICES"] == "0"
    assert not [key for key in pairs if "GRAPH" in key.upper()]


@one("comparator_real_fixture_pixels")
def _():
    data = json.loads(FIXTURES.read_text())
    item = next(x for x in data["fixtures"] if x["name"] == "single-cats")
    pixel = Path(item["images"][0]["pixels"])
    values = list(struct.unpack("<8f", pixel.read_bytes()[:32]))
    with tempfile.TemporaryDirectory() as tmp:
        native = Path(tmp) / "native.json"; oxide = Path(tmp) / "oxide.json"; out = Path(tmp) / "pair.json"
        native.write_text(json.dumps({"rows": [{"index": 0, "kind": "pooled", "values": values}]}))
        oxide.write_text(json.dumps({"rows": [{"index": 0, "kind": "pooled", "values": values}]}))
        BC.compare_whole(native, oxide, out)
        assert json.loads(out.read_text())["status"] == "passed"


@one("comparator_rejects_invalid_fields")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def write(name, rows):
            path = tmp / name; path.write_text(json.dumps({"rows": rows})); return path
        native = write("n.json", [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]},
                                  {"index": 1, "kind": "pooled", "values": [3.0]}])
        cases = {
            "missing": [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]}],
            "extra": [{"index": 0, "kind": "pooled", "values": [1.0, 2.0]},
                      {"index": 1, "kind": "pooled", "values": [3.0]},
                      {"index": 2, "kind": "pooled", "values": [4.0]}],
            "empty": [{"index": 0, "kind": "pooled", "values": []},
                      {"index": 1, "kind": "pooled", "values": [3.0]}],
            "nan": [{"index": 0, "kind": "pooled", "values": [float("nan"), 2.0]},
                    {"index": 1, "kind": "pooled", "values": [3.0]}],
            "empty_logits": [{"index": 0, "kind": "decision", "logits": []},
                             {"index": 1, "kind": "decision", "logits": [1.0]}],
        }
        for name, rows in cases.items():
            oxide = write(name + ".json", rows)
            try:
                BC.compare_whole(native, oxide, tmp / (name + "-out.json"))
            except RuntimeError:
                continue
            raise AssertionError(f"comparator accepted {name}")


@one("checked_cp_actual_argv_and_failure")
def _():
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=(1 if len(calls) == 1 else 0), stderr="mock copy failed")
    with mock.patch.object(BC, "docker", fake):
        try:
            BC.checked_cp("host-file", "remote-file")
            raise AssertionError("checked_cp accepted rc=1")
        except RuntimeError:
            pass
        BC.checked_cp("host-file", "remote-file")
    assert calls[0] == ("cp", "host-file", f"{BC.CONTAINER}:remote-file")


@one("run_container_actual_argv_and_failure")
def _():
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=2, stdout="out", stderr="failure")
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.log"
        with mock.patch.object(BC, "docker", fake):
            try:
                BC.run_container("/tmp/remote", 0, ["bin", "arg"], log)
                raise AssertionError("run_container accepted rc=2")
            except RuntimeError:
                pass
        assert log.exists() and "failure" in log.read_text()
    assert calls[0][0] == "exec" and BC.CONTAINER in calls[0] and "taskset" in calls[0]


@one("retrieve_actual_argv_and_failure")
def _():
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stderr="mock retrieval failed")
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(BC, "docker", fake):
            try:
                BC.retrieve("remote/out", Path(tmp) / "out")
                raise AssertionError("retrieve accepted rc=1")
            except RuntimeError:
                pass
    assert calls[0] == ("cp", f"{BC.CONTAINER}:remote/out", str(Path(tmp) / "out"))


@one("stage_uses_named_assets_and_remote_media_paths")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        remote = "/tmp/oxide-qwen35vl-map-policy/unit"
        cases = BC.build_cases(Path(tmp) / "cases", remote)
        assert [case["name"] for case in cases] == ["single-cats", "two-images", "three-frame-cats"]
        for case in cases:
            text = case["manifest"].read_text()
            assert "/tmp/oxide-all-model-feature-matrix-v1" not in text
            for fixture in json.loads(text)["fixtures"]:
                assert fixture["pixels"].startswith(remote + "/fixtures/")
                assert Path(fixture["pixels"]).name in json.loads(ASSETS.read_text())["assets"]


@one("case_filter_light_retry_uses_named_single_case")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        remote = "/tmp/oxide-qwen35vl-map-policy/unit-light"
        cases = BC.build_cases(Path(tmp) / "cases", remote, ["single-cats"])
        assert [case["name"] for case in cases] == ["single-cats"]
        media = json.loads(cases[0]["manifest"].read_text())
        assert len(media["fixtures"]) == 1
        assert media["fixtures"][0]["pixels"] == remote + "/fixtures/single-cats.image0.f32"
        try:
            BC.build_cases(Path(tmp) / "bad", remote, ["unknown-case"])
            raise AssertionError("unknown case accepted")
        except RuntimeError as error:
            assert "unknown/disabled case" in str(error)


@one("sdk_preflight_argv_uses_typed_container_path_and_expected_hash")
def _():
    models = BC.validate_model_records(["Qwen3.5-0.8B"])
    model = models["Qwen3.5-0.8B"]
    argv = BC.sdk_preflight_argv("/tmp/remote", "Qwen3.5-0.8B", model,
                                 "/tmp/remote/cases/single-cats.media.json",
                                 "/tmp/remote/preflight")
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert pairs["--model-container"] == model["container_path"]
    assert pairs["--model-container"] != model["host_path"]
    assert pairs["--model-container"].startswith("/bench/michaelfeil/")
    assert pairs["--expected-config-sha256"] == model["config_sha256"]
    assert pairs["--expected-index-sha256"] == model["index_sha256"]
    assert pairs["--host-model-path"] == model["host_path"]
    assert pairs["--union124"] == "/tmp/remote/runtime/kernels"



@one("native_create_handle_device_cstring_is_cuda_index0")
def _():
    calls = []
    creator = __import__("ctypes").CFUNCTYPE(
        __import__("ctypes").c_void_p,
        __import__("ctypes").POINTER(NATIVE_MOD.TeiConfig),
        __import__("ctypes").c_char_p,
        __import__("ctypes").c_int32)
    @creator
    def fake_create(cfg_pointer, device_cstring, dtype):
        calls.append((bytes(device_cstring), int(dtype)))
        return 1
    class FakeLib:
        pass
    fake = FakeLib()
    fake.tei_create = fake_create
    fake.tei_error = lambda: b""
    handle = NATIVE_MOD.create_handle(fake, NATIVE_MOD.TeiConfig())
    assert handle == 1
    assert calls == [(b"cuda:0", NATIVE_MOD.BF16)]


@one("native_weight_file_routing_actual_single_and_index_and_missing_reject")
def _():
    records = json.loads(MODELS.read_text())["checkpoints"]
    for name in ("Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B"):
        record = records[name]
        files = NATIVE_MOD.select_weight_files(
            record["host_path"], record["config_sha256"], record["index_sha256"])
        index = json.loads((Path(record["host_path"]) /
                            "model.safetensors.index.json").read_text())
        assert [path.name for path in files] == sorted(set(index["weight_map"].values()))
    elephant = records["Elephant-Qwen3.5-VL"]
    files = NATIVE_MOD.select_weight_files(
        elephant["host_path"], elephant["config_sha256"], None)
    assert files == [Path(elephant["host_path"]) / "model.safetensors"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = tmp / "config.json"
        config.write_text('{"probe": true}\n')
        expected = __import__("hashlib").sha256(config.read_bytes()).hexdigest()
        try:
            NATIVE_MOD.select_weight_files(tmp, expected, None)
            raise AssertionError("missing weight file was accepted")
        except SystemExit as error:
            assert "no model.safetensors.index.json" in str(error)
    try:
        NATIVE_MOD.select_weight_files(
            records["Qwen3.5-0.8B"]["host_path"],
            records["Qwen3.5-0.8B"]["config_sha256"], "0" * 64)
        raise AssertionError("wrong index SHA was accepted")
    except SystemExit as error:
        assert "index SHA" in str(error)


@one("native_argv_explicit_policy_and_oxide_argv_no_graphs")
def _():
    models = BC.validate_model_records(["Qwen3.5-0.8B", "Mammoth-Qwen3.5-VL"])
    for name in models:
        model = models[name]
        native = BC.native_whole_argv("/tmp/remote", name, model,
                                      "/tmp/remote/cases/x.media.json", "/tmp/remote/native")
        assert "--production-cuda-graphs" in native
        assert "--cuda-graphs" not in native
        assert native[native.index("--model") + 1] == model["container_path"]
        assert native[native.index("--expected-config-sha256") + 1] == model["config_sha256"]
        if model["index_sha256"]:
            assert native[native.index("--expected-index-sha256") + 1] == model["index_sha256"]
        plan = BC.native_plan_argv("/tmp/remote", model,
                                   "/tmp/remote/cases/x.media.json", "/tmp/remote/plan")
        assert plan[plan.index("--expected-config-sha256") + 1] == model["config_sha256"]
        for argv in (BC.oxide_whole_argv("/tmp/remote", model, "/tmp/remote/cases/x.media.json",
                                         "/tmp/remote/oxide.json"),
                     BC.oxide_life_argv("/tmp/remote", model, "/tmp/remote/cases/x.media.json",
                                        "/tmp/remote/life.json")):
            assert not [item for item in argv if "graph" in item.lower()]
            assert argv[0].endswith("/bin/qwen35-vl-consumer")
            assert argv[0] != BC.PYTHON
            assert argv[argv.index("--model") + 1] == model["container_path"]


@one("native_policy_exact_seven_keys_and_untouched_config")
def _():
    assert NATIVE_MOD.PRODUCTION_CUDA_GRAPH_POLICY == BC.EXPECTED_NATIVE_POLICY
    config = json.loads((Path("/workspace/model-performance/michaelfeil/onejev-perf/"
                              "backbone-comparison-20261023/models/Qwen3.5-0.8B/config.json")).read_text())
    before = copy.deepcopy(config)
    base = {}
    NATIVE_MOD.flatten(config, "", base)
    options = NATIVE_MOD.production_options(config, "mean")
    assert config == before
    for key, value in base.items():
        assert options[key] == value
    assert options["_pooling"] == "mean"
    assert {key: options[key] for key in BC.EXPECTED_NATIVE_POLICY} == BC.EXPECTED_NATIVE_POLICY
    assert len(BC.EXPECTED_NATIVE_POLICY) == 7


@one("native_apply_options_passes_all_seven_policy_keys")
def _():
    class FakeLib:
        def __init__(self):
            self.calls = []
        def tei_option(self, handle, key, value):
            self.calls.append((key.decode(), value.decode()))
            return 0
        def tei_error(self):
            return b""
    config = json.loads((Path("/workspace/model-performance/michaelfeil/onejev-perf/"
                              "backbone-comparison-20261023/models/Qwen3.5-0.8B/config.json")).read_text())
    options = NATIVE_MOD.production_options(config, "mean")
    fake = FakeLib()
    NATIVE_MOD.apply_options(fake, object(), options)
    actual = {key: value for key, value in fake.calls}
    for key, value in BC.EXPECTED_NATIVE_POLICY.items():
        assert actual[key] == value
    assert actual["_pooling"] == "mean"


@one("native_plan_only_real_config_emits_policy")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases = BC.build_cases(tmp / "cases", "/tmp/oxide-qwen35vl-map-policy/unit")
        out = tmp / "out"
        done = subprocess.run([sys.executable, str(NATIVE), "--plan-only",
                               "--library", "/unused", "--model",
                               "/workspace/model-performance/michaelfeil/onejev-perf/"
                               "backbone-comparison-20261023/models/Qwen3.5-0.8B",
                               "--expected-config-sha256",
                               "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
                               "--expected-index-sha256",
                               "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
                               "--media", str(cases[0]["manifest"]),
                               "--out", str(out)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        plan = json.loads((out / "native-whole-plan.json").read_text())
        assert plan["native_cuda_graph_policy"] == BC.EXPECTED_NATIVE_POLICY
        assert plan["tokens"] > 0


@one("native_gpu_path_requires_explicit_production_flag")
def _():
    done = subprocess.run([sys.executable, str(NATIVE), "--allow-gpu",
                           "--library", "/unused", "--model",
                           "/workspace/model-performance/michaelfeil/onejev-perf/"
                           "backbone-comparison-20261023/models/Qwen3.5-0.8B",
                           "--expected-config-sha256",
                           "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
                           "--expected-index-sha256",
                           "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
                           "--media", "/unused", "--out", "/unused"],
                          capture_output=True, text=True)
    assert done.returncode != 0
    assert "--production-cuda-graphs" in done.stderr


@one("sdk_preflight_rejects_host_path_before_torch")
def _():
    record = json.loads(MODELS.read_text())["checkpoints"]["Qwen3.5-0.8B"]
    args = SimpleNamespace(model_container=record["host_path"],
                           host_model_path=record["host_path"],
                           model_name="Qwen3.5-0.8B", model_format="index",
                           expected_config_sha256=record["config_sha256"],
                           expected_index_sha256=record["index_sha256"])
    result = PRE.validate_model(args)
    assert any("must be a /bench/michaelfeil" in item for item in result["failures"])
    assert any("host model path" in item for item in result["failures"])


@one("sdk_preflight_actual_host_config_index_hashes_all_five")
def _():
    old = PRE.CONTAINER_MODEL_PREFIX
    PRE.CONTAINER_MODEL_PREFIX = "/workspace/model-performance/michaelfeil"
    try:
        expected = json.loads((ROOT / "device-packet/checkpoints.json").read_text())["checkpoints"]
        for name, record in json.loads(MODELS.read_text())["checkpoints"].items():
            args = SimpleNamespace(model_container=record["host_path"],
                                   host_model_path="/host/provenance/" + name,
                                   model_name=name, model_format=record["format"],
                                   expected_config_sha256=record["config_sha256"],
                                   expected_index_sha256=record["index_sha256"])
            result = PRE.validate_model(args)
            assert result["failures"] == []
            assert result["model"]["config_sha256"] == expected[name]["config_sha256"]
            assert result["model"]["index_sha256"] == expected[name]["index_sha256"]
            if record["format"] == "index":
                assert result["model"]["tensor_count"] == expected[name]["tensor_count"]
                assert result["model"]["vision_tensor_count"] == expected[name]["vision_tensor_count"]
            else:
                assert result["model"]["tensor_count"] is None
    finally:
        PRE.CONTAINER_MODEL_PREFIX = old


@one("source_eight_rust_production_pins_unchanged")
def _():
    proof = BC.source_pin_proof()
    assert proof["unchanged"] is True
    assert len(proof["eight_rust_pins"]) == 8
    for rel, row in proof["eight_rust_pins"].items():
        assert row["source_sha256"] == row["expected"]
        assert row["owned_sha256"] == row["expected"]


@one("existing_binaries_3780_and_21c_unchanged")
def _():
    proof = BC.binary_proof(PAIRED, LIFE)
    assert proof["paired"]["actual"] == BC.PAIRED_SHA == "3780facf61529822f33a4cf1416bc6c8723acc1fa2cd2c778f7bc54f2140d83b"
    assert proof["owning_life"]["actual"] == BC.LIFE_SHA == "21c5603ee6e9eff06dae5a4344d83fba43ee8bd0ed9a31d462417970c0bf297a"


@one("native_policy_helper_pin_and_all_seven_keys")
def _():
    proof = BC.native_policy_proof()
    assert proof["policy"] == BC.EXPECTED_NATIVE_POLICY
    assert proof["sha256"] == __import__("hashlib").sha256(NATIVE.read_bytes()).hexdigest()
    assert proof["oxide_cuda_graphs"] is False


@one("v1_and_v2_preserved_snapshot_checks")
def _():
    assert __import__("hashlib").sha256((ROOT / "qwen35_vl_map_policy_controller.py").read_bytes()).hexdigest() == \
        "e8b13196ec9de9ed4b0840cc8ece42d02a3c808802003e326108319128d51e3f"
    assert __import__("hashlib").sha256((ROOT / "qwen35_vl_map_policy_controller_v2.py").read_bytes()).hexdigest() == \
        "bef456a206b82900ebf0d70cdb83993ca15e00526a5e9357a49552599f2f574c"
    assert __import__("hashlib").sha256((ROOT / "READY.json").read_bytes()).hexdigest() == \
        "6cf117fe82fee0a5d434a2d6025dbf87526a750de88bafa0efbb93c86f0d0995"
    assert __import__("hashlib").sha256((ROOT / "READY-v2.json").read_bytes()).hexdigest() == \
        "26f12d046a3e02fe2d2be227762da77e6246dcb0a2731b8a1e9bc19a4462f6ad"
    failed = ROOT / "evidence/failed-cpu-0.8B-single-v1"
    report = json.loads((failed / "report.json").read_text())
    assert report["run_id"] == "cpu-1793638200711671722"
    assert report["status"] == "failed"
    assert "rc=126" in report["error"]
    assert "sdk/sdk_map_preflight.py" in report["error"]
    log = (failed / "cpu-Qwen3.5-0.8B-single-cats-sdk.log").read_text()
    assert "taskset: failed to execute" in log and "Permission denied" in log


@one("helper_argv_nonexecutable_script_via_sys_executable_and_sdk_python_prefix")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        script = tmp / "tiny_own_script.py"
        script.write_text("import sys; print('TINY_OWN_OK'); sys.exit(0)\n")
        __import__("os").chmod(script, 0o644)
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        assert done.returncode == 0 and "TINY_OWN_OK" in done.stdout
    models = BC.validate_model_records(["Qwen3.5-0.8B"])
    model = models["Qwen3.5-0.8B"]
    builders = [
        BC.sdk_preflight_argv("/tmp/remote", "Qwen3.5-0.8B", model,
                              "/tmp/remote/cases/x.media.json", "/tmp/remote/pre"),
        BC.native_plan_argv("/tmp/remote", model,
                            "/tmp/remote/cases/x.media.json", "/tmp/remote/plan"),
        BC.native_whole_argv("/tmp/remote", "Qwen3.5-0.8B", model,
                             "/tmp/remote/cases/x.media.json", "/tmp/remote/native"),
    ]
    for argv in builders:
        assert argv[0] == "/opt/torch-sdk/bin/python3.12"
        assert argv[1].startswith("/tmp/remote/")
        assert argv[1].endswith(".py")
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(BC, "docker", fake):
            BC.run_container("/tmp/remote", "", builders[0], Path(tmp) / "run.log")
    actual = calls[0]
    assert actual[0] == "exec"
    python_index = actual.index(BC.CONTAINER) + 1
    assert actual[python_index:python_index + 5] == ("taskset", "-c", BC.CPU_LIST,
                                                     "/opt/torch-sdk/bin/python3.12",
                                                     builders[0][1])


@one("runtime_v3_device130_manifest_count_arch_files_and_freeze")
def _():
    runtime = BC.validate_runtime()
    dev = runtime["device130"]
    assert dev["manifest_sha256"] == "1d0b59dda1cef92c0e752553872e645adbdc07e93d40245817e82246f58b982d"
    assert dev["kernel_count"] == 130
    assert dev["architectures"] == ["sm_90"]
    assert dev["file_count"] == 13
    assert dev["sm_90_cubin_sha256"] == "695e72ffbcd1f7a7ac7648e2b4cf5d146516c7100820dd6f73c41181271f9d0e"
    assert __import__("hashlib").sha256((Path("/tmp/oxide-device130-compose-source-v1/freeze-compiled.json")).read_bytes()).hexdigest() == \
        "ea0fb11484f1d684f32896b608496c29bdcf266a6d0a8472f2138757f17e4024"


@one("runtime_v3_builders_pass_130_expectations_and_not_to_native_plan")
def _():
    models = BC.validate_model_records(["Qwen3.5-0.8B"])
    model = models["Qwen3.5-0.8B"]
    argv = BC.sdk_preflight_argv("/tmp/remote", "Qwen3.5-0.8B", model,
                                 "/tmp/remote/cases/x.media.json", "/tmp/remote/pre")
    pairs = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith("--")}
    assert pairs["--runtime-manifest-sha256"] == "1d0b59dda1cef92c0e752553872e645adbdc07e93d40245817e82246f58b982d"
    assert pairs["--runtime-kernel-count"] == "130"
    assert pairs["--runtime-architectures"] == "sm_90"
    assert pairs["--stage-expected"] == "/tmp/remote/stage-expected.json"
    assert argv[0] == "/opt/torch-sdk/bin/python3.12"
    plan = BC.native_plan_argv("/tmp/remote", model, "/tmp/remote/cases/x.media.json", "/tmp/remote/plan")
    assert "--runtime-manifest-sha256" not in plan
    assert plan[0] == "/opt/torch-sdk/bin/python3.12"


@one("runtime_inventory_v3_gdn_names_elfdeps_fa4_canonical26")
def _():
    inv = json.loads((ROOT / "device-packet/runtime-inventory-v3.json").read_text())
    assert set(inv["device130"]["files"]) == {
        "LICENSE-Candle-BSD3", "LICENSE-PyTorch-BSD3", "LICENSE-Qwen-VL-APACHE",
        "LICENSE-Qwen-VL-MIT", "NOTICE", "NOTICE.qwen-vl", "NOTICE.qwen35", "manifest.json",
        "sm_90/build.log", "sm_90/kernels.cubin", "sm_90/kernels.ptx",
        "sm_90/ptxas.log", "sm_90/symbols.log"}
    assert inv["gdn_raw"]["architectures"] == ["sm_90", "sm_100", "sm_120"]
    assert inv["gdn_aux"]["architectures"] == ["sm_90", "sm_100", "sm_120"]
    assert inv["gdn_provider"]["abi_version"] == 2
    assert inv["gdn_provider"]["architectures"] == ["sm_90a", "sm_100f", "sm_120f"]
    assert inv["gdn_provider"]["libraries"] == ["libcute_dsl_runtime.so", "libinfinitygdn.so", "libtvm_ffi.so"]
    assert "libtvm_ffi.so" in inv["gdn_provider"]["elf_needed"]["libinfinitygdn.so"]
    assert "libcute_dsl_runtime.so" in inv["gdn_provider"]["elf_needed"]["libinfinitygdn.so"]
    assert inv["fa4"]["host_path"] == "/tmp/oxide-fa4-canonical26-derived-v4/bundle"
    assert inv["fa4"]["manifest_sha256"] == "0e4af6e1fd40714300649dd98e6eb31a0c59e9e3cb493b377de07d57d63916a9"
    assert inv["device130"]["complete_target_set"] is False
    assert inv["device130"]["architecture_scope"] == ["sm_90"]
    assert inv["device130"]["old124_all3_compile_proof_preserved_separate"] is True
    assert len(inv["fa4"]["files"]) == 213


@one("runtime_v3_mock_stage_copies_device130_and_checks_actual_files")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cases = BC.build_cases(tmp / "cases", "/tmp/oxide-qwen35vl-map-policy-v3/unit")
        cp_calls, dir_calls = [], []
        with mock.patch.object(BC, "checked_docker", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr="")), \
             mock.patch.object(BC, "checked_cp", lambda s, t: cp_calls.append((str(s), str(t)))), \
             mock.patch.object(BC, "checked_cp_dir", lambda s, t: dir_calls.append((str(s), str(t)))), \
             mock.patch.object(BC, "verify_remote_stage", lambda *a, **k: {"status": "passed"}):
            result = BC.stage("/tmp/oxide-qwen35vl-map-policy-v3/unit", PAIRED, cases,
                              manifest_dir=tmp)
        assert result["verified_files"] >= 315
        rt = json.loads(Path("device-packet/runtime-paths-v3.json").read_text())["components"]
        assert (rt["device130"]["host_path"], "/tmp/oxide-qwen35vl-map-policy-v3/unit/runtime/kernels") in dir_calls
        assert (rt["fa4"]["host_path"], "/tmp/oxide-qwen35vl-map-policy-v3/unit/runtime/fa4") in dir_calls
        assert (rt["gdn_provider"]["host_path"], "/tmp/oxide-qwen35vl-map-policy-v3/unit/runtime/gdn-provider") in dir_calls
        manifest = json.loads((tmp / "unit-expected.json").read_text())
        cubin = manifest["files"]["runtime/kernels/sm_90/kernels.cubin"]
        assert cubin["sha256"] == "695e72ffbcd1f7a7ac7648e2b4cf5d146516c7100820dd6f73c41181271f9d0e"


@one("gate_pins_match_actual_new_files")
def _():
    gate = json.loads(GATE.read_text())
    digest = lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()
    assert gate["pins"]["controller"]["sha256"] == digest(CONTROLLER)
    assert gate["pins"]["controller_v1_preserved"]["sha256"] == digest(ROOT / "qwen35_vl_map_policy_controller.py")
    assert gate["pins"]["preflight_v3"]["sha256"] == digest(PREFLIGHT)
    assert gate["pins"]["native_device_file_routing_v3"]["sha256"] == digest(ROOT / "device-packet/native-device-file-routing-v3-runtime130.json")
    assert gate["pins"]["native_policy_helper"]["sha256"] == digest(NATIVE)
    assert gate["pins"]["models"]["sha256"] == digest(MODELS)
    assert gate["pins"]["assets"]["sha256"] == digest(ASSETS)
    assert gate["pins"]["runtime_paths_v3"]["sha256"] == digest(ROOT / "device-packet/runtime-paths-v3.json")
    assert gate["pins"]["runtime_inventory_v3"]["sha256"] == digest(ROOT / "device-packet/runtime-inventory-v3.json")
    assert gate["pins"]["model_manifest"]["sha256"] == digest(ROOT / "device-packet/model-manifest-v1.json")
    assert gate["pins"]["successor_pins"]["sha256"] == digest(ROOT / "device-packet/successor-pins-v3.json")
    pins = json.loads((ROOT / "device-packet/successor-pins-v3.json").read_text())
    assert sum(row["file_count"] for row in pins["runtime"].values()) == 315
    assert pins["runtime"]["device130"]["kernel_count"] == 130
    assert pins["runtime"]["device130"]["architectures"] == ["sm_90"]
    assert pins["runtime"]["fa4"]["files"]["sm_90/manifest.json"] == "0e4af6e1fd40714300649dd98e6eb31a0c59e9e3cb493b377de07d57d63916a9"
    assert pins["native_policy"] == BC.EXPECTED_NATIVE_POLICY
    assert pins["oxide_cuda_graphs"] is False


failed = [row for row in checks if not row["passed"]]
report = {
    "schema": "oxide-qwen35vl-map-policy-cpu-proof/v3",
    "status": "passed" if not failed else "failed",
    "checks": checks,
    "check_count": len(checks),
    "controller_sha256": __import__("hashlib").sha256(CONTROLLER.read_bytes()).hexdigest(),
    "preflight_sha256": __import__("hashlib").sha256(PREFLIGHT.read_bytes()).hexdigest(),
    "native_policy_helper_sha256": __import__("hashlib").sha256(NATIVE.read_bytes()).hexdigest(),
}
out = ROOT / "evidence/bounded-map-policy-cpu-proof-v3.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"status": report["status"], "checks": len(checks), "out": str(out)}))
if failed:
    raise SystemExit(1)
