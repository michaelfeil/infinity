#!/usr/bin/env python3
"""Bounded Qwen3.5-VL SDK map + Native graph-policy controller (V3 runtime130).

OwnedControllerRoot: /tmp/oxide-qwen35vl-sdk-map-policy-v1
SourceRoot (read-only): /tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1

Only two things are changed relative to the frozen source packet:
1. explicit named host/container checkpoint records (host hashes on host,
   container paths only inside SDK argv), and
2. an explicit seven-key Native production CUDA-graph policy helper.

V2 only prefixes the three Python helper argv builders with the explicit SDK
interpreter. It does not touch ELF consumer argv and does not rely on chmod,
ambient python or shebang execution. Every docker/stage/retrieve call checks
its return code. Oxide remains on its normal no-graph path. No checkpoint
config, math, weights or thresholds change.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

OWNED_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/tmp/oxide-all-model-feature-matrix-v1/qwen35-vl-lifecycle-fix-v1")
CONTAINER = "infinity-torch214-compare-sdk"
DOCKER = ("docker", "--host=unix:///var/run/docker.sock")
PYTHON = "/opt/torch-sdk/bin/python3.12"
B66 = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
VENDOR = "/tmp/oxide-host-cublas-130200"
REMOTE_BASE = "/tmp/oxide-qwen35vl-map-policy-v4"
CPU_LIST = "88-95"
PAIRED_BINARY = SOURCE_ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"
PAIRED_SHA = "3780facf61529822f33a4cf1416bc6c8723acc1fa2cd2c778f7bc54f2140d83b"
LIFE_BINARY = SOURCE_ROOT / "target-opt3-noLTO-harness/release/qwen35-vl-consumer"
LIFE_SHA = "21c5603ee6e9eff06dae5a4344d83fba43ee8bd0ed9a31d462417970c0bf297a"
MODELS = OWNED_ROOT / "device-packet/model-locations-v1.json"
ASSETS = OWNED_ROOT / "device-packet/native-assets-v1.json"
RUNTIME = OWNED_ROOT / "device-packet/runtime-paths-v3.json"
NATIVE_HELPER = OWNED_ROOT / "native/native_whole_model_policy.py"
SDK_PREFLIGHT = OWNED_ROOT / "native/sdk/sdk_map_preflight_v3.py"
DOCKER_SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS",
                    "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")

RUST_PRODUCTION_PINS = {
    "consumer/src/main.rs": "5d7a75d681a882be7149d4e543b9b861bcd5c674b0e416fd301a7d648f8d2a2b",
    "libs/infinity-v3/backends/oxide/src/lib.rs": "aeea32e693ea2d5120841fe4c28ae8527b8c1adcb09b008beab2815dde950721",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_device.rs": "1910837d81bdb0abc73a34c73c1879219b5baca7fe44bcb81fdb234d904ea5fb",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_engine.rs": "83fd2c8b3e233cbb6b42e2442900f809bacdc86d58b6f52932d363c54bb99b00",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_tests.rs": "c1f31d1622bdc548c512163a54802a47fe2bea8cd2f444090be32fe471ce5d0b",
    "libs/infinity-v3/backends/oxide/src/qwen35_vl_vision.rs": "b12143546b1ff7738cce854c060f33d4fb5db86a57cdfa9759c215d346a4b42e",
    "libs/infinity-v3/backends/oxide/device/src/main.rs": "c19ae20ff2ec668f1983b65f32992c7912ccca52d27b2c18d84b72503d71f6ae",
    "libs/infinity-v3/backends/oxide/device/src/qwen35vl.rs": "f7fd1a0165c15a8bc043d2c6010951449b17d2100a147e6fb305a89f034a477e",
}
EXPECTED_NATIVE_POLICY = {
    "_cuda_graphs": "true",
    "_cuda_graph_entries": "4",
    "_cuda_graph_reuse": "3",
    "_cuda_graph_max_bytes": "536870912",
    "_cuda_graph_max_captures": "8",
    "_cuda_graph_max_tokens": "4096",
    "_media_cuda_graphs": "false",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_env():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CUDA_", "INFINITY_", "LD_"))}
    for key in DOCKER_SELECTORS:
        env.pop(key, None)
    return env


def docker(*args, timeout=3600):
    return subprocess.run([*DOCKER, *map(str, args)], env=clean_env(),
                          capture_output=True, text=True, timeout=timeout,
                          check=False)


def checked_docker(*args, timeout=3600):
    done = docker(*args, timeout=timeout)
    if done.returncode:
        raise RuntimeError(
            f"docker {' '.join(map(str, args))} failed rc={done.returncode}: {done.stderr}")
    return done


def checked_cp(source, target):
    checked_docker("cp", str(source), f"{CONTAINER}:{target}")


def checked_cp_dir(source, target):
    checked_docker("cp", str(Path(source)) + "/.", f"{CONTAINER}:{target}")


def remote_env(remote, gpu):
    values = {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "INFINITY_OXIDE_KERNEL_MODULE": f"{remote}/runtime/kernels",
        "INFINITY_OXIDE_ATTENTION": "fa4",
        "INFINITY_FA4_ROOT": f"{remote}/runtime/fa4",
        "INFINITY_OXIDE_GDN_RAW_KERNELS": f"{remote}/runtime/gdn-raw",
        "INFINITY_OXIDE_GDN_PROVIDER": f"{remote}/runtime/gdn-provider",
        "INFINITY_OXIDE_GDN_AUX_KERNELS": f"{remote}/runtime/gdn-aux",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LD_BIND_NOW": "1",
        "LD_PRELOAD": f"{VENDOR}/libcublas.so.13:{VENDOR}/libcublasLt.so.13",
        "LD_LIBRARY_PATH": (f"{VENDOR}:/tmp/native214-fa4-mode20/install/lib:"
                            "/opt/torch-sdk/lib/python3.12/site-packages/torch/lib:"
                            "/usr/local/cuda/lib64"),
    }
    out = []
    for key, value in values.items():
        out += ["-e", f"{key}={value}"]
    return out


def run_container(remote, gpu, argv, log_path, timeout=7200):
    command = ["exec", *remote_env(remote, gpu), CONTAINER,
               "taskset", "-c", CPU_LIST, *map(str, argv)]
    done = docker(*command, timeout=timeout)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f"container command failed rc={done.returncode}: {argv}")
    return done


def retrieve(container_path, host_path):
    Path(host_path).parent.mkdir(parents=True, exist_ok=True)
    checked_docker("cp", f"{CONTAINER}:{container_path}", str(host_path))


def tree_files(path):
    return sorted(p for p in Path(path).rglob("*") if p.is_file())


def tree_manifest(path):
    root = Path(path)
    return {str(p.relative_to(root)): sha(p) for p in tree_files(root)}


def model_records():
    data = load_json(MODELS)
    return data["checkpoints"]


def selected_models(names):
    records = model_records()
    if not names:
        return list(records)
    missing = [name for name in names if name not in records]
    if missing:
        raise RuntimeError(f"unknown model name(s): {missing}")
    return names


def validate_model_records(names=None):
    """Host-side config/index validation. Never passes host paths inside."""
    data = load_json(MODELS)
    host_prefix = data["host_prefix"]
    container_prefix = data["container_prefix"]
    report = {}
    failures = []
    for name in selected_models(names):
        record = data["checkpoints"][name]
        host = Path(record["host_path"])
        container = record["container_path"]
        expected_container = str(host).replace(host_prefix, container_prefix, 1)
        if container != expected_container:
            failures.append(f"{name}: container_path is not the explicit host-prefix mapping")
        if not container.startswith(container_prefix + "/"):
            failures.append(f"{name}: container_path is not under {container_prefix}")
        if not host.is_dir():
            failures.append(f"{name}: host path missing {host}")
            continue
        config = host / record["config_rel"]
        if not config.is_file():
            failures.append(f"{name}: host config missing")
            continue
        config_sha = sha(config)
        if config_sha != record["config_sha256"]:
            failures.append(f"{name}: host config sha {config_sha} != expected")
        index_sha = None
        if record["format"] == "index":
            index = host / record["index_rel"]
            if not index.is_file():
                failures.append(f"{name}: host index missing")
            else:
                index_sha = sha(index)
                if index_sha != record["index_sha256"]:
                    failures.append(f"{name}: host index sha {index_sha} != expected")
        elif record["format"] != "single":
            failures.append(f"{name}: unsupported format {record['format']}")
        weights = host / record["weights_rel"]
        if not weights.is_file():
            failures.append(f"{name}: host weights entry missing {record['weights_rel']}")
        report[name] = {
            "host_path": str(host),
            "container_path": container,
            "format": record["format"],
            "config_sha256": config_sha,
            "index_sha256": index_sha,
            "weights_rel": record["weights_rel"],
            "container_path_under": container_prefix,
            "host_only_validation": True,
        }
    if failures:
        raise RuntimeError("model location validation failed: " + "; ".join(failures))
    return report


def validate_assets():
    data = load_json(ASSETS)
    fixture = data["fixture_set"]
    if sha(fixture["host_path"]) != fixture["sha256"]:
        raise RuntimeError("fixture set SHA changed")
    report = {}
    for key, asset in data["assets"].items():
        if sha(asset["host_path"]) != asset["sha256"]:
            raise RuntimeError(f"fixture asset SHA changed: {key}")
        report[key] = {"host_path": asset["host_path"],
                       "remote_name": asset["remote_name"],
                       "sha256": asset["sha256"]}
    return report


def validate_runtime():
    data = load_json(RUNTIME)
    report = {}
    for name, component in data["components"].items():
        root = Path(component["host_path"])
        if not root.is_dir():
            raise RuntimeError(f"runtime component missing: {name} {root}")
        manifest = root / component["manifest_rel"]
        if sha(manifest) != component["manifest_sha256"]:
            raise RuntimeError(f"runtime manifest SHA changed: {name}")
        count = len(tree_files(root))
        if count != component["file_count"]:
            raise RuntimeError(f"runtime file count changed: {name} {count}")
        if "freeze_compiled" in component:
            freeze = component["freeze_compiled"]
            if sha(freeze["path"]) != freeze["sha256"]:
                raise RuntimeError(f"runtime freeze SHA changed: {name}")
        if "kernel_count" in component:
            manifest_doc = json.loads(manifest.read_text())
            actual_kernels = manifest_doc.get("kernels")
            if not isinstance(actual_kernels, list) or len(actual_kernels) != component["kernel_count"]:
                raise RuntimeError(f"runtime kernel count changed: {name}")
        if "architectures" in component:
            manifest_doc = json.loads(manifest.read_text())
            actual_archs = [entry["architecture"] for entry in manifest_doc.get("architectures", [])]
            if actual_archs != component["architectures"]:
                raise RuntimeError(f"runtime architectures changed: {name} {actual_archs}")
        if "sm_90_cubin_sha256" in component:
            cubin = root / "sm_90/kernels.cubin"
            if sha(cubin) != component["sm_90_cubin_sha256"]:
                raise RuntimeError(f"runtime sm_90 cubin SHA changed: {name}")
        report[name] = {"host_path": str(root),
                        "remote_rel": component["remote_rel"],
                        "manifest_sha256": component["manifest_sha256"],
                        "file_count": count}
        for key in ("kernel_count", "architectures", "requested_architectures",
                    "complete_target_set", "sm_90_cubin_sha256"):
            if key in component:
                report[name][key] = component[key]
    return report


def source_pin_proof():
    """Prove the frozen source, the copied source, and the eight Rust pins match."""
    pins = load_json(SOURCE_ROOT / "device-packet/source-pins.json")["files"]
    checked = []
    mismatches = []
    for rel, expected in pins.items():
        source = SOURCE_ROOT / rel
        owned = OWNED_ROOT / rel
        if not source.is_file():
            mismatches.append(f"source root missing {rel}")
            continue
        if sha(source) != expected:
            mismatches.append(f"source root changed {rel}")
        if owned.is_file() and sha(owned) != expected:
            mismatches.append(f"owned copy changed {rel}")
        checked.append(rel)
    rust = {}
    for rel, expected in RUST_PRODUCTION_PINS.items():
        source = SOURCE_ROOT / rel
        owned = OWNED_ROOT / rel
        source_sha = sha(source)
        owned_sha = sha(owned) if owned.is_file() else None
        rust[rel] = {"expected": expected, "source_sha256": source_sha,
                     "owned_sha256": owned_sha}
        if source_sha != expected or owned_sha != expected:
            mismatches.append(f"Rust production pin changed: {rel}")
    if mismatches:
        raise RuntimeError("source pin proof failed: " + "; ".join(mismatches))
    return {"checked_files": len(checked), "eight_rust_pins": rust,
            "source_root": str(SOURCE_ROOT), "owned_root": str(OWNED_ROOT),
            "unchanged": True}


def binary_proof(consumer, life_consumer):
    paired = {"path": str(PAIRED_BINARY), "expected": PAIRED_SHA, "actual": sha(PAIRED_BINARY)}
    life = {"path": str(LIFE_BINARY), "expected": LIFE_SHA, "actual": sha(LIFE_BINARY)}
    if paired["actual"] != PAIRED_SHA:
        raise RuntimeError("paired binary SHA changed")
    if life["actual"] != LIFE_SHA:
        raise RuntimeError("owning-life binary SHA changed")
    selected = {"paired": {"path": str(consumer), "sha256": sha(consumer)},
                "owning_life": {"path": str(life_consumer), "sha256": sha(life_consumer)}}
    if Path(consumer).resolve() == PAIRED_BINARY.resolve() and selected["paired"]["sha256"] != PAIRED_SHA:
        raise RuntimeError("selected paired binary is not the pinned 3780 binary")
    if Path(life_consumer).resolve() == LIFE_BINARY.resolve() and selected["owning_life"]["sha256"] != LIFE_SHA:
        raise RuntimeError("selected owning-life binary is not the pinned 21c binary")
    return {"paired": paired, "owning_life": life, "selected": selected}


def native_policy_proof():
    module = load_module("native_whole_model_policy", NATIVE_HELPER)
    actual = module.PRODUCTION_CUDA_GRAPH_POLICY
    if actual != EXPECTED_NATIVE_POLICY:
        raise RuntimeError(f"Native production policy changed: {actual}")
    return {"helper": str(NATIVE_HELPER), "sha256": sha(NATIVE_HELPER),
            "policy": actual, "oxide_cuda_graphs": False}


def build_cases(directory, remote, selected_names=None):
    directory.mkdir(parents=True, exist_ok=True)
    assets = load_json(ASSETS)
    names = list(selected_names or assets["selected_cases"])
    missing = [name for name in names if name not in assets["selected_cases"]]
    if missing:
        raise RuntimeError(f"unknown/disabled case name(s): {missing}")
    fixtures = load_json(assets["fixture_set"]["host_path"])
    selected = {row["name"]: row for row in fixtures["fixtures"]
                if row["name"] in names}
    result = []
    for name in names:
        item = selected[name]
        images = []
        host_pixels = []
        for image in item["images"]:
            key = Path(image["pixels"]).name
            asset = assets["assets"][key]
            if sha(asset["host_path"]) != asset["sha256"]:
                raise RuntimeError(f"fixture SHA changed: {key}")
            remote_pixel = f"{remote}/fixtures/{asset['remote_name']}"
            images.append({"name": image["name"], "pixels": remote_pixel,
                           "rows": image["rows"], "patch_dim": image["patch_dim"],
                           "grid": image["grid_thw"], "merge": image["merge_size"],
                           "token_start": image["token_start"],
                           "token_count": image["token_count"]})
            host_pixels.append(asset["host_path"])
        kind = "video" if images and images[0]["grid"][0] > 1 else "image"
        media = {"text": {"input_ids": item["ids"], "token_type_ids": item["types"],
                          "position_ids": item["positions"],
                          "cumulative_seq_lengths": item["cumulative"],
                          "pooled_indices": [0], "raw_indices": [0]},
                 "fixtures": images}
        path = directory / f"{name}.media.json"
        path.write_text(json.dumps(media, indent=2) + "\n")
        result.append({"name": name, "kind": kind, "media": media,
                       "manifest": path, "host_pixels": host_pixels})
    return result


def stage(remote, consumer, cases, manifest_dir=None):
    if not (remote.startswith(REMOTE_BASE + "/") and remote != REMOTE_BASE):
        raise RuntimeError("remote stage must be a fresh run directory under the owned base")
    code = f"from pathlib import Path;Path({remote!r}).mkdir(parents=True)"
    checked_docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON, "-c", code)
    dirs = ["bin", "media", "fixtures", "cases", "native", "sdk",
            "runtime/kernels", "runtime/fa4", "runtime/gdn-raw",
            "runtime/gdn-provider", "runtime/gdn-aux"]
    code = ("from pathlib import Path;"
            f"[Path({remote!r}+'/'+p).mkdir(parents=True, exist_ok=True) for p in {dirs!r}]")
    checked_docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON, "-c", code)

    expected = {}

    def add(source, target):
        source = Path(source)
        expected[target] = {"path": f"{remote}/{target}", "sha256": sha(source)}

    add(consumer, "bin/qwen35-vl-consumer")
    checked_cp(consumer, f"{remote}/bin/qwen35-vl-consumer")
    for source, target in [(NATIVE_HELPER, "native/native_whole_model_policy.py"),
                           (SDK_PREFLIGHT, f"sdk/{SDK_PREFLIGHT.name}"),
                           (OWNED_ROOT / "native/sdk/remote_guard.py", "remote_guard.py")]:
        add(source, target)
        checked_cp(source, f"{remote}/{target}")

    assets = load_json(ASSETS)
    for key, asset in assets["assets"].items():
        if sha(asset["host_path"]) != asset["sha256"]:
            raise RuntimeError(f"fixture SHA changed before staging: {key}")
        target = f"fixtures/{asset['remote_name']}"
        add(asset["host_path"], target)
        checked_cp(asset["host_path"], f"{remote}/{target}")

    for case in cases:
        add(case["manifest"], f"cases/{case['manifest'].name}")
        checked_cp(case["manifest"], f"{remote}/cases/{case['manifest'].name}")

    trees = {}
    runtime = load_json(RUNTIME)
    for name, component in runtime["components"].items():
        source = component["host_path"]
        if sha(Path(source) / component["manifest_rel"]) != component["manifest_sha256"]:
            raise RuntimeError(f"runtime manifest SHA changed before staging: {name}")
        checked_cp_dir(source, f"{remote}/{component['remote_rel']}")
        tree = tree_manifest(source)
        if len(tree) != component["file_count"]:
            raise RuntimeError(f"runtime tree changed before staging: {name}")
        trees[f"{remote}/{component['remote_rel']}"] = sorted(tree)
        for rel, digest in tree.items():
            expected[f"{component['remote_rel']}/{rel}"] = {
                "path": f"{remote}/{component['remote_rel']}/{rel}", "sha256": digest}

    manifest_path = Path(remote).name + "-expected.json"
    verify_doc = {"files": expected, "trees": trees}
    local_doc = Path(manifest_dir or (OWNED_ROOT / "evidence")) / manifest_path
    local_doc.parent.mkdir(parents=True, exist_ok=True)
    local_doc.write_text(json.dumps(verify_doc, indent=2) + "\n")
    checked_cp(local_doc, f"{remote}/stage-expected.json")
    verify_remote_stage(remote, f"{remote}/stage-expected.json")
    return {"verified_files": len(expected),
            "trees": {k: len(v) for k, v in trees.items()},
            "manifest": str(local_doc), "manifest_sha256": sha(local_doc)}


def verify_remote_stage(remote, manifest):
    script = (
        "import hashlib,json,sys\n"
        "from pathlib import Path\n"
        "doc=json.loads(Path(sys.argv[1]).read_text())\n"
        "bad=[]\n"
        "for rel,item in doc['files'].items():\n"
        " p=Path(item['path'])\n"
        " if not p.is_file(): bad.append('missing '+rel); continue\n"
        " if hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: bad.append('sha '+rel)\n"
        "for root,names in doc['trees'].items():\n"
        " actual=sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())\n"
        " if actual!=names: bad.append('tree '+root)\n"
        "print(json.dumps({'status':'passed' if not bad else 'failed','remote':sys.argv[1],"
        "'files':len(doc['files']),'failures':bad}))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    done = docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON,
                  "-c", script, manifest)
    if done.returncode:
        raise RuntimeError(f"remote stage verification failed: {done.stdout} {done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def sdk_preflight_argv(remote, name, model, remote_media, out, runtime_component=None):
    if runtime_component is None:
        runtime_component = load_json(RUNTIME)["components"]["device130"]
    argv = [PYTHON, f"{remote}/sdk/{SDK_PREFLIGHT.name}",
            "--library", B66, "--torch-python", PYTHON,
            "--vendor", VENDOR,
            "--union124", f"{remote}/runtime/kernels",
            "--model-container", model["container_path"],
            "--host-model-path", model["host_path"],
            "--model-name", name,
            "--model-format", model["format"],
            "--expected-config-sha256", model["config_sha256"],
            "--runtime-manifest-sha256", runtime_component["manifest_sha256"],
            "--runtime-kernel-count", str(runtime_component["kernel_count"]),
            "--runtime-architectures", ",".join(runtime_component["architectures"]),
            "--stage-expected", f"{remote}/stage-expected.json",
            "--media", remote_media, "--out", out]
    if model["index_sha256"]:
        argv += ["--expected-index-sha256", model["index_sha256"]]
    return argv


def native_plan_argv(remote, model, remote_media, out):
    argv = [PYTHON, f"{remote}/native/native_whole_model_policy.py",
            "--plan-only", "--library", B66,
            "--model", model["container_path"],
            "--expected-config-sha256", model["config_sha256"],
            "--media", remote_media, "--out", out]
    if model["index_sha256"]:
        argv += ["--expected-index-sha256", model["index_sha256"]]
    return argv


def native_whole_argv(remote, name, model, remote_media, out):
    argv = [PYTHON, f"{remote}/native/native_whole_model_policy.py",
            "--allow-gpu", "--production-cuda-graphs", "--library", B66,
            "--model", model["container_path"],
            "--expected-config-sha256", model["config_sha256"],
            "--media", remote_media, "--out", out, "--pool", "mean"]
    if model["index_sha256"]:
        argv += ["--expected-index-sha256", model["index_sha256"]]
    if name in ("Elephant-Qwen3.5-VL", "Mammoth-Qwen3.5-VL"):
        argv.append("--decision")
    return argv


def oxide_whole_argv(remote, model, remote_media, out):
    return [f"{remote}/bin/qwen35-vl-consumer", "whole-factory",
            "--model", model["container_path"], "--media", remote_media,
            "--device", "0", "--pool", "mean", "--out", out]


def oxide_life_argv(remote, model, remote_media, out):
    return [f"{remote}/bin/qwen35-vl-consumer", "owning-lifecycle",
            "--model", model["container_path"], "--media", remote_media,
            "--device", "0", "--iterations", "4", "--out", out]


def finite(values):
    return (isinstance(values, list) and len(values) > 0 and
            all(isinstance(x, (int, float)) and float(x) == float(x) and
                abs(float(x)) != float("inf") for x in values))


def compare_whole(native_file, oxide_file, out_file):
    native = load_json(native_file)
    oxide = load_json(oxide_file)
    nrows = {(row.get("index", 0), row.get("kind")): row for row in native.get("rows", [])}
    orows = {(row.get("index", 0), row.get("kind")): row for row in oxide.get("rows", [])}
    if not nrows or set(nrows) != set(orows):
        raise RuntimeError("whole rows are empty or coverage differs")
    comparisons = []
    for key in sorted(nrows):
        a, b = nrows[key], orows[key]
        if "logits" in a or "logits" in b:
            av, bv = a.get("logits"), b.get("logits")
            if not finite(av) or not finite(bv) or len(av) != len(bv):
                raise RuntimeError(f"empty/invalid decision logits at {key}")
            maximum = max(abs(x - y) for x, y in zip(av, bv))
            passed = maximum <= 0.015
            comparisons.append({"index": key[0], "kind": key[1], "class": "decision",
                                "elements": len(av), "max_abs": maximum, "passed": passed})
        else:
            av, bv = a.get("values"), b.get("values")
            if not finite(av) or not finite(bv) or len(av) != len(bv):
                raise RuntimeError(f"empty/invalid embedding values at {key}")
            maximum = max(abs(x - y) for x, y in zip(av, bv))
            passed = maximum <= 0.015 + 0.015 * max(max(abs(x) for x in av), 1.0)
            comparisons.append({"index": key[0], "kind": key[1], "class": "embedding",
                                "elements": len(av), "max_abs": maximum, "passed": passed})
    report = {"schema": "oxide-qwen35vl-map-policy-pair/v1",
              "status": "passed" if all(x["passed"] for x in comparisons) else "failed",
              "comparisons": comparisons}
    Path(out_file).write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("paired whole comparison failed")


def phase_pins(args):
    models = validate_model_records(args.model_name)
    assets = validate_assets()
    runtime = validate_runtime()
    source = source_pin_proof()
    binaries = binary_proof(args.consumer, args.life_consumer)
    policy = native_policy_proof()
    report = {
        "schema": "oxide-qwen35vl-map-policy-pins/v1",
        "roles": {"source_root_readonly": str(SOURCE_ROOT),
                  "owned_controller_root": str(OWNED_ROOT),
                  "host_side": "config/index SHA validation only",
                  "container_side": "named /bench/michaelfeil paths only"},
        "container": CONTAINER,
        "b66": {"container_path": B66, "host_present": Path(B66).is_file()},
        "vendor": {"container_path": VENDOR, "host_present": Path(VENDOR).is_dir()},
        "models": models,
        "assets": assets,
        "runtime": runtime,
        "source": source,
        "binaries": binaries,
        "native_policy": policy,
        "oxide_cuda_graphs": False,
        "non_claims": ["author did not run Docker/GPU/Torch/engine"],
    }
    print(json.dumps(report, indent=2))
    return report


def phase_plan(args):
    models = validate_model_records(args.model_name)
    policy = native_policy_proof()
    consumer = str(PAIRED_BINARY)
    life = str(LIFE_BINARY)
    plan = {
        "schema": "oxide-qwen35vl-map-policy-plan/v1",
        "roles": {"source_root_readonly": str(SOURCE_ROOT),
                  "owned_controller_root": str(OWNED_ROOT)},
        "models": {name: {"host_path": row["host_path"],
                          "container_path": row["container_path"]}
                   for name, row in models.items()},
        "native_policy": policy["policy"],
        "oxide_cuda_graphs": False,
        "phases": {
            "cpu": ["python3", "qwen35_vl_map_policy_controller_v3.py", "--phase", "cpu",
                    "--run", "--model-name", "Qwen3.5-0.8B", "--case", "single-cats",
                    "--output", "/tmp/oxide-qwen35vl-sdk-map-policy-output/cpu-0.8B-single-v3"],
            "paired": ["python3", "qwen35_vl_map_policy_controller_v3.py", "--phase", "paired",
                       "--run", "--model-name", "Qwen3.5-0.8B", "--case", "single-cats",
                       "--consumer", consumer,
                       "--output", "/tmp/oxide-qwen35vl-sdk-map-policy-output/paired-0.8B-single-v3"],
            "life": ["python3", "qwen35_vl_map_policy_controller_v3.py", "--phase", "life",
                     "--run", "--model-name", "Qwen3.5-0.8B", "--case", "single-cats",
                     "--life-consumer", life,
                     "--output", "/tmp/oxide-qwen35vl-sdk-map-policy-output/life-0.8B-single-v3"],
        },
        "container_env": remote_env("/tmp/oxide-qwen35vl-map-policy/plan", 0),
    }
    print(json.dumps(plan, indent=2))
    return plan


def fresh_output(path):
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"fresh output directory required: {output}")
    if output.is_relative_to(OWNED_ROOT):
        raise RuntimeError("output must be outside the owned source tree")
    output.mkdir(parents=True)
    return output


def run_phase(args, phase):
    if os.geteuid() != 0 and phase in ("cpu", "paired", "life"):
        raise RuntimeError("Docker phases are Root-only")
    output = fresh_output(args.output)
    run_id = f"{phase}-{time.time_ns()}"
    remote = f"{REMOTE_BASE}/{run_id}"
    models = validate_model_records(args.model_name)
    assets = validate_assets()
    runtime = validate_runtime()
    policy = native_policy_proof()
    source_proof = source_pin_proof()
    binaries = binary_proof(PAIRED_BINARY, LIFE_BINARY)
    consumer = args.consumer if phase != "life" else args.life_consumer
    consumer = Path(consumer)
    if not consumer.is_file():
        raise RuntimeError(f"consumer binary missing: {consumer}")
    selected_binary = sha(consumer)
    if phase == "paired" and consumer.resolve() == PAIRED_BINARY.resolve() and selected_binary != PAIRED_SHA:
        raise RuntimeError("paired binary does not match pinned 3780")
    if phase == "life" and consumer.resolve() == LIFE_BINARY.resolve() and selected_binary != LIFE_SHA:
        raise RuntimeError("owning-life binary does not match pinned 21c")

    cases = build_cases(output / "cases", remote, args.case)
    report = {
        "schema": "oxide-qwen35vl-map-policy-controller/v4", "phase": phase,
        "run_id": run_id, "remote": remote, "container": CONTAINER,
        "GPU_executed": phase != "cpu", "models": models, "cases": [],
        "assets": assets, "runtime": runtime, "native_policy": policy,
        "source": source_proof, "binaries": binaries,
        "oxide_cuda_graphs": False, "consumer": {"path": str(consumer),
                                                 "sha256": selected_binary},
        "roles": {"source_root_readonly": str(SOURCE_ROOT),
                  "owned_controller_root": str(OWNED_ROOT)},
    }
    persist = lambda: (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    persist()
    try:
        report["stage"] = stage(remote, consumer, cases)
        persist()
        for name in selected_models(args.model_name):
            model = models[name]
            for case in cases:
                label = f"{phase}-{name}-{case['name']}".replace("/", "_")
                out_case = output / f"{name}-{case['name']}"
                out_case.mkdir()
                remote_media = f"{remote}/cases/{case['manifest'].name}"
                if phase == "cpu":
                    preflight_out = f"{remote}/preflight-{label}"
                    argv = sdk_preflight_argv(remote, name, model,
                                              remote_media, preflight_out)
                    run_container(remote, "", argv, output / f"{label}-sdk.log")
                    plan_out = f"{remote}/plan-{label}"
                    argv = native_plan_argv(remote, model, remote_media, plan_out)
                    run_container(remote, "", argv, output / f"{label}-plan.log")
                    retrieve(preflight_out, out_case / "sdk-preflight")
                    retrieve(plan_out, out_case / "native-plan")
                    report["cases"].append({"model": name, "fixture": case["name"],
                                            "kind": case["kind"],
                                            "container_path": model["container_path"],
                                            "host_path": model["host_path"],
                                            "output": str(out_case),
                                            "status": "cpu-stage-passed"})
                elif phase == "paired":
                    native_out = f"{remote}/out-native-{label}"
                    oxide_out = f"{remote}/out-oxide-{label}.json"
                    argv = native_whole_argv(remote, name, model,
                                             remote_media, native_out)
                    run_container(remote, 0, argv, output / f"{label}-native.log")
                    argv = oxide_whole_argv(remote, model, remote_media, oxide_out)
                    run_container(remote, 0, argv, output / f"{label}-oxide.log")
                    retrieve(native_out, out_case / "native")
                    retrieve(oxide_out, out_case / "oxide.json")
                    compare_whole(out_case / "native/native-whole.json",
                                  out_case / "oxide.json", out_case / "pair.json")
                    report["cases"].append({"model": name, "fixture": case["name"],
                                            "kind": case["kind"],
                                            "container_path": model["container_path"],
                                            "host_path": model["host_path"],
                                            "output": str(out_case),
                                            "status": "paired-passed"})
                else:
                    oxide_out = f"{remote}/out-life-{label}.json"
                    argv = oxide_life_argv(remote, model, remote_media, oxide_out)
                    run_container(remote, 0, argv, output / f"{label}.log")
                    retrieve(oxide_out, out_case / "life.json")
                    life = load_json(out_case / "life.json")
                    if (life.get("status") != "passed" or
                            life.get("invalid_media", {}).get("rejected") is not True):
                        raise RuntimeError(f"owning lifecycle did not pass: {name}/{case['name']}")
                    report["cases"].append({"model": name, "fixture": case["name"],
                                            "kind": case["kind"],
                                            "container_path": model["container_path"],
                                            "host_path": model["host_path"],
                                            "output": str(out_case),
                                            "status": "owning-life-passed"})
                persist()
        report["status"] = {"cpu": "root-cpu-stage-passed",
                            "paired": "root-paired-quality-passed",
                            "life": "root-owning-life-passed"}[phase]
        persist()
        print(json.dumps({"status": report["status"], "output": str(output)}))
        return report
    except BaseException as error:
        report.update(status="failed", error=str(error))
        persist()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pins", "plan", "cpu", "paired", "life"),
                        required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--consumer", default=str(PAIRED_BINARY))
    parser.add_argument("--life-consumer", default=str(LIFE_BINARY))
    parser.add_argument("--model-name", action="append", default=None,
                        help="restrict to named checkpoint(s); default all five")
    parser.add_argument("--case", action="append", default=None,
                        help="restrict to named fixture case(s); default all three")
    args = parser.parse_args()
    if args.phase == "pins":
        phase_pins(args)
        return
    if args.phase == "plan":
        phase_plan(args)
        return
    if not args.run:
        raise SystemExit("--run is required for cpu/paired/life")
    if args.output is None:
        raise SystemExit("--output FRESHDIR is required")
    run_phase(args, args.phase)


if __name__ == "__main__":
    main()
