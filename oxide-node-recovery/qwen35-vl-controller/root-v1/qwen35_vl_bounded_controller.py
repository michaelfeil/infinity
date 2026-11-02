#!/usr/bin/env python3
"""Bounded Root gate controller for the Qwen3.5-VL owning lifecycle.

CPU/pins phases are executable without Docker. The paired and life phases stage
the pinned consumer, media fixtures and runtime directories into the existing
SDK container, use explicit remote paths, check every docker cp return code and
retrieve every output. No GPU or Docker command is executed by the author.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "infinity-torch214-compare-sdk"
PYTHON = "/opt/torch-sdk/bin/python3.12"
B66 = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
VENDOR = "/tmp/oxide-host-cublas-130200"
KERNELS = "/tmp/infinity-oxide-parity-20261101/device-union124-all3-v1/kernels"
FA4 = "/tmp/oxide-fa4-canonical26-derived-v4/bundle"
GDN_RAW = "/tmp/infinity-oxide-parity-20261101/qwen35-gdn-foundation-v1/kernels"
GDN_PROVIDER = "/tmp/infinity-oxide-parity-20261101/qwen35-gdn-aot-compiled-root-v2"
GDN_AUX = "/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels"
REMOTE_BASE = "/tmp/oxide-qwen35vl-bounded"
CPU_LIST = "88-95"
MODEL = "/workspace/model-performance/michaelfeil/onejev-perf/backbone-comparison-20261023/models/Qwen3.5-0.8B"
DOCKER_SELECTORS = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_", "INFINITY_", "LD_"))}
    for key in DOCKER_SELECTORS:
        env.pop(key, None)
    return env

def docker(*args, timeout=3600):
    return subprocess.run(["docker", "--host=unix:///var/run/docker.sock", *map(str, args)],
                          env=clean_env(), capture_output=True, text=True, timeout=timeout, check=False)

def checked_docker(*args, timeout=3600):
    done = docker(*args, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"docker {' '.join(map(str,args))} failed rc={done.returncode}: {done.stderr}")
    return done

def checked_cp(source, target):
    source = str(source)
    target = str(target)
    if source.endswith("/."):
        checked_docker("cp", source, f"{CONTAINER}:{target}")
    else:
        checked_docker("cp", source, f"{CONTAINER}:{target}")

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
        "LD_LIBRARY_PATH": f"{VENDOR}:/tmp/native214-fa4-mode20/install/lib:/opt/torch-sdk/lib/python3.12/site-packages/torch/lib:/usr/local/cuda/lib64",
    }
    out = []
    for key, value in values.items():
        out += ["-e", f"{key}={value}"]
    return out

def stage(remote, consumer, media):
    checked_docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON, "-c",
                   f"from pathlib import Path;[Path(p).mkdir(parents=True) for p in [{remote!r}+'/bin',{remote!r}+'/media',{remote!r}+'/fixtures',{remote!r}+'/runtime/kernels',{remote!r}+'/runtime/fa4',{remote!r}+'/runtime/gdn-raw',{remote!r}+'/runtime/gdn-provider',{remote!r}+'/runtime/gdn-aux',{remote!r}+'/native']]")
    checked_cp(consumer, f"{remote}/bin/qwen35-vl-consumer")
    checked_cp(ROOT / "native/native_whole_model.py", f"{remote}/native/native_whole_model.py")
    manifest = json.loads(Path(media).read_text())
    base = Path(media).resolve().parent
    for fixture in manifest.get("fixtures", []):
        pixel = Path(fixture["pixels"])
        if not pixel.is_absolute():
            pixel = base / pixel
        target = f"{remote}/fixtures/{pixel.name}"
        checked_cp(pixel, target)
        fixture["pixels"] = target
    checked_cp(media, f"{remote}/media/{Path(media).name}")
    # Rewrite the remote copy after staging the original so both stay explicit.
    code = f"from pathlib import Path;Path({remote!r}+'/media/{Path(media).name}').write_text({json.dumps(json.dumps(manifest))})"
    checked_docker("exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER, PYTHON, "-c", code)
    for source, target in [(KERNELS, "kernels"), (FA4, "fa4"), (GDN_RAW, "gdn-raw"),
                           (GDN_PROVIDER, "gdn-provider"), (GDN_AUX, "gdn-aux")]:
        if not Path(source).is_dir():
            raise RuntimeError(f"missing pinned runtime directory {source}")
        checked_cp_dir(source, f"{remote}/runtime/{target}")
    return f"{remote}/bin/qwen35-vl-consumer", f"{remote}/media/{Path(media).name}"

def run_container(remote, gpu, argv, log_path):
    command = ["exec", *remote_env(remote, gpu), CONTAINER, "taskset", "-c", CPU_LIST, *map(str, argv)]
    done = docker(*command, timeout=3600)
    Path(log_path).write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f"container command failed rc={done.returncode}: {argv}")
    return done

def retrieve(container_path, host_path):
    Path(host_path).parent.mkdir(parents=True, exist_ok=True)
    checked_docker("cp", f"{CONTAINER}:{container_path}", str(host_path))

def finite(values):
    return isinstance(values, list) and len(values) > 0 and all(isinstance(x, (int, float)) and float(x) == float(x) and abs(float(x)) != float("inf") for x in values)

def compare_whole(native_file, oxide_file, out_file):
    native = json.loads(Path(native_file).read_text())
    oxide = json.loads(Path(oxide_file).read_text())
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
            comparisons.append({"index": key[0], "kind": key[1], "class": "decision", "elements": len(av), "max_abs": maximum, "passed": passed})
        else:
            av, bv = a.get("values"), b.get("values")
            if not finite(av) or not finite(bv) or len(av) != len(bv):
                raise RuntimeError(f"empty/invalid embedding values at {key}")
            maximum = max(abs(x - y) for x, y in zip(av, bv))
            passed = maximum <= 0.015 + 0.015 * max(max(abs(x) for x in av), 1.0)
            comparisons.append({"index": key[0], "kind": key[1], "class": "embedding", "elements": len(av), "max_abs": maximum, "passed": passed})
    report = {"schema": "oxide-qwen35vl-bounded-pair/v1", "status": "passed" if all(x["passed"] for x in comparisons) else "failed", "comparisons": comparisons}
    Path(out_file).write_text(json.dumps(report, indent=2) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("paired whole comparison failed")

def phase_pins(args):
    paths = {"consumer": args.consumer, "kernels": KERNELS, "fa4": FA4,
             "gdn_raw": GDN_RAW, "gdn_provider": GDN_PROVIDER, "gdn_aux": GDN_AUX}
    if args.media is not None:
        paths["media"] = args.media
    missing = [name for name, path in paths.items() if not Path(path).exists()]
    report = {"schema": "oxide-qwen35vl-bounded-pins/v1", "missing": missing,
              "hashes": {name: (sha(path) if Path(path).is_file() else None) for name, path in paths.items()},
              "container": CONTAINER, "b66": {"container_path": B66, "host_present": Path(B66).is_file()},
              "vendor": {"container_path": VENDOR, "host_present": Path(VENDOR).is_dir()},
              "manifest": sha(Path(KERNELS) / "manifest.json"),
              "fa4_manifest": sha(Path(FA4) / "sm_90/manifest.json"),
              "gdn_manifest": sha(Path(GDN_RAW) / "manifest.json")}
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)

def phase_cpu(args):
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("fresh output required")
    done = subprocess.run(["python3", str(ROOT / "root-v1/qwen35_vl_sdk_controller_v2.py"), "--phase", "cpu", "--run", "--output", str(out)], env=clean_env(), text=True)
    if done.returncode:
        raise SystemExit(done.returncode)

def phase_plan(args):
    gate = json.loads((ROOT / "root-v1/ROOT-GATE-V8.json").read_text())
    print(json.dumps({"schema": "oxide-qwen35vl-bounded-plan/v1",
                      "phases": {name: body["command"] for name, body in gate["phases"].items()},
                      "env": remote_env("/tmp/oxide-qwen35vl-bounded/plan", 0)}, indent=2))

def phase_paired(args):
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("fresh output required")
    out.mkdir(parents=True)
    run_id = f"paired-{time.time_ns()}"
    remote = f"{REMOTE_BASE}/{run_id}"
    consumer, media = stage(remote, args.consumer, args.media)
    native_out = f"{remote}/out-native"
    oxide_out = f"{remote}/out-oxide.json"
    run_container(remote, 0, [PYTHON, f"{remote}/native/native_whole_model.py", "--allow-gpu",
                              "--library", B66, "--model", MODEL, "--media", media,
                              "--out", native_out, "--pool", "mean"], out / "native.log")
    run_container(remote, 0, [consumer, "whole-factory", "--model", MODEL, "--media", media,
                              "--device", "0", "--pool", "mean", "--out", oxide_out], out / "oxide.log")
    retrieve(native_out, out / "native")
    retrieve(oxide_out, out / "oxide.json")
    compare_whole(out / "native/native-whole.json", out / "oxide.json", out / "pair.json")

def phase_life(args):
    out = Path(args.output)
    if out.exists():
        raise RuntimeError("fresh output required")
    out.mkdir(parents=True)
    run_id = f"life-{time.time_ns()}"
    remote = f"{REMOTE_BASE}/{run_id}"
    consumer, media = stage(remote, args.consumer, args.media)
    oxide_out = f"{remote}/out-life.json"
    run_container(remote, 0, [consumer, "owning-lifecycle", "--model", MODEL, "--media", media,
                              "--device", "0", "--iterations", "4", "--out", oxide_out], out / "life.log")
    retrieve(oxide_out, out / "life.json")
    report = json.loads((out / "life.json").read_text())
    if report.get("status") != "passed" or report.get("invalid_media", {}).get("rejected") is not True:
        raise RuntimeError("owning lifecycle report did not pass")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pins", "cpu", "paired", "life", "plan"), required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    if args.phase == "pins":
        phase_pins(args); return
    if args.phase == "plan":
        phase_plan(args); return
    if not args.run:
        raise SystemExit("--run is required for cpu/paired/life")
    if args.output is None:
        raise SystemExit("--output FRESHDIR is required")
    if args.media is None and args.phase != "cpu":
        raise SystemExit("--media is required")
    {"cpu": phase_cpu, "paired": phase_paired, "life": phase_life}[args.phase](args)

if __name__ == "__main__":
    main()
