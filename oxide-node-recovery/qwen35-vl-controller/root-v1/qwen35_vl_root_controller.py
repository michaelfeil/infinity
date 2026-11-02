#!/usr/bin/env python3
"""Root-owned paired controller for the isolated Qwen3.5-VL leaf.

The author never runs Docker, GPU, CUDA, Torch, NVCC, network or device builds.
Root (GPU 0) executes the printed/`--execute` commands inside the pinned SDK
container. Native gold is the actual b66 C++ tower compiled from the public
Native sources; Oxide gold is the real `qwen35-vl-consumer tower-device` path.

Stages:
  host-preflight  verify base pin, checkpoints, preprocessors, b66 identity,
                  old-export proof and source pins (no payload read)
  native-dump     dump original model.visual.* tensors + flattened config
  native-build    compile native/tower_probe.cpp against the public sources
  native-gold     run the Native tower per fixture
  oxide-tower     run the Oxide tower per fixture (CPU or device)
  pair            compare Native vs Oxide merged rows (rtol/atol)
  lifetime        repeated Oxide tower create/run/drop
  perf            repeated Oxide tower run
  text            full Native tei_forward vs Oxide text path (blocked until
                  the official text integration exposes external mRoPE)
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKT = ROOT / "device-packet"
PLAN = json.loads((PKT / "BUILD-PLAN.json").read_text())
CHECKPOINTS = json.loads((PKT / "checkpoints.json").read_text())["checkpoints"]
PINS = json.loads((PKT / "source-pins.json").read_text())
CONTAINER = PLAN["sdk"]["container"]
MARK = "QWEN35_VL_PREFLIGHT_COMPLETE"
KERNELS = "/tmp/infinity-oxide-parity-20261101/qwen35-public-f417-device99-v3-offline/kernels"
FA4 = "/tmp/oxide-fa4-canonical26-derived-v4/bundle"
GDN_RAW = "/tmp/infinity-oxide-parity-20261101/qwen35-gdn-foundation-v1/kernels"
GDN_PROVIDER = "/tmp/infinity-oxide-parity-20261101/qwen35-gdn-aot-compiled-root-v2"
GDN_AUX = "/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels"

def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run(argv, execute: bool, env=None, dry=None):
    if not execute:
        print(" ".join(str(item) for item in argv))
        return None
    completed = subprocess.run([str(item) for item in argv], text=True,
                               capture_output=True, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {argv}\n"
                           f"stdout={completed.stdout}\nstderr={completed.stderr}")
    if dry is not None:
        dry.parent.mkdir(parents=True, exist_ok=True)
        dry.write_text(completed.stdout)
    return completed

def docker_exec(remote_argv, execute: bool, env=None, cpus="88-95"):
    quoted = " ".join(str(item) for item in remote_argv)
    command = ["docker", "exec"]
    for key, value in (env or {}).items():
        command += ["-e", f"{key}={value}"]
    command += [CONTAINER, "bash", "-lc", f"taskset -c {cpus} {quoted}"]
    return run(command, execute)

def host_preflight(args):
    failures = []
    if PINS["base_public_commit_full"] != "e582ce9":
        failures.append("base pin is not e582ce9")
    for name, record in CHECKPOINTS.items():
        path = Path(record["path"])
        if not path.is_dir():
            failures.append(f"{name}: missing checkpoint directory")
            continue
        for field, filename in [("config_sha256", "config.json")]:
            target = path / filename
            if not target.is_file():
                failures.append(f"{name}: missing {filename}")
            elif sha(target) != record[field]:
                failures.append(f"{name}: {filename} hash drift")
        index_target = path / "model.safetensors.index.json"
        if index_target.is_file():
            if record.get("index_sha256") and sha(index_target) != record["index_sha256"]:
                failures.append(f"{name}: index hash drift")
        elif not (path / "model.safetensors").is_file():
            failures.append(f"{name}: missing model.safetensors or index")
        if not record.get("combined_processor"):
            for filename in ["preprocessor_config.json", "video_preprocessor_config.json"]:
                if not (path / filename).is_file() and name != "Qwen3.5-35B-A3B":
                    failures.append(f"{name}: missing original {filename}")
    proof = json.loads((PKT / "old-exports-proof.json").read_text())
    if proof["status"] != "passed" or proof["old_exports"] != 86 \
            or proof["qwen35vl_additive_exports"] != 7:
        failures.append("old export proof is not passing")
    if not (PKT / "old86.txt").is_file():
        failures.append("missing old86 pin")
    for name, digest in PINS["files"].items():
        target = ROOT / name
        if not target.is_file() or sha(target) != digest:
            failures.append(f"source pin drift: {name}")
    sdk = PLAN["sdk"]
    preflight = {
        "schema": "oxide-qwen35vl-root-preflight/v1",
        "marker": MARK,
        "status": "passed" if not failures else "failed",
        "base_commit": PINS["base_public_commit"],
        "container": CONTAINER,
        "b66": {"path": sdk["b66"], "sha256": sdk["b66_sha256"]},
        "torch": {"python": sdk["torch_python"], "version": sdk["torch_version"],
                  "git": sdk["torch_git"], "root": sdk["torch_root"]},
        "cublaslt": sdk["cublaslt"],
        "checkpoints": {name: record["path"] for name, record in CHECKPOINTS.items()},
        "old_exports_proof": proof,
        "failures": failures,
    }
    out = Path(args.out) / "preflight.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps({"marker": MARK, "status": preflight["status"],
                      "failures": failures, "out": str(out)}))
    if failures:
        raise SystemExit(1)

def media_fixtures(path: Path):
    manifest = json.loads(path.read_text())
    fixtures = manifest.get("fixtures", [])
    if not fixtures:
        raise SystemExit(f"{path} has no fixtures")
    return manifest, fixtures

def write_native_fixtures(manifest: Path, out: Path):
    data, fixtures = media_fixtures(manifest)
    base = manifest.parent
    lines = []
    for fixture in fixtures:
        if fixture.get("kind", "image") != "image":
            continue
        name = fixture["name"]
        grid = fixture["grid"]
        merge = fixture["merge"]
        rows = grid[0] * grid[1] * grid[2]
        pixels = Path(fixture["pixels"])
        if not pixels.is_absolute():
            pixels = base / pixels
        patch_dim = fixture["patch_dim"]
        if pixels.stat().st_size != rows * patch_dim * 4:
            raise SystemExit(f"{pixels} size does not match rows*patch_dim*4")
        lines.append("\t".join(str(value) for value in
                               [name, rows, patch_dim, grid[0], grid[1], grid[2],
                                merge, fixture.get("token_start", 0), pixels]))
    out.write_text("\n".join(lines) + "\n")
    return out

def native_dump(args):
    checkpoint = CHECKPOINTS[args.model]["path"]
    remote = args.remote
    docker_exec(["mkdir", "-p", f"{remote}/tensors"], args.execute)
    docker_exec([PLAN["sdk"]["torch_python"], f"{remote}/dump_vision_tensors.py",
                 checkpoint, f"{remote}/tensors"], args.execute,
                env={"CUDA_VISIBLE_DEVICES": ""})
    if args.execute:
        run(["docker", "cp", f"{CONTAINER}:{remote}/tensors", str(Path(args.out) / "tensors")], True)
    print(json.dumps({"stage": "native-dump", "checkpoint": checkpoint,
                      "out": str(Path(args.out) / "tensors")}))

def native_build(args):
    command = [
        "cmake", "-S", f"{args.remote}/native", "-B", f"{args.remote}/build",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_PREFIX_PATH", PLAN["sdk"]["torch_root"],
    ]
    docker_exec(command, args.execute)
    docker_exec(["cmake", "--build", f"{args.remote}/build", "--target", "tower_probe", "-j", "4"], args.execute)
    print(json.dumps({"stage": "native-build", "binary": f"{args.remote}/build/tower_probe",
                      "build_plan": str(PKT / "BUILD-PLAN.json")}))

def native_gold(args):
    fixtures = write_native_fixtures(Path(args.media), Path(args.out) / "fixtures.txt")
    if args.execute:
        run(["docker", "cp", str(fixtures), f"{CONTAINER}:{args.remote}/fixtures.txt"], True)
    docker_exec(["mkdir", "-p", f"{args.remote}/gold"], args.execute)
    docker_exec([f"{args.remote}/build/tower_probe", f"{args.remote}/tensors",
                 f"{args.remote}/fixtures.txt", f"{args.remote}/gold",
                 args.dtype, "cuda:0"], args.execute,
                env={"CUDA_VISIBLE_DEVICES": args.gpu})
    if args.execute:
        run(["docker", "cp", f"{CONTAINER}:{args.remote}/gold", str(Path(args.out) / "gold")], True)
    print(json.dumps({"stage": "native-gold", "out": str(Path(args.out) / "gold")}))

def oxide_tower(args):
    command = [args.consumer, "tower-device" if args.device else "tower-cpu",
               "--model", CHECKPOINTS[args.model]["path"], "--media", args.media,
               "--out", args.oxide_out]
    if args.device:
        command += ["--device", args.gpu]
        docker_exec(command, args.execute, env={"CUDA_VISIBLE_DEVICES": args.gpu,
                                                "INFINITY_OXIDE_KERNEL_MODULE": args.kernel_module,
                                                "INFINITY_FA4_ROOT": args.fa4_root})
    else:
        run(command, args.execute)
    print(json.dumps({"stage": "oxide-tower", "out": args.oxide_out,
                      "arm": "device" if args.device else "cpu"}))

def pair(args):
    gold = json.loads((Path(args.gold) / "gold.json").read_text())
    oxide = json.loads(Path(args.oxide).read_text())
    gold_rows = {row["name"]: row for row in gold["rows"]}
    oxide_rows = {row["name"]: row for row in oxide["rows"]}
    if set(gold_rows) != set(oxide_rows):
        raise SystemExit(f"fixture sets differ: {sorted(gold_rows)} vs {sorted(oxide_rows)}")
    report = {"schema": "oxide-qwen35vl-pair/v1", "comparisons": [], "status": "passed"}
    for name in sorted(gold_rows):
        native = (Path(args.gold) / f"{name}.f32").read_bytes()
        values_native = [float.fromhex("0x0p+0")] if False else None
        import struct
        native_values = struct.unpack(f"<{len(native)//4}f", native)
        oxide_values = oxide_rows[name]["merged"]
        if len(native_values) != len(oxide_values):
            report["status"] = "failed"
            report["comparisons"].append({"name": name, "failure": "length mismatch",
                                          "native": len(native_values),
                                          "oxide": len(oxide_values)})
            continue
        diffs = [abs(a - b) for a, b in zip(native_values, oxide_values)]
        maximum = max(diffs) if diffs else 0.0
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        dot = sum(a * b for a, b in zip(native_values, oxide_values))
        nn = math.sqrt(sum(a * a for a in native_values))
        no = math.sqrt(sum(b * b for b in oxide_values))
        cosine = dot / (nn * no) if nn and no else 0.0
        passed = maximum <= args.atol + args.rtol * max(
            max((abs(a) for a in native_values), default=0.0), 1.0)
        if not passed:
            report["status"] = "failed"
        report["comparisons"].append({
            "name": name, "elements": len(diffs), "max_abs": maximum,
            "mean_abs": mean, "cosine": cosine, "passed": passed,
        })
    out = Path(args.out) / "pair.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out)}))
    if report["status"] != "passed":
        raise SystemExit(1)

def factory_plan(args):
    out = Path(args.out) / "run-plans"
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for name, record in CHECKPOINTS.items():
        target = out / f"{name}.json"
        command = [args.consumer, "plan-full", "--model", record["path"],
                   "--out", str(target)]
        run(command, args.execute)
        if args.execute:
            index[name] = json.loads(target.read_text())
    if args.execute:
        (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"stage": "factory-plan", "out": str(out), "count": len(CHECKPOINTS)}))

def native_whole(args):
    checkpoint = CHECKPOINTS[args.model]["path"]
    out = Path(args.out) / "native-whole"
    command = [PLAN["sdk"]["torch_python"], f"{args.remote}/native/native_whole_model.py",
               "--library", PLAN["sdk"]["b66"], "--model", checkpoint,
               "--media", args.media, "--out", str(out), "--pool", args.pool]
    if args.decision:
        command.append("--decision")
    command.append("--allow-gpu")
    docker_exec(command, args.execute,
                env={"CUDA_VISIBLE_DEVICES": args.gpu})
    if args.execute:
        run(["docker", "cp", f"{CONTAINER}:{out}", str(Path(args.out) / "native-whole")], True)
    print(json.dumps({"stage": "native-whole", "out": str(Path(args.out) / "native-whole")}))

def oxide_whole(args):
    command = [args.consumer, "whole-factory", "--model", CHECKPOINTS[args.model]["path"],
               "--media", args.media, "--device", args.gpu, "--pool", args.pool,
               "--fa4-root", args.fa4_root,
               "--gdn-raw-kernels", args.gdn_raw, "--gdn-provider", args.gdn_provider,
               "--gdn-aux-kernels", args.gdn_aux, "--out", args.oxide_out]
    docker_exec(command, args.execute, env={"CUDA_VISIBLE_DEVICES": args.gpu,
                                            "INFINITY_FA4_ROOT": args.fa4_root,
                                            "INFINITY_OXIDE_GDN_RAW_KERNELS": args.gdn_raw,
                                            "INFINITY_OXIDE_GDN_PROVIDER": args.gdn_provider,
                                            "INFINITY_OXIDE_GDN_AUX_KERNELS": args.gdn_aux})
    print(json.dumps({"stage": "oxide-whole", "out": args.oxide_out}))

def pair_whole(args):
    native = json.loads((Path(args.native) / "native-whole.json").read_text())
    oxide = json.loads(Path(args.oxide).read_text())
    native_rows = {(row.get("index", 0), row["kind"]): row for row in native["rows"]}
    oxide_rows = {(row.get("index", 0), row["kind"]): row for row in oxide["rows"]}
    if set(native_rows) != set(oxide_rows):
        raise SystemExit(f"whole-model row sets differ: {sorted(native_rows)} vs {sorted(oxide_rows)}")
    report = {"schema": "oxide-qwen35vl-whole-pair/v1", "comparisons": [], "status": "passed"}
    for key in sorted(native_rows):
        native_row = native_rows[key]
        oxide_row = oxide_rows[key]
        if "logits" in native_row or "logits" in oxide_row:
            native_logits = native_row.get("logits", [])
            oxide_logits = oxide_row.get("logits", [])
            if not native_logits or not oxide_logits or len(native_logits) != len(oxide_logits):
                report["status"] = "failed"
                report["comparisons"].append({"index": key[0], "kind": key[1], "failure": "empty_or_length_logits"})
                continue
            diffs = [abs(x - y) for x, y in zip(native_logits, oxide_logits)]
            action_delta = abs(native_row.get("action_probability", 0.0)
                               - oxide_row.get("action_probability", 0.0))
            maximum = max(diffs + [action_delta]) if diffs else action_delta
            passed = (len(native_logits) == len(oxide_logits)
                      and maximum <= args.atol + args.rtol)
            if not passed:
                report["status"] = "failed"
            report["comparisons"].append({"index": key[0], "kind": key[1],
                                          "max_abs": maximum, "passed": passed})
            continue
        a, b = native_row.get("values", []), oxide_row.get("values", [])
        if not a or not b or len(a) != len(b):
            report["status"] = "failed"
            report["comparisons"].append({"index": key[0], "kind": key[1], "failure": "empty_or_length_values"})
            continue
        diffs = [abs(x - y) for x, y in zip(a, b)]
        maximum = max(diffs) if diffs else 0.0
        mean = sum(diffs) / len(diffs) if diffs else 0.0
        passed = maximum <= args.atol + args.rtol * max(max((abs(x) for x in a), default=0.0), 1.0)
        if not passed:
            report["status"] = "failed"
        report["comparisons"].append({"index": key[0], "kind": key[1],
                                      "max_abs": maximum, "mean_abs": mean, "passed": passed})
    out = Path(args.out) / "pair-whole.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out)}))
    if report["status"] != "passed":
        raise SystemExit(1)

def lifetime_or_perf(args, command):
    argv = [args.consumer, command, "--model", CHECKPOINTS[args.model]["path"],
            "--media", args.media, "--device", args.gpu,
            "--iterations", str(args.iterations), "--out", args.out]
    docker_exec(argv, args.execute, env={"CUDA_VISIBLE_DEVICES": args.gpu,
                                         "INFINITY_OXIDE_KERNEL_MODULE": args.kernel_module,
                                         "INFINITY_FA4_ROOT": args.fa4_root})
    print(json.dumps({"stage": command, "out": args.out}))

def text_stage(args):
    message = {
        "schema": "oxide-qwen35vl-text-stage/v1",
        "status": "blocked",
        "reason": "full text prefill is owned by the official Qwen3.5 text integration; "
                  "this leaf supplies fused visual rows and external partial mRoPE phases",
        "native_command": "b66 tei_forward(ids, media_positions, TeiImage*, ...) with the original config",
        "oxide_requirement": "Qwen35 text engine must accept fused embeddings and external phases",
    }
    out = Path(args.out) / "text-stage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(message, indent=2) + "\n")
    print(json.dumps(message))

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    def common(p):
        p.add_argument("--model", choices=sorted(CHECKPOINTS), default="Qwen3.5-0.8B")
        p.add_argument("--execute", action="store_true")
        p.add_argument("--remote", default="/tmp/oxide-qwen35vl-root")
        p.add_argument("--out", default=str(ROOT / "root-v1/out"))
    p = sub.add_parser("host-preflight"); common(p); p.set_defaults(func=host_preflight)
    p = sub.add_parser("native-dump"); common(p); p.set_defaults(func=native_dump)
    p = sub.add_parser("native-build"); common(p); p.set_defaults(func=native_build)
    p = sub.add_parser("native-gold"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--dtype", default="bfloat16"); p.add_argument("--gpu", default="0")
    p.set_defaults(func=native_gold)
    p = sub.add_parser("oxide-tower"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    p.add_argument("--oxide-out", default=str(ROOT / "root-v1/out/oxide.json"))
    p.add_argument("--device", action="store_true"); p.add_argument("--gpu", default="0")
    p.add_argument("--kernel-module", default=KERNELS)
    p.add_argument("--fa4-root", default=FA4)
    p.set_defaults(func=oxide_tower)
    p = sub.add_parser("pair"); common(p); p.add_argument("--gold", required=True)
    p.add_argument("--oxide", required=True); p.add_argument("--atol", type=float, default=0.015)
    p.add_argument("--rtol", type=float, default=0.015); p.set_defaults(func=pair)
    p = sub.add_parser("lifetime"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    p.add_argument("--gpu", default="0"); p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--kernel-module", default=KERNELS)
    p.add_argument("--fa4-root", default=FA4)
    p.set_defaults(func=lambda a: lifetime_or_perf(a, "lifetime"))
    p = sub.add_parser("perf"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    p.add_argument("--gpu", default="0"); p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--kernel-module", default=KERNELS)
    p.add_argument("--fa4-root", default=FA4)
    p.set_defaults(func=lambda a: lifetime_or_perf(a, "perf"))
    p = sub.add_parser("factory-plan"); common(p)
    p.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    p.set_defaults(func=factory_plan)
    p = sub.add_parser("native-whole"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--pool", default="mean"); p.add_argument("--gpu", default="0")
    p.add_argument("--decision", action="store_true")
    p.set_defaults(func=native_whole)
    p = sub.add_parser("oxide-whole"); common(p); p.add_argument("--media", required=True)
    p.add_argument("--pool", default="mean"); p.add_argument("--gpu", default="0")
    p.add_argument("--consumer", default=str(ROOT / "target-opt3-noLTO/release/qwen35-vl-consumer"))
    p.add_argument("--oxide-out", default=str(ROOT / "root-v1/out/oxide-whole.json"))
    p.add_argument("--fa4-root", default=FA4)
    p.add_argument("--gdn-raw", default=GDN_RAW)
    p.add_argument("--gdn-provider", default=GDN_PROVIDER)
    p.add_argument("--gdn-aux", default=GDN_AUX)
    p.set_defaults(func=oxide_whole)
    p = sub.add_parser("pair-whole"); common(p); p.add_argument("--native", required=True)
    p.add_argument("--oxide", required=True); p.add_argument("--atol", type=float, default=0.015)
    p.add_argument("--rtol", type=float, default=0.015); p.set_defaults(func=pair_whole)
    p = sub.add_parser("text"); common(p); p.set_defaults(func=text_stage)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    args.func(args)

if __name__ == "__main__":
    main()
