#!/usr/bin/env python3
"""Plan/execute the real paired Native b66 CPU stage.

- Model weights are read only from the container's existing /bench mapping.
- No checkpoint is copied and no config is rewritten.
- Only controller source, the Oxide host consumer, fixture pixels and the
  geometry report are staged into a fresh remote run directory.
- The inner remote_guard owns the container process group; the host CLI is not
  the containment boundary.
"""
from __future__ import annotations
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "infinity-torch214-compare-sdk"
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]
SDK_PY = "/opt/torch-sdk/bin/python3.12"
B66 = "/tmp/native214-fa4-mode20/install/lib/libtei_torch.so"
VENDOR = "/tmp/oxide-host-cublas-130200"
HOST_CONSUMER = "/tmp/opencode/qwen-vl-host-consumer-e7baa-target/release/examples/qwen3_vl_host_consumer"
NATIVE_CONSUMER = "/tmp/opencode/qwen-vl-native-release-target/release/qwen3-vl-b66-consumer"
PACKET_8B = str(ROOT / "root-gate/native/Root-control-packet.native.json")
FIXTURES = str(ROOT / "native-baseline/fixtures")
GEOMETRY = str(ROOT / "qwen-vl-geometry-v1/CONFIG-TUPLES.json")

CASES = [
    {
        "name": "qwen3-vl-8b",
        "host": "/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B",
        "container": "/bench/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B",
        "native_packet": PACKET_8B,
        "candidate": {"mode": "all", "gpu_forward": True},
    },
    {
        "name": "colqwen3-4b",
        "host": "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b",
        "container": "/bench/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b",
        "native_packet": None,
        "candidate": {"mode": "all", "gpu_forward": True},
    },
    {
        "name": "nemotron-4b",
        "host": "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2",
        "container": "/bench/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2",
        "native_packet": None,
        "candidate": {"mode": "all", "gpu_forward": True},
    },
    {
        "name": "nemotron-8b-optional",
        "host": "/tmp/perplexity-embedding-audit/multivector-support/nvidia--nemotron-colembed-vl-8b-v2",
        "container": None,
        "native_packet": None,
        "candidate": {"mode": "all", "gpu_forward": True},
        "skip_reason": "No /bench mapping verified; do not copy weights. Root may add a read-only mapping.",
    },
]


def docker_env():
    env = dict(os.environ)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        env.pop(key, None)
    return env


def guard_command(remote: Path, state: str, seconds: int, inner: list[str]) -> list[str]:
    return [
        *DOCKER, "exec", "-e", "CUDA_VISIBLE_DEVICES=", CONTAINER,
        "setsid", "--wait", SDK_PY, str(remote / "remote_guard.py"),
        "--state", str(remote / f"state-{state}.json"), "--seconds", str(seconds),
        "--", *inner,
    ]


def plan(run_id: str) -> dict:
    remote = Path("/tmp/qwen3-vl-native-stage") / run_id
    commands = []
    for case in CASES:
        if case["container"] is None:
            commands.append({"case": case["name"], "status": "skipped", "reason": case["skip_reason"]})
            continue
        verify = guard_command(remote, f"bind-{case['name']}", 1200, [
            "taskset", "-c", "160-167", SDK_PY, str(remote / "verify_b66.py")
        ])
        candidate = guard_command(remote, f"cpu-{case['name']}", 1200, [
            "taskset", "-c", "160-167", str(remote / "qwen3_vl_host_consumer"),
            "--checkpoint", case["container"],
            "--fixtures", str(remote / "fixtures.json"),
            "--mode", "all",
            "--out", str(remote / "candidate-cpu" / case["name"]),
        ])
        native_cpu = None
        if case["native_packet"] is not None:
            native_cpu = guard_command(remote, f"native-cpu-{case['name']}", 2400, [
                "taskset", "-c", "160-167", SDK_PY, str(remote / "native_vl_cpu.py"),
                "--library", B66,
                "--packet", str(remote / f"{case['name']}.native.json"),
                "--fixtures", str(remote / "fixtures.json"),
                "--fixture-root", str(remote / "fixtures"),
                "--out", str(remote / "native-cpu" / f"{case['name']}.json"),
            ])
        commands.append({
            "case": case["name"],
            "container_checkpoint": case["container"],
            "host_checkpoint_readonly": case["host"],
            "verify_bindings": verify,
            "native_cpu_whole_model": native_cpu,
            "cpu_candidate": candidate,
            "quality_candidate": {
                "native_packet": case["native_packet"],
                "native": None if case["native_packet"] is None else guard_command(remote, f"quality-native-{case['name']}", 1800, [
                    "taskset", "-c", "184-191", str(remote / "qwen3-vl-b66-consumer"),
                    str(remote / f"{case['name']}.native.json"), str(remote / "fixtures.json"), B66,
                    str(remote / "native-quality" / case["name"]),
                    "--library-sha", "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e",
                ]),
                "oxide": guard_command(remote, f"quality-oxide-{case['name']}", 1800, [
                    "taskset", "-c", "184-191", str(remote / "qwen3_vl_host_consumer"),
                    "--checkpoint", case["container"], "--fixtures", str(remote / "fixtures.json"),
                    "--mode", "gpu-forward", "--out", str(remote / "oxide-quality" / case["name"]),
                ]),
                "oxide_lifecycle": guard_command(remote, f"quality-life-{case['name']}", 1800, [
                    "taskset", "-c", "184-191", str(remote / "qwen3_vl_host_consumer"),
                    "--checkpoint", case["container"], "--fixtures", str(remote / "fixtures.json"),
                    "--mode", "gpu-lifecycle", "--out", str(remote / "oxide-lifecycle" / case["name"]),
                ]),
            },
        })
    return {
        "schema": "oxide-qwen3-vl-native-stage/v2",
        "run_id": run_id,
        "container": CONTAINER,
        "remote_dir": str(remote),
        "b66": B66,
        "vendor": VENDOR,
        "no_weight_copy": True,
        "read_only_models": [c["container"] for c in CASES if c["container"]],
        "staged_inputs": {
            "controller": str(ROOT / "native-b66-cpu-stage"),
            "host_consumer": HOST_CONSUMER,
            "native_consumer": NATIVE_CONSUMER,
            "fixtures": FIXTURES,
            "geometry": GEOMETRY,
        },
        "commands": commands,
        "status": "pending_root_cpu_and_quality",
    }


def stage(remote: Path) -> None:
    remote.mkdir(parents=True, exist_ok=True)
    subprocess.run([*DOCKER, "exec", CONTAINER, "mkdir", "-p", str(remote)], check=True, env=docker_env())
    subprocess.run(
        [*DOCKER, "cp", f"{ROOT / 'native-b66-cpu-stage'}/.", f"{CONTAINER}:{remote}"],
        check=True,
        env=docker_env(),
    )
    subprocess.run(
        [*DOCKER, "cp", FIXTURES, f"{CONTAINER}:{remote / 'fixtures'}"],
        check=True,
        env=docker_env(),
    )
    # Rewrite only host paths in the fixture manifest and packet JSONs. No
    # checkpoint weight is copied and no config is rewritten.
    fixture_doc = json.loads(Path(FIXTURES).read_text())
    for fixture in fixture_doc["fixtures"]:
        for image in fixture.get("images", []):
            image["pixels"] = str(remote / "fixtures" / Path(image["pixels"]).name)
    staging = Path("/tmp/qwen3-vl-native-stage-json") / remote.name
    staging.mkdir(parents=True, exist_ok=True)
    fixtures_out = staging / "fixtures.json"
    fixtures_out.write_text(json.dumps(fixture_doc, indent=2) + "\n")
    subprocess.run(
        [*DOCKER, "cp", str(fixtures_out), f"{CONTAINER}:{remote / 'fixtures.json'}"],
        check=True,
        env=docker_env(),
    )
    for case in CASES:
        if case["native_packet"] is None or case["container"] is None:
            continue
        packet = json.loads(Path(case["native_packet"]).read_text())
        packet["checkpoint"] = case["container"]
        packet["fixtures"] = str(remote / "fixtures.json")
        packet_out = staging / f"{case['name']}.native.json"
        packet_out.write_text(json.dumps(packet, indent=2) + "\n")
        subprocess.run(
            [*DOCKER, "cp", str(packet_out), f"{CONTAINER}:{remote / f'{case['name']}.native.json'}"],
            check=True,
            env=docker_env(),
        )
    for source in (HOST_CONSUMER, NATIVE_CONSUMER, GEOMETRY):
        subprocess.run([*DOCKER, "cp", source, f"{CONTAINER}:{remote / Path(source).name}"], check=True, env=docker_env())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", choices=["cpu", "quality"], default="cpu")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan_doc = plan(args.run_id)
    if args.plan or not args.run:
        print(json.dumps(plan_doc, indent=2))
        return
    if args.output is None or args.output.exists():
        raise SystemExit("--run requires a fresh --output directory")
    remote = Path(plan_doc["remote_dir"])
    stage(remote)
    results = {"schema": "oxide-qwen3-vl-native-stage-results/v2", "run_id": args.run_id, "phase": args.phase, "commands": []}
    for entry in plan_doc["commands"]:
        if entry.get("status") == "skipped":
            results["commands"].append(entry)
            continue
        selected = [entry["verify_bindings"]]
        if entry.get("native_cpu_whole_model") is not None:
            selected.append(entry["native_cpu_whole_model"])
        selected.append(entry["cpu_candidate"])
        if args.phase == "quality":
            if entry["quality_candidate"]["native"] is None:
                results["commands"].append({"case": entry["case"], "status": "native_packet_missing"})
                continue
            selected = [entry["quality_candidate"]["native"], entry["quality_candidate"]["oxide"]]
        for command in selected:
            completed = subprocess.run(command, text=True, capture_output=True, env=docker_env())
            results["commands"].append({"argv": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
            if completed.returncode != 0:
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
                raise SystemExit(f"remote command failed: {shlex.join(command)}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()
