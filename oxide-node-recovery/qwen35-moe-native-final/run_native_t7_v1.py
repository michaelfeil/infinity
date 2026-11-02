"""Root-only T7 Native gold using the already passed LIGHT V2 namespace."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
CONTROL = ROOT / "native-control-v2" / "root_control.py"
CASES = ROOT / "starter-t7-a-v1.json"
RUN_ID = "q35moe-native-light-v2"


def load_control():
    spec = importlib.util.spec_from_file_location("q35_frozen_native_control", CONTROL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--report", type=Path, default=ROOT / "root-native-t7-v1.report.json")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 3600:
        raise ValueError("timeout must be bounded in 1..3600")
    control = load_control()
    freeze = json.loads((ROOT / "native-light-freeze-v2.json").read_text())
    source_root = CONTROL.parent
    for name, entry in freeze["files"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"frozen Native source changed: {name}")
    selected = control.select_t7_a(CASES)
    step = control.phase_plan(SimpleNamespace(phase="starter-native"),
                              f"{control.CONTAINER_BASE}/{RUN_ID}")[0]
    step["argv"] += ["--device", "cuda:0"]
    plan = {"schema": "q35moe-native-t7-plan/v1", "run": args.run,
            "runid": RUN_ID, "case_file": str(CASES), "case_sha256": hashlib.sha256(CASES.read_bytes()).hexdigest(),
            "device": "cuda:0", "step": step, "engine_limit": 1,
            "scope": "actual GPU Native T7 mean/fit probe; no CPU model forward"}
    if not args.run:
        print(json.dumps(plan, indent=2))
        return 0
    host = control.HOST_BASE / RUN_ID
    if not (host / "light.json").is_file():
        raise ValueError("passed actual LIGHT V2 report is required")
    if args.report.exists():
        raise ValueError("refusing to overwrite an existing Root T7 report")
    if not json.loads((host / "light.json").read_text()).get("ok"):
        raise ValueError("LIGHT V2 did not pass")
    report = {**plan, "status": "running", "records": []}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    remote = f"{control.CONTAINER_BASE}/{RUN_ID}"
    try:
        staged, reuse = control.stage_host(RUN_ID)
        if not reuse or staged != host:
            raise ValueError("expected owned existing LIGHT V2 namespace")
        (host / "cases.json").write_text(json.dumps(selected, indent=2) + "\n")
        control.stage_container(host, remote, True, ["cases.json"])
        before = control.pins_container(remote, host, "t7-before")
        intended = control.guard_argv(step, remote, args.timeout)
        result = control.run_bounded(step, host, remote, args.timeout)
        report["records"].append(control.merge_record({"argv": intended}, result))
        control.copy_back(remote, host, step["copy_back"])
        after = control.pins_container(remote, host, "t7-after")
        report["pins"] = control.pin_gate(before, after, "starter-native")
        control.check_phase(SimpleNamespace(phase="starter-native"), host)
        gold = host / "native-gold-t7.json"
        local_gold = ROOT / "root-native-t7-v1.gold.json"
        if local_gold.exists():
            raise ValueError("refusing to overwrite prior Native gold")
        shutil.copy2(gold, local_gold)
        report.update(status="passed", gold=str(local_gold), gold_sha256=hashlib.sha256(local_gold.read_bytes()).hexdigest())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            control.copy_back(remote, host, step["copy_back"])
        except Exception as cleanup:
            report["evidence_copy_error"] = str(cleanup)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        raise
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
