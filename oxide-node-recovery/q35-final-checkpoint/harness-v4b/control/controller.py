#!/usr/bin/env python3
"""External process-group controller for the Rust harness. Runs one model per process."""
import argparse, json, os, signal, subprocess, sys, time

MODES = {
    "validate-only": [],
    "metadata": [],
    "quality": ["--pool"],
    "life-service": ["--pool"],
    "life-core": ["--pool"],
}
POOLS = ["cls", "mean", "last-token"]


def run(cmd, timeout, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        try:
            code = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            code = None
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
    return {"command": cmd, "returncode": code, "timed_out": timed_out,
            "duration_seconds": round(time.time() - started, 3),
            "log": str(log_path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--model", default="/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", action="append", choices=list(MODES), required=True)
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    root = __import__("pathlib").Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
    if not env.get("CUDA_VISIBLE_DEVICES"):
        print("refusing: CUDA_VISIBLE_DEVICES must be explicitly set for GPU runs", file=sys.stderr)
        return 2
    runs = []
    for mode in args.mode:
        variants = POOLS if mode in ("quality", "life-service", "life-core") else [None]
        for pool in variants:
            label = mode if pool is None else f"{mode}-{pool}"
            report = root / f"{label}.report.json"
            log = root / f"{label}.log"
            cmd = [args.binary, mode, "--cases", args.cases, "--report", str(report)]
            if mode not in ("validate-only",):
                cmd += ["--model", args.model]
            if pool:
                cmd += ["--pool", pool]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            outcome = run(cmd, args.timeout, log, env)
            fatal = None
            if outcome["returncode"] == 0 and report.is_file():
                try:
                    data = json.loads(report.read_text())
                    if data.get("status") != "ok" or data.get("errors"):
                        fatal = "report status/errors are not clean"
                    if data.get("typed_errors"):
                        fatal = "typed unexpected BackendError present"
                    for case in data.get("quality", {}).get("cases", []):
                        if case.get("status") != "ok":
                            fatal = f"case {case.get('name')} status {case.get('status')}"
                            break
                except Exception as exc:
                    fatal = f"report parse: {exc}"
            elif outcome["returncode"] == 0:
                fatal = "success exit without report"
            outcome["fatal"] = fatal
            if fatal:
                outcome["returncode"] = outcome["returncode"] or 1
            runs.append({"label": label, "report": str(report), **outcome})
    summary = {"schema": "oxide-q35moe-unified-controller/v1", "runs": runs}
    (root / "controller-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(r["returncode"] == 0 and not r["timed_out"] for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
