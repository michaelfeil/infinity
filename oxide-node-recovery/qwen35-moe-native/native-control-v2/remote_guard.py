#!/usr/bin/env python3
"""Owned in-container guard (v4): bounded timeout plus identity-checked cleanup.

ROOT is this file's directory, so state and guard share the exact
/tmp/qwen35-moe-sdk-root-v1/RUNID namespace. Every stop re-checks PID start
ticks and command identity before signalling the owned process group.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parent
TIMEOUT = "/usr/bin/timeout"


def start_ticks(pid: int):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError, PermissionError):
        return False


def cmdline(pid: int):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError):
        return None
    return [item.decode() for item in raw if item]


def load_state(path: Path) -> dict:
    state = json.loads(Path(path).read_text())
    if (not isinstance(state, dict) or not isinstance(state.get("pid"), int)
            or not isinstance(state.get("start_ticks"), str)
            or not isinstance(state.get("argv"), list)):
        raise RuntimeError("Owned guard state is malformed")
    return state


def stop_state(state_path: Path) -> dict:
    state_path = Path(state_path).resolve()
    if state_path.parent != ROOT or state_path == ROOT:
        raise RuntimeError("State must live directly in the owned run root")
    if not state_path.exists():
        return {"status": "not-started", "group_signal_sent": False}
    state = load_state(state_path)
    pid = state["pid"]
    if start_ticks(pid) != state["start_ticks"]:
        return {"status": "already-exited-or-reused", "pid": pid,
                "group_signal_sent": False}
    actual = cmdline(pid)
    if actual is None:
        return {"status": "already-exited-or-reused", "pid": pid,
                "group_signal_sent": False}
    if actual != state["argv"]:
        if str(Path(__file__).resolve()) not in actual or str(state_path) not in actual:
            raise RuntimeError("Owned PID command identity changed; no signal sent")
        os.kill(pid, signal.SIGTERM)
        return {"status": "guard-stopped-before-exec", "pid": pid,
                "group_signal_sent": False}
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return {"status": "already-exited-or-reused", "pid": pid,
                "group_signal_sent": False}
    if group != pid:
        raise RuntimeError("Owned timeout process-group identity differs")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 2.0
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    if start_ticks(pid) == state["start_ticks"]:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"status": "owned-group-terminated", "pid": pid, "group_signal_sent": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    state = args.state.resolve()
    if state.parent != ROOT or state == ROOT:
        raise SystemExit("State must live directly in the owned run root")
    if args.stop:
        print(json.dumps(stop_state(state)))
        return 0
    command_argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if state.exists() or not state.parent.is_dir() or not 1 <= args.seconds <= 3600:
        raise SystemExit("Fresh state and a bounded positive deadline are required")
    if not command_argv:
        raise SystemExit("A nonempty guarded argv is required")
    command = [TIMEOUT, "--signal=TERM", "--kill-after=15s",
               f"{args.seconds}s"] + command_argv
    pid = os.getpid()
    if os.getpgrp() != pid:
        os.setsid()
    if os.getpgrp() != pid:
        raise SystemExit("Owned guard could not establish its process group")
    ticks = start_ticks(pid)
    if ticks is None:
        raise SystemExit("Owned guard identity is not observable")
    with state.open("x") as handle:
        handle.write(json.dumps({"pid": pid, "start_ticks": ticks,
                                 "argv": command}, indent=2) + "\n")
    os.execv(command[0], command)
    return 1


if __name__ == "__main__":
    main()
