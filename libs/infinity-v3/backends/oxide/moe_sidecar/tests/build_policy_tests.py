#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU-only checks for build.py policy and manifest/check-only behavior.

These checks never invoke nvcc, initialize CUDA, query a device or use the
network. They exercise the explicit check-only default, fresh-output refusal,
missing-CUTLASS refusal, and the CPU-side JSON summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build.py"
DEFAULT_SOURCE = ROOT.parents[2]
MARKER = Path("backends/libtorch/cpp/kernels/moe_symbols.h")
SOURCE_FOR_TEST = os.environ.get("OXIDE_MOE_SOURCE_ROOT")
if SOURCE_FOR_TEST is None and (DEFAULT_SOURCE / MARKER).is_file():
    SOURCE_FOR_TEST = str(DEFAULT_SOURCE)
SOURCE_ARGS = ["--source-root", SOURCE_FOR_TEST] if SOURCE_FOR_TEST else []


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUILD), *SOURCE_ARGS, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    failures = 0

    def check(condition: bool, message: str) -> None:
        nonlocal failures
        if not condition:
            print(f"FAIL: {message}", file=sys.stderr)
            failures += 1

    result = run("--check-only")
    check(result.returncode == 0, "--check-only succeeds")
    if result.returncode == 0:
        summary = json.loads(result.stdout)
        check(summary.get("mode") == "check-only", "check-only mode reported")
        check(summary.get("network_access") is False, "network disabled")
        check(summary.get("gpu_query") is False, "GPU query disabled")
        check(summary.get("cuda_invoked") is False, "CUDA not invoked")
        check(len(summary.get("source_hashes", {})) >= 7, "source hashes recorded")
        check(len(summary.get("planned_arches", [])) == 3, "all arches planned")

    with tempfile.TemporaryDirectory(prefix="oxide-policy-") as temp:
        temp_path = Path(temp)
        empty = temp_path / "empty"
        empty.mkdir()
        result = run("--check-only", "--output", str(empty))
        check(result.returncode == 0, "empty output directory accepted")

        nonempty = temp_path / "nonempty"
        nonempty.mkdir()
        (nonempty / "stale.txt").write_text("stale\n", encoding="utf-8")
        result = run("--check-only", "--output", str(nonempty))
        check(result.returncode != 0, "non-empty output directory rejected")

        result = run("--build", "--arch", "90")
        check(result.returncode != 0, "build without CUTLASS/output rejected")
        check("--cutlass-root" in result.stderr, "missing CUTLASS is reported")

        fake_cutlass = temp_path / "cutlass"
        fake_cutlass.mkdir()
        output = temp_path / "out"
        result = run(
            "--build",
            "--cutlass-root",
            str(fake_cutlass),
            "--output",
            str(output),
            "--arch",
            "90",
        )
        check(result.returncode != 0, "invalid CUTLASS root rejected")
        check("cutlass.h" in result.stderr, "missing CUTLASS header reported")
        check(not output.exists(), "no output tree created before validation")

        result = run("--build", "--arch", "99")
        check(result.returncode != 0, "unsupported arch rejected")
        check("unsupported --arch" in result.stderr, "unsupported arch reported")

    if failures:
        print(f"{failures} build policy test failure(s)", file=sys.stderr)
        return 1
    print("build policy tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
