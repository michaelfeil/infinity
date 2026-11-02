#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile and run the CPU-only ABI tests against built shared libraries.

This runner does not import CUDA/PyTorch, query a GPU, or access the network.
It only invokes the host C compiler and the test executable.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="directory containing liboxide_moe_sm{90,100,120}.so",
    )
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        help="restrict to an architecture (repeatable/comma-separated)",
    )
    parser.add_argument(
        "--library",
        action="append",
        default=[],
        type=Path,
        help="explicit library path (repeatable; overrides discovery)",
    )
    parser.add_argument(
        "--cc",
        default=os.environ.get("CC", "cc"),
        help="host C compiler (default: $CC or cc)",
    )
    return parser.parse_args(argv)


def wanted_arches(values: list[str]) -> set[int]:
    result: set[int] = set()
    for value in values:
        for part in value.replace("/", ",").split(","):
            part = part.strip()
            if part:
                result.add(int(part))
    return result


def discover(build_dir: Path, arches: set[int]) -> list[Path]:
    libraries = []
    for path in sorted(build_dir.glob("liboxide_moe_sm*.so")):
        for arch in (90, 100, 120):
            if path.name == f"liboxide_moe_sm{arch}.so" and (not arches or arch in arches):
                libraries.append(path)
    return libraries


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    arches = wanted_arches(args.arch)
    libraries = [path.resolve() for path in args.library]
    if not libraries:
        libraries = discover(args.build_dir.resolve(), arches)
    if not libraries:
        print("error: no built liboxide_moe_sm*.so libraries found", file=sys.stderr)
        return 2

    cc = shutil.which(args.cc)
    if not cc:
        print(f"error: C compiler not found: {args.cc}", file=sys.stderr)
        return 2

    exe = args.build_dir.resolve() / "oxide_moe_cpu_tests"
    compile_cmd = [
        cc,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "include"),
        str(ROOT / "tests" / "cpu_tests.c"),
        "-ldl",
        "-o",
        str(exe),
    ]
    print("+ " + " ".join(compile_cmd), flush=True)
    subprocess.run(compile_cmd, check=True)
    run_cmd = [str(exe), *[str(path) for path in libraries]]
    print("+ " + " ".join(run_cmd), flush=True)
    return subprocess.run(run_cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
