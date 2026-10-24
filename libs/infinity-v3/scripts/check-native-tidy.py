#!/usr/bin/env python3
"""Check maintained C++ translation units using their CMake compile commands."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_directory", type=Path)
    parser.add_argument("--clang-tidy", default="clang-tidy")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "backends/libtorch/cpp"
    database = args.build_directory.resolve()
    commands = json.loads((database / "compile_commands.json").read_text())
    files = set()
    for command in commands:
        path = (Path(command["directory"]) / command["file"]).resolve()
        if path.suffix != ".cpp" or not path.is_relative_to(source):
            continue
        if {"vendor", "fixtures", "snapshots"}.intersection(path.relative_to(source).parts):
            continue
        files.add(path)
    if not files:
        parser.error("compilation database contains no maintained C++ translation units")

    def check(path):
        result = subprocess.run(
            [
                args.clang_tidy, str(path), "-p", str(database), "--quiet",
                "--header-filter=(^|/)backends/libtorch/cpp/",
                "--exclude-header-filter=(^|/)(vendor|fixtures|snapshots)/",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        return path, result

    failed = False
    with ThreadPoolExecutor(max_workers=2) as pool:
        for path, result in pool.map(check, sorted(files)):
            print(f"Checking {path.relative_to(source)}", flush=True)
            print(result.stdout, end="", flush=True)
            failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
