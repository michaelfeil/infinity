#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build helper for the standalone routed-MoE ABI package.

The default mode is --check-only: it performs local file/config checks, prints
source hashes and the planned targets, and never invokes CUDA, the network or a
GPU query.  --build stages the source tree, compiles one standalone shared
library per requested architecture and writes a build manifest.

No source is downloaded and no device is queried by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = None
ABI_VERSION = 1

ARCH_GENCODE = {
    90: "arch=compute_90a,code=sm_90a",
    100: "arch=compute_100,code=sm_100",
    120: "arch=compute_120,code=sm_120",
}

ARCH_POLICY = {
    90: "portable SM80-CUTLASS path plus SM90-CUTLASS Hopper path",
    100: "portable SM80-CUTLASS path only; compile-only Native policy",
    120: "portable SM80-CUTLASS path only; compile-only Native policy",
}

SOURCE_FILES = [
    "include/oxide_moe.h",
    "src/oxide_moe_shim.cpp",
    "source/backends/libtorch/cpp/kernels/moe_symbols.h",
    "source/backends/libtorch/cpp/vendor/candle_kernels/moe/grouped_gemm.cu",
    "source/backends/libtorch/cpp/vendor/candle_kernels/moe/grouped_gemm_hopper.cu",
    "source/backends/libtorch/cpp/vendor/candle_kernels/kernels/gemma4_moe.cu",
    "source/backends/libtorch/cpp/vendor/candle_kernels/kernels/gemma4_moe_kernels.cuh",
]

LICENSE_FILES = [
    "LICENSE",
    "LICENSING.md",
    "THIRD_PARTY_NOTICES.md",
    "LICENSES/MIT.txt",
    "LICENSES/CUTLASS-BSD-3-Clause.txt",
]

TEST_FILES = [
    "tests/cpu_tests.c",
    "tests/run_cpu_tests.py",
    "tests/build_policy_tests.py",
]

STAGE_FILES = SOURCE_FILES + LICENSE_FILES + TEST_FILES + ["README.md", "build.py"]

REFERENCE_ONLY = [
    "source/backends/libtorch/cpp/kernels/moe_kernels.cpp",
]

EXPORTED_SYMBOLS = [
    "oxide_moe_abi_version",
    "oxide_moe_query_build",
    "oxide_moe_query_model",
    "oxide_moe_query_workspace",
    "oxide_moe_prepare",
    "oxide_moe_forward_bf16",
]

FORBIDDEN_DEPENDENCY_TOKENS = ("torch", "c10", "aten", "python")
FORBIDDEN_EXPORT_TOKENS = (
    "torch",
    "c10",
    "aten",
    "python",
    "gemma4_",
    "qwen3_",
    "qwen35_",
    "hopper_",
    "tei_",
)


class BuildError(RuntimeError):
    pass


def source_root() -> Path:
    if SOURCE_ROOT is not None:
        return SOURCE_ROOT
    try:
        candidate = ROOT.parents[2]
    except IndexError:
        candidate = None
    marker = "backends/libtorch/cpp/kernels/moe_symbols.h"
    if candidate is not None and (candidate / marker).is_file():
        return candidate
    raise BuildError(
        "public source root is outside the checkout; pass "
        "--source-root <path-to-libs/infinity-v3>"
    )


def resolve(rel: str) -> Path:
    if rel.startswith("source/"):
        return source_root() / rel[len("source/"):]
    return ROOT / rel


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: list[str]) -> dict[str, str]:
    return {rel: sha256_file(resolve(rel)) for rel in sorted(paths)}


def parse_arches(values: list[str]) -> list[int]:
    result: list[int] = []
    if not values:
        return result
    for value in values:
        for part in re.split(r"[,/]", value):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                for arch in (90, 100, 120):
                    if arch not in result:
                        result.append(arch)
                continue
            try:
                arch = int(part)
            except ValueError as exc:
                raise BuildError(f"invalid --arch value: {part!r}") from exc
            if arch not in ARCH_GENCODE:
                raise BuildError(f"unsupported --arch value: {arch}")
            if arch not in result:
                result.append(arch)
    return result


def inspect_cutlass(root: Path) -> dict[str, object]:
    include = root / "include"
    cutlass_header = include / "cutlass" / "cutlass.h"
    if not cutlass_header.is_file():
        raise BuildError(f"CUTLASS root does not contain include/cutlass/cutlass.h: {root}")
    info: dict[str, object] = {
        "root": str(root.resolve()),
        "include": str(include.resolve()),
        "cutlass_header_sha256": sha256_file(cutlass_header),
        "version_header_sha256": None,
        "version": None,
        "license_file": None,
        "license_sha256": None,
        "license_path": None,
    }
    for license_name in ("LICENSE.txt", "LICENSE", "LICENSE.md"):
        license_candidate = root / license_name
        if license_candidate.is_file():
            info["license_file"] = license_name
            info["license_sha256"] = sha256_file(license_candidate)
            info["license_path"] = str(license_candidate.resolve())
            break
    version_header = include / "cutlass" / "version.h"
    if version_header.is_file():
        info["version_header_sha256"] = sha256_file(version_header)
        text = version_header.read_text(encoding="utf-8", errors="replace")
        major = re.search(r"#define\s+CUTLASS_MAJOR\s+(\d+)", text)
        minor = re.search(r"#define\s+CUTLASS_MINOR\s+(\d+)", text)
        patch = re.search(r"#define\s+CUTLASS_PATCH\s+(\d+)", text)
        if major and minor and patch:
            info["version"] = f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"
    return info


def fresh_output(path: Path, create: bool = True) -> None:
    if path.exists():
        if not path.is_dir():
            raise BuildError(f"--output is not a directory: {path}")
        if any(path.iterdir()):
            raise BuildError(f"--output must be fresh or empty: {path}")
    elif create:
        path.mkdir(parents=True)


def stage_sources(output: Path) -> Path:
    staged = output / "staged"
    for rel in STAGE_FILES:
        src = resolve(rel)
        if not src.is_file():
            raise BuildError(f"required package file is missing: {rel}")
        dst = staged / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return staged


def version_script() -> str:
    lines = ["{", "  global:"]
    lines.extend(f"    {name};" for name in EXPORTED_SYMBOLS)
    lines.extend(["  local:", "    *;", "};", ""])
    return "\n".join(lines)


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def compiler_path(name: str, label: str) -> str:
    found = shutil.which(name)
    if not found:
        raise BuildError(f"{label} not found on PATH: {name}")
    return found


def run_capture(cmd: list[str]) -> str:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return completed.stdout


def verify_elf(library: Path, readelf: str) -> dict[str, object]:
    needed_output = run_capture([readelf, "-d", str(library)])
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", needed_output)
    forbidden_needed = [
        name for name in needed if any(token in name.lower() for token in FORBIDDEN_DEPENDENCY_TOKENS)
    ]
    if forbidden_needed:
        raise BuildError(
            "forbidden shared-library dependency in "
            f"{library.name}: {', '.join(forbidden_needed)}"
        )

    symbol_output = run_capture([readelf, "-W", "--dyn-syms", str(library)])
    defined: set[str] = set()
    for line in symbol_output.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] == "Num:":
            continue
        name = parts[7].split("@", 1)[0]
        ndx = parts[6]
        if name and ndx != "UND":
            defined.add(name)

    missing = [name for name in EXPORTED_SYMBOLS if name not in defined]
    if missing:
        raise BuildError(f"missing required ABI exports in {library.name}: {', '.join(missing)}")

    forbidden_exports = [
        name
        for name in defined
        if any(token in name.lower() for token in FORBIDDEN_EXPORT_TOKENS)
    ]
    if forbidden_exports:
        raise BuildError(
            f"forbidden raw/Torch-style exports in {library.name}: "
            + ", ".join(sorted(forbidden_exports))
        )

    unexpected = sorted(name for name in defined if not name.startswith("oxide_moe_"))
    if unexpected:
        raise BuildError(
            f"unexpected non-ABI exports in {library.name}: " + ", ".join(unexpected)
        )
    return {
        "needed": sorted(needed),
        "forbidden_needed": forbidden_needed,
        "required_exports_present": EXPORTED_SYMBOLS,
        "unexpected_non_abi_exports": unexpected,
        "readelf_commands": [
            [readelf, "-d", str(library)],
            [readelf, "-W", "--dyn-syms", str(library)],
        ],
    }


def build_arch(
    arch: int,
    staged: Path,
    output: Path,
    cutlass_info: dict[str, object],
    nvcc: str,
    cxx: str,
    readelf: str,
    script: Path,
) -> dict[str, object]:
    gencode = ARCH_GENCODE[arch]
    obj_dir = output / "obj" / f"sm{arch}"
    obj_dir.mkdir(parents=True, exist_ok=True)

    cutlass_include = Path(str(cutlass_info["include"]))
    kernels_dir = staged / "source/backends/libtorch/cpp/vendor/candle_kernels"
    moe_dir = kernels_dir / "moe"
    header_dir = staged / "source/backends/libtorch/cpp/kernels"

    shim_obj = obj_dir / "oxide_moe_shim.o"
    shim_cmd = [
        cxx,
        "-std=c++17",
        "-Wall", "-Wextra", "-Werror",
        "-O3",
        "-fPIC",
        "-fvisibility=hidden",
        "-I",
        str(staged / "include"),
        "-I",
        str(header_dir),
        f"-DOXIDE_MOE_BUILD_ARCH={arch}",
        "-c",
        str(staged / "src/oxide_moe_shim.cpp"),
        "-o",
        str(shim_obj),
    ]
    run(shim_cmd)

    cuda_sources = [moe_dir / "grouped_gemm.cu"]
    if arch == 90:
        cuda_sources.append(moe_dir / "grouped_gemm_hopper.cu")

    cuda_objects: list[Path] = []
    commands: list[list[str]] = [[str(part) for part in shim_cmd]]
    for source in cuda_sources:
        obj = obj_dir / f"{source.stem}.o"
        cmd = [
            nvcc,
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-O3",
            "-c",
            "-Xcompiler",
            "-fPIC,-fvisibility=hidden",
            "-gencode",
            gencode,
            "-I",
            str(cutlass_include),
            "-I",
            str(Path(str(cutlass_info["root"])) / "tools/util/include"),
            "-I",
            str(kernels_dir),
            "-I",
            str(moe_dir),
            "-I",
            str(kernels_dir / "kernels"),
            "-I",
            str(header_dir),
            "-include",
            str(header_dir / "moe_symbols.h"),
            f"-DOXIDE_MOE_BUILD_ARCH={arch}",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        run(cmd)
        commands.append([str(part) for part in cmd])
        cuda_objects.append(obj)

    library = output / f"liboxide_moe_sm{arch}.so"
    link_cmd = [
        nvcc,
        "-shared",
        str(shim_obj),
        *[str(obj) for obj in cuda_objects],
        "-Xlinker",
        f"--version-script={script}",
        "-o",
        str(library),
    ]
    run(link_cmd)
    commands.append([str(part) for part in link_cmd])

    verification = verify_elf(library, readelf)
    commands.extend(verification["readelf_commands"])
    return {
        "arch": arch,
        "gencode": gencode,
        "policy": ARCH_POLICY[arch],
        "library": library.name,
        "library_sha256": sha256_file(library),
        "commands": commands,
        "verification": verification,
    }


def verify_provenance(source_hashes: dict[str, str]) -> dict[str, object]:
    provenance_path = ROOT / "PROVENANCE.json"
    if not provenance_path.is_file():
        raise BuildError("PROVENANCE.json is required for reproducible sidecar builds")
    provenance = json.loads(provenance_path.read_text())
    rows = provenance.get("sources", {})
    if not isinstance(rows, dict) or not rows:
        raise BuildError("PROVENANCE.json has no source rows")
    for rel, row in rows.items():
        expected = row.get("package_sha256")
        actual = source_hashes.get(rel)
        if not isinstance(expected, str) or actual != expected:
            raise BuildError(f"provenance source hash mismatch: {rel}")
        if "matches_qualified" in row and row["matches_qualified"] is not True:
            raise BuildError(f"source is not qualified: {rel}")
        if "matches_public_repo" in row and row["matches_public_repo"] is not True:
            raise BuildError(f"source is not byte-identical to public repo: {rel}")
    if provenance.get("qualified_sidecar_build_manifest_sha256") is None:
        raise BuildError("provenance lacks qualified sidecar manifest pin")
    return provenance


def check_only(args: argparse.Namespace, arches: list[int]) -> int:
    missing = [rel for rel in STAGE_FILES if not resolve(rel).is_file()]
    if missing:
        raise BuildError("missing package files: " + ", ".join(missing))
    if args.output:
        fresh_output(args.output, create=False)
    source_hashes = hash_files(SOURCE_FILES)
    provenance = verify_provenance(source_hashes)
    license_hashes = hash_files(LICENSE_FILES)
    test_and_doc_hashes = hash_files(TEST_FILES + ["README.md", "build.py"])
    cutlass_info = inspect_cutlass(args.cutlass_root) if args.cutlass_root else None
    summary = {
        "mode": "check-only",
        "abi_version": ABI_VERSION,
        "planned_arches": [
            {"arch": arch, "gencode": ARCH_GENCODE[arch], "policy": ARCH_POLICY[arch]}
            for arch in arches
        ],
        "source_hashes": source_hashes,
        "provenance_sha256": sha256_file(ROOT / "PROVENANCE.json"),
        "provenance_verified": True,
        "package": provenance.get("package"),
        "license_hashes": license_hashes,
        "test_and_doc_hashes": test_and_doc_hashes,
        "cutlass": cutlass_info,
        "reference_only_not_compiled": REFERENCE_ONLY,
        "exported_symbols": EXPORTED_SYMBOLS,
        "network_access": False,
        "gpu_query": False,
        "cuda_invoked": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build(args: argparse.Namespace, arches: list[int]) -> int:
    if not args.cutlass_root:
        raise BuildError("--build requires --cutlass-root pointing to a pinned public CUTLASS tree")
    if not args.output:
        raise BuildError("--build requires --output pointing to a fresh directory")
    cutlass_info = inspect_cutlass(args.cutlass_root)
    nvcc = compiler_path(args.nvcc, "CUDA compiler")
    cxx = compiler_path(args.cxx, "host C++ compiler")
    readelf = compiler_path("readelf", "ELF reader")

    fresh_output(args.output)
    if cutlass_info.get("license_path"):
        upstream_license = args.output / "LICENSES" / "CUTLASS-UPSTREAM-LICENSE.txt"
        upstream_license.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(str(cutlass_info["license_path"])), upstream_license)
        cutlass_info["staged_license"] = "LICENSES/CUTLASS-UPSTREAM-LICENSE.txt"
        cutlass_info["staged_license_sha256"] = sha256_file(upstream_license)
    staged = stage_sources(args.output)
    source_hashes = hash_files(SOURCE_FILES)
    provenance = verify_provenance(source_hashes)
    license_hashes = hash_files(LICENSE_FILES)
    test_hashes = hash_files(TEST_FILES)
    staged_hashes = {rel: sha256_file(staged / rel) for rel in sorted(STAGE_FILES)}
    source_manifest = {
        "package": "oxide-moe-standalone",
        "abi_version": ABI_VERSION,
        "source_hashes": source_hashes,
        "license_files": LICENSE_FILES,
        "license_hashes": license_hashes,
        "test_hashes": test_hashes,
        "staged_files": STAGE_FILES,
        "staged_hashes": staged_hashes,
        "cutlass": cutlass_info,
        "reference_only_not_compiled": REFERENCE_ONLY,
        "arch_policy": {str(arch): ARCH_POLICY[arch] for arch in arches},
    }
    (args.output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    script = args.output / "oxide_moe_exports.map"
    script.write_text(version_script(), encoding="utf-8")

    targets = []
    for arch in arches:
        targets.append(
            build_arch(
                arch,
                staged,
                args.output,
                cutlass_info,
                nvcc,
                cxx,
                readelf,
                script,
            )
        )

    build_manifest = {
        "package": "oxide-moe-standalone",
        "abi_version": ABI_VERSION,
        "source_manifest": "SOURCE_MANIFEST.json",
        "source_manifest_sha256": sha256_file(args.output / "SOURCE_MANIFEST.json"),
        "build_args": {
            "arches": arches,
            "cutlass_root": str(args.cutlass_root.resolve()),
            "output": str(args.output.resolve()),
            "nvcc": nvcc,
            "cxx": cxx,
            "readelf": readelf,
        },
        "targets": targets,
        "exported_symbols": EXPORTED_SYMBOLS,
        "network_access": False,
        "gpu_query": False,
        "notes": [
            "No source was downloaded.",
            "No GPU or device property was queried.",
            "SM100/SM120 are built from the portable SM80-CUTLASS kernels only.",
            "SM100/SM120 are compile-only until real-GPU qualification by Root.",
            "Kernel and source math were not modified.",
            "Produced ELF dependencies and required exports were checked with readelf.",
            "No Torch/ATen/c10 dependency or raw operator export is permitted.",
        ],
    }
    (args.output / "BUILD_MANIFEST.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mode": "build", "output": str(args.output), "targets": targets}, indent=2))
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="checks only (default)")
    parser.add_argument("--source-root", type=Path, default=None,
                        help="public repository/backend source root for source/ paths")
    parser.add_argument("--build", action="store_true", help="stage and compile shared libraries")
    parser.add_argument("--cutlass-root", type=Path, help="pinned public CUTLASS tree")
    parser.add_argument("--output", type=Path, help="fresh output directory for --build")
    parser.add_argument(
        "--arch",
        action="append",
        default=[],
        help="target architecture: 90, 100, 120 or all (repeatable/comma-separated)",
    )
    parser.add_argument(
        "--nvcc",
        default=os.environ.get("CUDACXX", "nvcc"),
        help="CUDA compiler (default: $CUDACXX or nvcc)",
    )
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX", "c++"),
        help="host C++ compiler (default: $CXX or c++)",
    )
    args = parser.parse_args(argv)
    global SOURCE_ROOT
    SOURCE_ROOT = args.source_root.resolve() if args.source_root else None

    try:
        if args.check_only and args.build:
            raise BuildError("--check-only and --build are mutually exclusive")
        if not args.build:
            args.check_only = True
        arches = parse_arches(args.arch)
        if not arches:
            if args.build:
                raise BuildError("--build requires at least one --arch (90, 100, 120 or all)")
            arches = [90, 100, 120]
        if args.check_only:
            return check_only(args, arches)
        return build(args, arches)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
