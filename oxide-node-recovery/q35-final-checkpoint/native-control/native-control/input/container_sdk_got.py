#!/usr/bin/env python3
"""Container-only actual b66 SDK load, header ABI, and resolved GOT proof.

The proof follows the already-qualified CPU-only relocation method from
/tmp/oxide-dense-native-e582-v2/verify_bindings.py: map the loaded ELF, read
dynamic relocation entries with readelf, dereference each cublas GOT slot in
this process, and resolve the target with dladdr. A dependency listing or
readelf output alone is never treated as the actual in-memory GOT.

This helper must run under the SDK container Python with LD_BIND_NOW=1, the
130200 vendor libraries on LD_PRELOAD, and CUDA_VISIBLE_DEVICES=''. It imports
Torch only inside the container. It never calls tei_create/tei_ready/forward,
never allocates an engine, and never creates a CUDA context.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_SDK_SHA256 = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
DEFAULT_VENDOR = Path("/tmp/oxide-host-cublas-130200")
DEFAULT_TORCH_PREFIX = "2.14.1+cu130"
DEFAULT_TORCH_GIT_PREFIX = "5c488"
DEFAULT_CUBLASLT_VERSION = 130200
DEFAULT_TEI_FORWARD_ARGS = 19
C_API_SYMBOLS = [
    "tei_error",
    "tei_last_error_kind",
    "tei_device_preflight",
    "tei_create",
    "tei_weight",
    "tei_load_safetensors",
    "tei_option",
    "tei_ready",
    "tei_forward",
    "tei_forward_submit",
    "tei_output_wait",
    "tei_output_destroy",
    "tei_graph_count",
    "tei_graph_stats",
    "tei_destroy",
]


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_zero_offset_base(obj: Path) -> int:
    obj = obj.resolve()
    matches = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        target = parts[5]
        try:
            same = Path(target).resolve() == obj
        except OSError:
            same = False
        if not same:
            continue
        if int(parts[2], 16) == 0:
            matches.append(int(parts[0].split("-", 1)[0], 16))
    if len(matches) != 1:
        raise RuntimeError(f"expected one zero-offset mapping for {obj}, got {matches}")
    return matches[0]


def load_bias(obj: Path) -> int:
    return mapped_zero_offset_base(obj)


def _dladdr():
    dl = ctypes.CDLL(None)
    dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dl.dladdr.restype = ctypes.c_int
    return dl


def resolved_relocations(obj: Path, symbol_prefix: str) -> list[dict]:
    """Dereference real GOT slots and resolve each target through dladdr."""
    base = load_bias(obj)
    text = subprocess.run(
        ["readelf", "-rW", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dl = _dladdr()
    result = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        kind = parts[2]
        if kind not in ("R_X86_64_JUMP_SLOT", "R_X86_64_GLOB_DAT"):
            continue
        symbol = parts[4].split("@", 1)[0]
        if not symbol.startswith(symbol_prefix):
            continue
        offset = int(parts[0], 16)
        pointer = ctypes.c_void_p.from_address(base + offset).value
        info = DlInfo()
        found = dl.dladdr(ctypes.c_void_p(pointer), ctypes.byref(info)) if pointer else 0
        path = info.dli_fname.decode() if found and info.dli_fname else None
        result.append(
            {
                "relocation_offset": f"0x{offset:x}",
                "relocation_type": kind,
                "symbol": symbol,
                "pointer_nonzero": bool(pointer),
                "resolved_path": path,
                "resolved_realpath": str(Path(path).resolve()) if path else None,
            }
        )
    if not result:
        raise RuntimeError(f"no {symbol_prefix}* GOT relocations found in {obj}")
    return result


def bridge_arg_count(path: Path, symbol: str) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text()
    match = re.search(rf"\b(?:int32_t|void\s*\*)\s+{re.escape(symbol)}\s*\(", text)
    if not match:
        return None
    start = match.end()
    depth = 0
    count = 0
    token = False
    body = text[start:]
    body = body[: body.find(");")]
    for char in body:
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth -= 1
        elif char == "," and depth == 0:
            if token:
                count += 1
            token = False
        elif not char.isspace():
            token = True
    if token:
        count += 1
    return count


def runner_needed(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = subprocess.run(
        ["readelf", "-dW", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    needed = []
    for line in text.splitlines():
        match = re.search(r"\(NEEDED\).*\[(.*)\]", line)
        if match:
            needed.append(match.group(1))
    return needed


def vendor_record(vendor: Path, expected_version: int) -> tuple[dict, list[str]]:
    errors = []
    libraries = {}
    for name in ("libcublasLt.so.13", "libcublas.so.13"):
        path = vendor / name
        if not path.is_file():
            errors.append(f"vendor library missing: {path}")
            libraries[name] = {"path": str(path), "exists": False}
            continue
        libraries[name] = {
            "path": str(path),
            "realpath": str(path.resolve()),
            "exists": True,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    version = None
    lt = vendor / "libcublasLt.so.13"
    if lt.is_file():
        try:
            lib = ctypes.CDLL(str(lt), mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
            lib.cublasLtGetVersion.restype = ctypes.c_size_t
            version = int(lib.cublasLtGetVersion())
        except Exception as error:
            errors.append(f"cublasLtGetVersion failed: {error}")
    if version != expected_version:
        errors.append(f"cublasLt version {version} != {expected_version}")
    record = {
        "path": str(vendor),
        "libraries": libraries,
        "cublaslt_version": version,
        "expected_cublaslt_version": expected_version,
    }
    return record, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--expected-sdk-sha", default=DEFAULT_SDK_SHA256)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    parser.add_argument("--expected-cublaslt-version", type=int, default=DEFAULT_CUBLASLT_VERSION)
    parser.add_argument("--bridge-header", type=Path, required=True)
    parser.add_argument("--expected-tei-forward-args", type=int, default=DEFAULT_TEI_FORWARD_ARGS)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-torch-prefix", default=DEFAULT_TORCH_PREFIX)
    parser.add_argument("--expected-torch-git-prefix", default=DEFAULT_TORCH_GIT_PREFIX)
    parser.add_argument("--allow-cuda-visible-devices", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd != args.allow_cuda_visible_devices:
        errors.append(
            f"CUDA_VISIBLE_DEVICES={cvd!r}; expected {args.allow_cuda_visible_devices!r}"
        )
    if os.environ.get("LD_BIND_NOW") != "1":
        errors.append("LD_BIND_NOW=1 is required")

    torch_version = None
    torch_git_version = None
    torch_cuda_before = None
    torch_cuda_after = None
    try:
        import torch  # container-only

        torch_version = torch.__version__
        torch_git_version = getattr(torch.version, "git_version", None)
        torch_cuda_before = bool(torch.cuda.is_initialized())
        if torch_cuda_before:
            errors.append("Torch CUDA was initialized before the CPU SDK proof")
        if not torch_version.startswith(args.expected_torch_prefix):
            errors.append(
                f"Torch version {torch_version!r} does not start with {args.expected_torch_prefix!r}"
            )
        if args.expected_torch_git_prefix and (
            not torch_git_version or not torch_git_version.startswith(args.expected_torch_git_prefix)
        ):
            errors.append(
                f"Torch git version {torch_git_version!r} does not start with "
                f"{args.expected_torch_git_prefix!r}"
            )
    except Exception as error:
        errors.append(f"Torch import failed: {error}")

    sdk_sha = sha256_file(args.sdk) if args.sdk.is_file() else None
    if not args.sdk.is_file():
        errors.append(f"SDK is missing: {args.sdk}")
    elif sdk_sha != args.expected_sdk_sha:
        errors.append(f"SDK sha256 {sdk_sha} != {args.expected_sdk_sha}")

    header_sha = sha256_file(args.bridge_header) if args.bridge_header.is_file() else None
    forward_args = bridge_arg_count(args.bridge_header, "tei_forward")
    submit_args = bridge_arg_count(args.bridge_header, "tei_forward_submit")
    if forward_args != args.expected_tei_forward_args:
        errors.append(
            f"bridge.h tei_forward argument count {forward_args} != {args.expected_tei_forward_args}"
        )

    sdk = None
    missing_symbols = []
    got_targets: list[dict] = []
    if args.sdk.is_file():
        try:
            sdk = ctypes.CDLL(str(args.sdk), mode=os.RTLD_NOW | ctypes.RTLD_GLOBAL)
        except Exception as error:
            errors.append(f"actual SDK load failed: {error}")
    if sdk is not None:
        for symbol in C_API_SYMBOLS:
            try:
                getattr(sdk, symbol)
            except Exception:
                missing_symbols.append(symbol)
        if missing_symbols:
            errors.append(f"SDK missing C ABI symbols: {missing_symbols}")
        try:
            got_targets = resolved_relocations(args.sdk.resolve(), "cublas")
        except Exception as error:
            errors.append(f"SDK GOT resolution failed: {error}")
    vendor_path = args.vendor.resolve()
    for target in got_targets:
        resolved = target.get("resolved_realpath")
        target["vendor_130200"] = bool(
            resolved and Path(resolved).parent == vendor_path
        )
        if not target["pointer_nonzero"]:
            errors.append(f"null GOT slot for {target['symbol']}")
        if not target["vendor_130200"]:
            errors.append(f"cublas GOT target outside vendor: {target}")
    if sdk is not None and not got_targets:
        errors.append("SDK has no resolved cublas GOT targets")

    vendor, vendor_errors = vendor_record(args.vendor, args.expected_cublaslt_version)
    errors.extend(vendor_errors)

    runner = None
    if args.runner:
        runner = {
            "path": str(args.runner),
            "exists": args.runner.is_file(),
            "sha256": sha256_file(args.runner) if args.runner.is_file() else None,
            "needed": runner_needed(args.runner),
            "needed_tei_torch": "libtei_torch.so" in runner_needed(args.runner),
        }
        if not runner["exists"]:
            errors.append(f"runner is missing: {args.runner}")
        elif not runner["needed_tei_torch"]:
            errors.append("runner is not linked to libtei_torch.so")

    try:
        import torch  # container-only

        torch_cuda_after = bool(torch.cuda.is_initialized())
    except Exception:
        torch_cuda_after = None
    if torch_cuda_after:
        errors.append("Torch CUDA was initialized during the CPU SDK proof")

    report = {
        "schema": "oxide-native-b66-container-sdk-got/v7",
        "sdk": str(args.sdk),
        "sdk_sha256": sdk_sha,
        "expected_sdk_sha256": args.expected_sdk_sha,
        "sdk_sha256_matches": sdk_sha == args.expected_sdk_sha,
        "sdk_loaded": sdk is not None,
        "sdk_load_mode": "os.RTLD_NOW|ctypes.RTLD_GLOBAL",
        "tei_forward_symbol": bool(sdk is not None and not missing_symbols),
        "missing_c_api_symbols": missing_symbols,
        "bridge_header": str(args.bridge_header),
        "bridge_header_sha256": header_sha,
        "tei_forward_args": forward_args,
        "expected_tei_forward_args": args.expected_tei_forward_args,
        "tei_forward_submit_args": submit_args,
        "runner": runner,
        "cublas_got_targets": got_targets,
        "cublas_got_target_count": len(got_targets),
        "vendor": vendor,
        "torch_version": torch_version,
        "torch_git_version": torch_git_version,
        "expected_torch_prefix": args.expected_torch_prefix,
        "torch_cuda_initialized_before": torch_cuda_before,
        "torch_cuda_initialized_after": torch_cuda_after,
        "cuda_visible_devices": cvd,
        "cuda_context_created": False,
        "engine_created": False,
        "engine_or_forward_called": False,
        "ld_bind_now": os.environ.get("LD_BIND_NOW"),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "errors": errors,
        "ok": not errors,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
