#!/usr/bin/env python3
"""Container-side actual b66 SDK load and vendor GOT proof.

Run inside the SDK container with LD_BIND_NOW=1 and the vendor libraries on
LD_PRELOAD/LD_LIBRARY_PATH. It imports Torch only inside the container, loads
the actual SDK with ctypes (RTLD_NOW), resolves every cublas relocation through
dladdr to /tmp/oxide-host-cublas-130200, loads libcublasLt.so.13 and checks
130200, and parses bridge.h's 19-argument tei_forward declaration. It never
creates an engine or CUDA context.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, re, subprocess, sys
from pathlib import Path

EXPECTED_SHA = "b66d70b61b39e1bf0d565aca33bac7c5e21849677396ce5216f5baf73027d37e"
VENDOR = Path("/tmp/oxide-host-cublas-130200")
EXPECTED_VERSION = 130200

class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_bias(obj: Path) -> int:
    matches = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        target = parts[-1] if parts else ""
        if target != str(obj):
            continue
        start = int(parts[0].split("-")[0], 16)
        offset = int(parts[2], 16)
        if offset == 0:
            matches.append(start)
    if len(matches) != 1:
        raise RuntimeError(f"expected one zero-offset mapping for {obj}, got {matches}")
    return matches[0]

def resolved_cublas_targets(obj: Path) -> list[dict]:
    base = load_bias(obj)
    text = subprocess.run(["readelf", "-rW", str(obj)], check=True, capture_output=True, text=True).stdout
    dl = ctypes.CDLL(None)
    dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dl.dladdr.restype = ctypes.c_int
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[2] not in ("R_X86_64_JUMP_SLOT", "R_X86_64_GLOB_DAT"):
            continue
        symbol = parts[4].split("@")[0]
        if not symbol.startswith("cublas"):
            continue
        pointer = ctypes.c_void_p.from_address(base + int(parts[0], 16)).value
        info = DlInfo()
        found = dl.dladdr(ctypes.c_void_p(pointer), ctypes.byref(info))
        path = info.dli_fname.decode() if found and info.dli_fname else None
        out.append({"symbol": symbol, "resolved_path": path, "vendor": bool(path and path.startswith(str(VENDOR) + "/"))})
    return out

def bridge_forward_args(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text()
    start = text.find("int32_t tei_forward(")
    if start < 0:
        return None
    end = text.find(");", start)
    if end < 0:
        return None
    signature = text[start:end]
    body = signature.split("(", 1)[1]
    # Count top-level commas; the declaration contains no function-pointer args.
    depth = 0
    count = 1 if body.strip() else 0
    for char in body:
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth -= 1
        elif char == "," and depth == 0:
            count += 1
    return count

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdk", type=Path, required=True)
    ap.add_argument("--runner", type=Path)
    ap.add_argument("--expected-sha", default=EXPECTED_SHA)
    ap.add_argument("--vendor", type=Path, default=VENDOR)
    ap.add_argument("--expected-version", type=int, default=EXPECTED_VERSION)
    ap.add_argument("--bridge-header", type=Path)
    a = ap.parse_args()
    errors = []
    if os.environ.get("LD_BIND_NOW") != "1":
        errors.append("LD_BIND_NOW=1 is required")
    if not a.sdk.is_file():
        errors.append(f"SDK is missing: {a.sdk}")
    sdk_sha = sha(a.sdk) if a.sdk.is_file() else None
    if sdk_sha and sdk_sha != a.expected_sha:
        errors.append(f"SDK sha256 {sdk_sha} != {a.expected_sha}")
    torch_version = None
    cuda_initialized = None
    try:
        import torch  # container-only
        torch_version = torch.__version__
        cuda_initialized = torch.cuda.is_initialized()
        if cuda_initialized:
            errors.append("Torch CUDA was initialized during CPU SDK validation")
    except Exception as error:
        errors.append(f"Torch import failed: {error}")
    sdk = None
    if a.sdk.is_file():
        try:
            sdk = ctypes.CDLL(str(a.sdk), mode=ctypes.RTLD_NOW | ctypes.RTLD_GLOBAL)
        except Exception as error:
            errors.append(f"actual SDK load failed: {error}")
    tei_forward_symbol = False
    if sdk is not None:
        try:
            tei_forward_symbol = bool(sdk.tei_forward)
        except Exception:
            errors.append("SDK does not export tei_forward")
    header_args = bridge_forward_args(a.bridge_header) if a.bridge_header else None
    if header_args != 19:
        errors.append(f"bridge.h tei_forward argument count {header_args} != 19")
    targets = []
    if sdk is not None:
        try:
            targets = resolved_cublas_targets(a.sdk.resolve())
        except Exception as error:
            errors.append(f"SDK GOT resolution failed: {error}")
    if not targets:
        errors.append("SDK has no resolved cublas GOT targets")
    for target in targets:
        if not target["vendor"]:
            errors.append(f"cublas GOT target outside 130200 vendor: {target}")
    version = None
    vendor_lt = a.vendor / "libcublasLt.so.13"
    if not vendor_lt.is_file():
        errors.append(f"vendor cublasLt missing: {vendor_lt}")
    else:
        try:
            lib = ctypes.CDLL(str(vendor_lt))
            lib.cublasLtGetVersion.restype = ctypes.c_size_t
            version = int(lib.cublasLtGetVersion())
            if version != a.expected_version:
                errors.append(f"cublasLt version {version} != {a.expected_version}")
        except Exception as error:
            errors.append(f"cublasLt version call failed: {error}")
    report = {
        "schema": "oxide-mistral3-container-sdk-got/v1",
        "sdk": str(a.sdk),
        "sdk_sha256": sdk_sha,
        "expected_sha256": a.expected_sha,
        "runner": str(a.runner) if a.runner else None,
        "sdk_loaded": sdk is not None,
        "tei_forward_symbol": tei_forward_symbol,
        "bridge_header": str(a.bridge_header) if a.bridge_header else None,
        "tei_forward_args": header_args,
        "vendor": str(a.vendor),
        "cublaslt_version": version,
        "expected_version": a.expected_version,
        "torch_version": torch_version,
        "torch_cuda_initialized": cuda_initialized,
        "cublas_got_targets": targets,
        "errors": errors,
        "ok": not errors,
        "cuda_context_created": False,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    sys.exit(main())
