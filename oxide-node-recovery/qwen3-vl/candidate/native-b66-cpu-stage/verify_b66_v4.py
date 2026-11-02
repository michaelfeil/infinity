#!/usr/bin/env python3
"""CPU-only proof that b66 and libtorch_cuda resolve cublas relocations to the
pinned container vendor pair. Imports Torch only; creates no CUDA context."""
import argparse
import ctypes as C
import hashlib
import json
import subprocess
from pathlib import Path

from load_probe import load_shared

B66 = Path("/tmp/native214-fa4-mode20/install/lib/libtei_torch.so")
VENDOR = Path("/tmp/oxide-host-cublas-130200")

class DlInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("base", C.c_void_p), ("symbol", C.c_char_p), ("address", C.c_void_p)]

def verify():
    import torch
    # Load the actual SDK objects BEFORE capturing /proc/self/maps.
    torch_cuda = Path(torch.__file__).resolve().parent / "lib/libtorch_cuda.so"
    torch_handle = load_shared(torch_cuda)
    lib = load_shared(B66)
    maps = Path("/proc/self/maps").read_text().splitlines()
    dl = C.CDLL("libdl.so.2")
    dl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dl.dladdr.restype = C.c_int
    objects = [torch_cuda, B66]
    result = {
        "gpu_initialized": torch.cuda.is_initialized(),
        "vendor": str(VENDOR),
        "objects": {},
        "all_blas_relocations_vendor": True,
    }
    for obj in objects:
        matched = [line.split() for line in maps if line.split()[-1] == str(obj)]
        if not matched:
            raise RuntimeError(f"object not mapped: {obj}")
        starts = [int(parts[0].split("-")[0], 16) for parts in matched if int(parts[2], 16) == 0]
        if len(starts) != 1:
            raise RuntimeError(f"expected one zero-offset LOAD mapping: {obj}")
        base = starts[0]
        targets = []
        process = subprocess.Popen(["readelf", "-rW", str(obj)], stdout=subprocess.PIPE, text=True)
        for line in process.stdout:
            parts = line.split()
            if len(parts) < 5 or parts[2] not in ("R_X86_64_JUMP_SLOT", "R_X86_64_GLOB_DAT"):
                continue
            symbol = parts[4].split("@")[0]
            if not symbol.startswith("cublas"):
                continue
            pointer = C.c_void_p.from_address(base + int(parts[0], 16)).value
            info = DlInfo()
            found = dl.dladdr(C.c_void_p(pointer), C.byref(info))
            path = info.name.decode() if found and info.name else None
            good = path is not None and path.startswith(str(VENDOR) + "/")
            targets.append({"symbol": symbol, "resolved_path": path, "vendor": good})
            result["all_blas_relocations_vendor"] &= good
        if process.wait() != 0:
            raise RuntimeError("readelf failed")
        if not targets:
            raise RuntimeError(f"no BLAS relocation targets: {obj}")
        result["objects"][str(obj)] = targets
    _ = torch_handle
    lib.cublasLtGetVersion.restype = C.c_size_t
    result["cublaslt_version"] = lib.cublasLtGetVersion()
    if result["gpu_initialized"]:
        raise RuntimeError("Torch CUDA initialized during CPU bindings check")
    if result["cublaslt_version"] != 130200:
        raise RuntimeError(f"unexpected cublasLt {result['cublaslt_version']}")
    if not result["all_blas_relocations_vendor"]:
        raise RuntimeError("a cublas relocation is not owned by the pinned vendor")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-sha", default=None)
    args = parser.parse_args()
    actual = hashlib.sha256(B66.read_bytes()).hexdigest()
    if args.library_sha is not None and actual != args.library_sha:
        raise SystemExit(f"b66 sha mismatch: {actual} != {args.library_sha}")
    result = verify()
    result["b66_sha256"] = actual
    out = Path(__file__).resolve().parent / "b66-binding-proof.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "cublaslt": result["cublaslt_version"], "objects": {k: len(v) for k, v in result["objects"].items()}}))
