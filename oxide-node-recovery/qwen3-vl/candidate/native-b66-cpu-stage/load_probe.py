#!/usr/bin/env python3
"""CPU-only shared-library load/maps/symbol helper.

Used by verify_b66.py so the actual SDK library is loaded before /proc/self/maps
is inspected, and by the CPU regression test with a real host library.
"""
import ctypes as C
import os
from pathlib import Path

class DlInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("base", C.c_void_p), ("symbol", C.c_char_p), ("address", C.c_void_p)]

def load_shared(path) -> C.CDLL:
    # os.RTLD_NOW is the Python/crt constant; ctypes does not expose RTLD_NOW.
    return C.CDLL(str(Path(path).resolve()), mode=os.RTLD_NOW | C.RTLD_GLOBAL)

def mapped_paths() -> set[str]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split()
        if parts:
            paths.add(parts[-1])
    return paths

def symbol_owner(handle: C.CDLL, symbol: str):
    pointer = C.cast(getattr(handle, symbol), C.c_void_p)
    dl = C.CDLL("libdl.so.2")
    dl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dl.dladdr.restype = C.c_int
    info = DlInfo()
    if not dl.dladdr(pointer, C.byref(info)) or not info.name:
        return None
    return Path(info.name.decode()).resolve()

def load_and_assert_mapped(path, symbol=None) -> C.CDLL:
    resolved = Path(path).resolve()
    if str(resolved) in mapped_paths():
        raise RuntimeError(f"{resolved} was already mapped before the probe")
    handle = load_shared(resolved)
    if str(resolved) not in mapped_paths():
        raise RuntimeError(f"{resolved} was not mapped after C.CDLL")
    if symbol is not None:
        owner = symbol_owner(handle, symbol)
        if owner != resolved:
            raise RuntimeError(f"symbol {symbol} owner {owner} != {resolved}")
    return handle
