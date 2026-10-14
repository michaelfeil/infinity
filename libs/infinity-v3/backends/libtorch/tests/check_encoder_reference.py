"""Run encoder_reference_ffi.cpp against fixtures generated independently by Transformers."""
import ctypes as c
import json
import pathlib
import sys
import numpy as np
from safetensors.numpy import load_file

lib = c.CDLL(sys.argv[1])
lib.fixture_create.restype = c.c_void_p
lib.fixture_error.restype = c.c_char_p
lib.fixture_option.argtypes = [c.c_void_p, c.c_char_p, c.c_char_p]
lib.fixture_weight.argtypes = [c.c_void_p, c.c_char_p, c.c_void_p, c.c_void_p, c.c_int64]
lib.fixture_ready.argtypes = [c.c_void_p, c.c_int]
lib.fixture_forward.argtypes = [c.c_void_p] + [c.c_void_p] * 4 + [c.c_int64] * 2 + [c.c_int] * 2 + [c.c_void_p]
lib.fixture_delete.argtypes = [c.c_void_p]

def check(status):
    if status:
        raise RuntimeError(lib.fixture_error().decode())

def flatten(config, prefix=""):
    for key, value in config.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from flatten(value, name + ".")
        elif value is not None:
            yield name, value if isinstance(value, str) else json.dumps(value)

for path in sorted(pathlib.Path(sys.argv[2]).iterdir()):
    config = json.loads((path / "config.json").read_text())
    inputs = json.loads((path / "inputs.json").read_text())
    reference = load_file(path / "reference.safetensors")
    weights = load_file(path / "model.safetensors")
    ids = np.array(inputs["ids"], dtype=np.int64)
    types = np.ones_like(ids)
    if path.name == "mpnet":
        types[:] = 0
    positions = np.array(sum([list(range(n)) for n in inputs["lengths"]], []), dtype=np.int64)
    offsets = np.array([0] + list(np.cumsum(inputs["lengths"])), dtype=np.int32)
    for gpu in [0, 1]:
        handle = lib.fixture_create()
        try:
            for key, value in flatten(config):
                lib.fixture_option(handle, key.encode(), value.encode())
            for name, value in weights.items():
                value = value.astype(np.float32)
                shape = np.array(value.shape, dtype=np.int64)
                check(lib.fixture_weight(handle, name.encode(), value.ctypes.data, shape.ctypes.data, value.ndim))
            check(lib.fixture_ready(handle, gpu))
            for name, expected in reference.items():
                output = np.zeros_like(expected)
                prediction = 0 if name == "hidden" else 2 if path.name.endswith("token") else 1
                check(lib.fixture_forward(handle, ids.ctypes.data, types.ctypes.data, positions.ctypes.data,
                    offsets.ctypes.data, len(inputs["lengths"]), max(inputs["lengths"]), gpu, prediction, output.ctypes.data))
                error = np.max(np.abs(output - expected))
                print(path.name, "CUDA-fp16" if gpu else "CPU-fp32", name, "max_abs", error, flush=True)
                assert error < (0.006 if gpu else 2e-5)
        finally:
            lib.fixture_delete(handle)
