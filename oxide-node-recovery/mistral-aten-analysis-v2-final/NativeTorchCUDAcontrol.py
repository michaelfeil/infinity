#!/usr/bin/env python3
"""Root-run CUDA control for the Mistral3 model-dtype BF16 mean.

Run only on GPU 0 with the pinned SDK, for example:

    CUDA_VISIBLE_DEVICES=0 /opt/torch-sdk/bin/python3.12 \
        NativeTorchCUDAcontrol.py --out ./native_cuda_control_out

The script refuses to run unless CUDA_VISIBLE_DEVICES is exactly "0". It uses
NumPy nowhere: CPU-contiguous bytes are read with ctypes.string_at(data_ptr, n).
Inputs are generated deterministically on CPU, then the native control is
computed on cuda:0.

Default `--suite first` is the bounded full-word control: one pattern per
(H, T), cycling sparse cancellation -> stress -> normal across the T set. Use
`--suite all` for all three patterns at every (H, T); that is roughly three
times the binary input volume.
"""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

PINNED_GIT = "5c4886908584029761b579af026dcfb627c84070"
T_VALUES = (5, 7, 63, 64, 65, 1020, 1021, 1024, 2048, 4096)
H_VALUES = (3072, 4096, 5120)
EPS_BITS = 0x3280  # BF16 2^-26
PATTERNS = ("sparse", "normal", "stress")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def widen(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def aten_ctas_per_output(rows, width, num_sm, max_threads_per_sm):
    """Host mirror of Reduce.cuh global-reduction C for the admitted layout."""
    if rows < 1021 or max_threads_per_sm < 128:
        return 1
    target = int(num_sm) * (int(max_threads_per_sm) // 128)
    grid = (int(width) + 127) // 128
    if grid == 0 or grid > target:
        return 1
    values_per_thread = (int(rows) + 3) // 4
    ctas1 = (target + grid - 1) // grid
    ctas2 = (values_per_thread + 15) // 16
    ctas3 = (values_per_thread + 255) // 256
    ctas = min(max(ctas1, ctas3), ctas2)
    return max(ctas, 1)


def tensor_bytes(tensor):
    tensor = tensor.contiguous().cpu()
    nbytes = tensor.numel() * tensor.element_size()
    return ctypes.string_at(tensor.data_ptr(), nbytes)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def lcg_bits(kind, width, seed):
    state = seed & 0xFFFFFFFF
    bits = []
    stress_exponents = (0x3B, 0x3D, 0x3F, 0x40, 0x41, 0x42)
    for column in range(width):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        if kind == "stress":
            exponent = stress_exponents[(state >> 24) % len(stress_exponents)]
        else:
            exponent = 0x3F + ((state >> 29) & 1)
        mantissa = (state >> 8) & 0x7F
        value_bits = (exponent << 7) | mantissa
        if (state >> 16) & 1:
            value_bits |= 0x8000
        bits.append(value_bits)
    if kind == "stress":
        # Adjacent ± pairs force cancellation while still spanning magnitudes.
        for column in range(1, width, 5):
            bits[column] = bits[column - 1] ^ 0x8000
    return bits


def make_input(torch, kind, rows, width, seed):
    if kind == "sparse":
        result = torch.zeros((rows, width), dtype=torch.bfloat16)
        feature = seed % 127
        if rows < 64:
            # T=5..63 witness: ATen leaves eps; sequential leaves 2*eps.
            prefix = (1.0, widen(EPS_BITS), -1.0, widen(EPS_BITS), widen(EPS_BITS))
        else:
            # T>=64 witness: the block-y tree cancels to 0; sequential leaves 2*eps.
            prefix = (1.0, -1.0, widen(EPS_BITS), widen(EPS_BITS))
        limit = min(rows, len(prefix))
        for row in range(limit):
            result[row, feature] = prefix[row]
        return result

    row_count = min(rows, 8)
    small_rows = []
    for row in range(row_count):
        row_bits = lcg_bits(kind, width, seed ^ ((row + 1) * 0x9E3779B1))
        small_rows.append([widen(bits) for bits in row_bits])
    small = torch.tensor(small_rows, dtype=torch.bfloat16)
    repeat = (rows + row_count - 1) // row_count
    return small.repeat((repeat, 1))[:rows].clone().contiguous()


def choose_patterns(suite):
    if suite == "all":
        return {t: PATTERNS for t in T_VALUES}
    mapping = {}
    for total in T_VALUES:
        if total <= 1021:
            mapping[total] = ("sparse",)
        elif total in (1024, 2048):
            mapping[total] = ("stress",)
        else:
            mapping[total] = ("normal",)
    return mapping


def device_properties(torch):
    props = torch.cuda.get_device_properties(0)
    names = (
        "name",
        "major",
        "minor",
        "total_memory",
        "multi_processor_count",
        "max_threads_per_multi_processor",
        "warp_size",
        "regs_per_multiprocessor",
        "shared_memory_per_multiprocessor",
        "shared_memory_per_block",
        "l2_cache_size",
    )
    result = {}
    for name in names:
        if hasattr(props, name):
            result[name] = getattr(props, name)
    if "multi_processor_count" not in result:
        raise SystemExit("device properties: multi_processor_count missing")
    if "max_threads_per_multi_processor" not in result:
        raise SystemExit("device properties: max_threads_per_multi_processor missing")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./native_cuda_control_out")
    parser.add_argument("--suite", choices=("first", "all"), default="first")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit('refusing to run: set CUDA_VISIBLE_DEVICES="0" explicitly')

    import torch  # noqa: PLC0415 - guarded import after the device check

    git_version = torch.version.git_version or ""
    if not git_version.startswith(PINNED_GIT[:12]):
        raise SystemExit(
            f"refusing to run: torch git {git_version!r} is not pinned {PINNED_GIT}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise SystemExit("current CUDA device is not 0")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    props = device_properties(torch)
    num_sm = int(props["multi_processor_count"])
    max_threads = int(props["max_threads_per_multi_processor"])

    patterns = choose_patterns(args.suite)
    estimate = sum(t * h * 2 for h in H_VALUES for t in T_VALUES
                   for _ in patterns[t])
    print(f"suite={args.suite} cases={sum(len(v) for v in patterns.values()) * len(H_VALUES)} "
          f"estimated_input_bytes={estimate}")

    manifest = {
        "tool": "NativeTorchCUDAcontrol.py",
        "pin": PINNED_GIT,
        "torch_version": torch.__version__,
        "torch_git": git_version,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count_visible": torch.cuda.device_count(),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "device_props": props,
        "cases": [],
    }

    for h_index, height in enumerate(H_VALUES):
        for t_index, rows in enumerate(T_VALUES):
            start = 1 + (t_index * 7 + h_index * 11) % 23
            for pattern in patterns[rows]:
                seed = (0x5C488 + h_index * 1009 + t_index * 9176 + PATTERNS.index(pattern))
                source = make_input(torch, pattern, rows, height, seed)
                storage = torch.zeros((start + rows + 3, height), dtype=torch.bfloat16)
                storage[:start].fill_(1.0)
                storage.narrow(0, start, rows).copy_(source)
                storage[start + rows:].fill_(-1.0)

                storage_cuda = storage.to("cuda:0", non_blocking=False)
                value_cuda = storage_cuda.narrow(0, start, rows)
                sum_cuda = value_cuda.sum(0, True)
                mean_cuda = (sum_cuda / value_cuda.size(0)).to(torch.float32)

                sum_cpu = sum_cuda.to("cpu").contiguous()
                mean_cpu = mean_cuda.to("cpu").contiguous()
                input_bytes = tensor_bytes(storage)
                sum_bytes = tensor_bytes(sum_cpu)
                mean_bytes = tensor_bytes(mean_cpu)

                case_id = f"h{height}_t{rows}_{pattern}"
                case_dir = out / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                files = {}
                for name, data, dtype, shape in (
                    ("input_storage_bf16.bin", input_bytes, "bfloat16",
                     [start + rows + 3, height]),
                    ("sum_bf16.bin", sum_bytes, "bfloat16", [1, height]),
                    ("mean_f32.bin", mean_bytes, "float32", [1, height]),
                ):
                    path = case_dir / name
                    path.write_bytes(data)
                    files[name] = {
                        "path": str(path.relative_to(out)),
                        "bytes": len(data),
                        "sha256": sha256(data),
                        "dtype": dtype,
                        "shape": shape,
                    }

                ctas = aten_ctas_per_output(rows, height, num_sm, max_threads)
                manifest["cases"].append({
                    "id": case_id,
                    "h": height,
                    "t": rows,
                    "pattern": pattern,
                    "seed": seed,
                    "selected": [1],
                    "cu_seqlens": [0, start, start + rows],
                    "storage_start_row": start,
                    "storage_rows": start + rows + 3,
                    "storage_prefix_value": 1.0,
                    "storage_tail_value": -1.0,
                    "selected_input_shape": [rows, height],
                    "ctas_per_output_expected": ctas,
                    "grid_x_expected": (height + 127) // 128,
                    "files": files,
                })
                print(f"{case_id} start={start} C={ctas}")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    print(f"cases={len(manifest['cases'])}")


if __name__ == "__main__":
    main()
