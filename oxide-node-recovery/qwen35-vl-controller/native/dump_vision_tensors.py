#!/usr/bin/env python3
"""Dump original Qwen3.5 vision tensors and flattened config for the Native probe.

Runs inside the pinned SDK container; reads only the original checkpoint shards
named by the original index. No network, no config rewrite, no tensor mutation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def flatten(value, prefix, out):
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flatten(item, f"{prefix}.{index}", out)
    elif isinstance(value, bool):
        out[prefix] = "true" if value else "false"
    elif value is not None:
        out[prefix] = str(value)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    root = Path(args.checkpoint)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    wanted = sorted(name for name in weight_map if name.startswith("model.visual."))
    if not wanted:
        raise SystemExit("no model.visual.* tensors in original index")
    shards = sorted({weight_map[name] for name in wanted})
    tensors = {}
    for shard in shards:
        tensors.update(load_file(str(root / shard), device="cpu"))
    blob = bytearray()
    lines = []
    dtype_name = None
    for name in wanted:
        tensor = tensors[name].contiguous()
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise SystemExit(f"unsupported dtype {tensor.dtype} for {name}")
        if dtype_name is None:
            dtype_name = str(tensor.dtype).replace("torch.", "")
        elif dtype_name != str(tensor.dtype).replace("torch.", ""):
            raise SystemExit("mixed vision dtypes are not supported by the probe")
        offset = len(blob)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        blob.extend(raw)
        shape = ",".join(str(dim) for dim in tensor.shape)
        lines.append(f"{name}\t{shape}\t{offset}\t{len(raw)}")
    (out / "tensors.bin").write_bytes(bytes(blob))
    (out / "tensors.txt").write_text("\n".join(lines) + "\n")
    config = json.loads((root / "config.json").read_text())
    options = {}
    flatten(config, "", options)
    (out / "options.txt").write_text(
        "\n".join(f"{key}\t{value}" for key, value in sorted(options.items())) + "\n"
    )
    (out / "manifest.json").write_text(json.dumps({
        "checkpoint": str(root),
        "dtype": dtype_name,
        "tensors": len(lines),
        "bytes": len(blob),
        "names": wanted,
    }, indent=2) + "\n")
    print(json.dumps({"status": "ok", "tensors": len(lines), "bytes": len(blob)}))

if __name__ == "__main__":
    main()
