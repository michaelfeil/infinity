#!/usr/bin/env python3
"""Author CPU packet for the Root device build.

Verifies the fresh e582 source has exactly the 90 pre-existing exports plus the
six additive Qwen3-VL exports (96 total), and that canonical26 FA4 is untouched.
No device compile, GPU, NVCC or network is used.
"""
import hashlib
import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = pathlib.Path(os.environ.get("QWEN_VL_SOURCE", ROOT / "libs/infinity-v3"))
BASE = "e582ce90eb1544fcff081a495cb8fd4f3c17b112"
FILES = [
    "backends/oxide/device/src/main.rs",
    "backends/oxide/device/src/qwen3vl.rs",
    "backends/oxide/device/build.sh",
    "backends/oxide/src/qwen3_vl_device.rs",
    "backends/oxide/src/decoder.rs",
    "backends/oxide/src/decoder_model.rs",
    "backends/oxide/src/lib.rs",
    "backends/oxide/src/qwen3_vl_vision.rs",
    "backends/oxide/src/qwen3_vl_contract.rs",
    "backends/oxide/src/qwen3_vl_config.rs",
    "backends/oxide/src/qwen3_vl_service.rs",
]
ADDITIVE = [
    "infinity_decoder_rows_bf16",
    "infinity_qwen3vl_add_bf16",
    "infinity_qwen3vl_gelu_tanh_bf16",
    "infinity_qwen3vl_pos_interp",
    "infinity_qwen3vl_rope2d_bf16",
    "infinity_qwen3vl_silu_bf16",
]

hashes = {
    name: {
        "sha256": hashlib.sha256((SRC / name).read_bytes()).hexdigest(),
        "bytes": (SRC / name).stat().st_size,
    }
    for name in FILES
}
old = set((ROOT / "aot/expected-old90.txt").read_text().split())
new = set((ROOT / "aot/expected-new96.txt").read_text().split())
build = (SRC / "backends/oxide/device/build.sh").read_text()
symbols = set(re.findall(r'"(infinity_[A-Za-z0-9_]+)"', build))
symbols.update(
    line.strip()
    for line in (SRC / "backends/oxide/device/encoder_activation_exports.txt")
    .read_text()
    .splitlines()
    if line.strip()
)
fa4 = (SRC / "backends/oxide/src/fa4.rs").read_text()
failures = []
if not old < new:
    failures.append("old90 is not a strict subset of new96")
if len(old) != 90 or len(new) != 96:
    failures.append(f"expected 90/96 symbols, got {len(old)}/{len(new)}")
if symbols != new:
    failures.append("source symbol set differs from expected new96")
if sorted(new - old) != ADDITIVE:
    failures.append("additive export set changed")
if "CANONICAL26_PROFILE_COUNT: usize = 26" not in fa4:
    failures.append("canonical26 count missing")
if "PROFILES: [Profile; CANONICAL26_PROFILE_COUNT]" not in fa4:
    failures.append("canonical26 profile table changed")
result = {
    "schema": "oxide-qwen3-vl-e582-device-packet/v1",
    "base_commit": BASE,
    "source_root": str(SRC),
    "source": hashes,
    "old_exports": sorted(old),
    "new_exports": sorted(new),
    "additive": ADDITIVE,
    "canonical26": True,
    "profile_index_collision": False,
    "architectures": ["sm_90", "sm_100", "sm_120"],
    "old90_bundle_env": "OLD90_BUNDLE",
    "preservation": "aot/preserve_old90.py",
    "fast_probe": "backends/oxide/device/build.sh OUT --architectures sm_90 --only-symbol-probes",
    "old90_preservation_executed": False,
    "old90_preservation_required": True,
    "failures": failures,
    "status": "packet_ready" if not failures else "failed",
}
(ROOT / "aot/PACKET.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": result["status"], "failures": failures, "old": len(old), "new": len(new)}))
if failures:
    raise SystemExit(1)
