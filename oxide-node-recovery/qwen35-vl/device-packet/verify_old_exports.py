#!/usr/bin/env python3
"""Prove the additive Qwen3.5-VL leaf preserves the published 86 exports."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "libs/infinity-v3/backends/oxide/device/build.sh"
OLD86 = Path(__file__).resolve().parent / "old86.txt"
QWEN35 = ROOT / "libs/infinity-v3/backends/oxide/device/qwen35_exports.txt"
QWEN35VL = ROOT / "libs/infinity-v3/backends/oxide/device/qwen35vl_exports.txt"
ENCODER = ROOT / "libs/infinity-v3/backends/oxide/device/encoder_activation_exports.txt"
QWEN35VL_SRC = ROOT / "libs/infinity-v3/backends/oxide/device/src/qwen35vl.rs"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_names(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "old-exports-proof.json")
    args = parser.parse_args()
    text = BUILD.read_text()
    prefix = text.split("# Additive optional encoder-activation exports")[0]
    old = set()
    for match in re.finditer(r'symbols = sorted\((?:symbols \+ )?\[(.*?)\]\)', prefix, re.S):
        old.update(re.findall(r'"([^"]+)"', match.group(1)))
    pinned = set(read_names(OLD86))
    qwen35 = read_names(QWEN35)
    qwen35vl = read_names(QWEN35VL)
    encoder = read_names(ENCODER)
    failures = []
    if old != pinned:
        failures.append(f"old export set drifted: missing={sorted(pinned-old)} extra={sorted(old-pinned)}")
    if len(qwen35) != 9 or len(set(qwen35)) != 9:
        failures.append(f"Qwen3.5 foundation exports are not nine unique names: {qwen35}")
    if len(qwen35vl) != 7 or len(set(qwen35vl)) != 7:
        failures.append(f"qwen35vl exports are not exactly seven unique names: {qwen35vl}")
    if len(encoder) != 4 or len(set(encoder)) != 4:
        failures.append(f"encoder activation exports drifted: {encoder}")
    overlap = (old & set(qwen35)) | (old & set(encoder)) | (old & set(qwen35vl)) | (set(qwen35) & set(qwen35vl))
    if overlap:
        failures.append(f"additive export collision: {sorted(overlap)}")
    if "qwen35_exports = [" not in text or "symbols = sorted(symbols + qwen35_exports)" not in text:
        failures.append("build.sh does not append the qualified Qwen3.5 text leaf")
    if "qwen35vl_exports = [" not in text or "symbols = sorted(symbols + qwen35vl_exports)" not in text:
        failures.append("build.sh does not append the qwen35vl leaf")
    for needle in ('manifest["sources"]["qwen35vl_exports.txt"]',
                   'manifest["sources"]["src/qwen35vl.rs"]',
                   '"qwen35vl_exports.txt",'):
        if needle not in text:
            failures.append(f"build.sh missing {needle}")
    report = {
        "schema": "oxide-qwen35vl-old-exports-proof/v1",
        "status": "passed" if not failures else "failed",
        "old_exports": len(old),
        "foundation_additive_exports": len(qwen35),
        "encoder_activation_additive_exports": len(encoder),
        "qwen35vl_additive_exports": len(qwen35vl),
        "new_total_exports": len(old) + len(qwen35) + len(encoder) + len(qwen35vl),
        "qwen35vl_names": qwen35vl,
        "old86_pin_sha256": sha(OLD86),
        "build_sha256": sha(BUILD),
        "qwen35vl_source_sha256": sha(QWEN35VL_SRC),
        "semantic_cubin_comparison": "Root-only: requires the published old86 bundle",
        "failures": failures,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failures": failures, "out": str(args.out)}))
    return 0 if not failures else 1

if __name__ == "__main__":
    raise SystemExit(main())
