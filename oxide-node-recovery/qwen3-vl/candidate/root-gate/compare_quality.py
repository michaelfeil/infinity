#!/usr/bin/env python3
"""Coupled Native-vs-Oxide Qwen3-VL quality comparison.

Inputs:
  --native  Native consumer results.json (b66 release consumer)
  --oxide   directory with per-pool Oxide consumer runs:
              oxide-root/pool0/oxide-consumer.json   (Cls)
              oxide-root/pool1/oxide-consumer.json   (Mean)
              oxide-root/pool2/oxide-consumer.json   (LastToken)
              oxide-root/raw/oxide-consumer.json     (optional token rows)
  --fixtures root-gate/native/fixtures.json
  --out     comparison JSON

Every value is read from actual f32 outputs. Missing output or missing
thresholds fails the arm; nothing is synthesized.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

POOLS = {"pool0": "cls", "pool1": "mean", "pool2": "last_token"}
THRESHOLDS = {
    "pooled": {"max_abs": 0.01, "relative_l2": 0.001, "cosine": 0.9999},
    "raw": {"max_abs": 0.02, "relative_l2": 0.002, "cosine": 0.999},
}


def read_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 4:
        raise SystemExit(f"{path} is not a whole f32 file")
    return [value[0] for value in struct.iter_unpack("<f", data)]


def metrics(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right) or not left:
        return {"values": len(left), "other_values": len(right), "geometry": False}
    max_abs = 0.0
    left_norm = 0.0
    right_norm = 0.0
    dot = 0.0
    for a, b in zip(left, right):
        if not (math.isfinite(a) and math.isfinite(b)):
            return {"values": len(left), "finite": False}
        diff = a - b
        max_abs = max(max_abs, abs(diff))
        left_norm += a * a
        right_norm += b * b
        dot += a * b
    left_norm = math.sqrt(left_norm)
    right_norm = math.sqrt(right_norm)
    relative = (
        math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right))) / left_norm
        if left_norm > 0
        else float("inf")
    )
    cosine = dot / (left_norm * right_norm) if left_norm > 0 and right_norm > 0 else 0.0
    return {
        "values": len(left),
        "finite": True,
        "geometry": True,
        "max_abs": max_abs,
        "relative_l2": relative,
        "cosine": cosine,
    }


def resolve_output(path: str, base: Path) -> Path:
    """Resolve a remote-recorded path to the retrieved local artifact."""
    recorded = Path(path)
    if recorded.is_file():
        return recorded
    candidate = base / recorded.name
    return candidate if candidate.is_file() else recorded


def load_oxide(run_dir: Path) -> dict[str, dict]:
    report = json.loads((run_dir / "oxide-consumer.json").read_text())
    records = report["records"]
    names = [record["fixture"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate fixture record")
    return {record["fixture"]: record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--oxide", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-raw", action="store_true")
    args = parser.parse_args()

    native_report = json.loads(args.native.read_text())
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    expected = {fixture["name"]: fixture for fixture in fixtures}
    required_roles = list(POOLS) + (["raw"] if args.allow_raw else [])
    failures = []
    results = []

    for role in required_roles:
        if not (args.oxide / role).is_dir():
            failures.append({"role": role, "error": "missing oxide role directory"})

    native: dict[str, dict] = {}
    for row in native_report["observations"]:
        name = row["fixture"]
        if name not in expected:
            failures.append({"fixture": name, "error": "unexpected Native observation"})
            continue
        if name in native:
            failures.append({"fixture": name, "error": "duplicate Native observation"})
        native[name] = row
    for name in expected:
        if name not in native:
            failures.append({"fixture": name, "error": "missing Native observation"})

    oxide_pools = {}
    for role in required_roles:
        run_dir = args.oxide / role
        if not run_dir.is_dir():
            continue
        try:
            mapping = load_oxide(run_dir)
            extra = set(mapping) - set(expected)
            if extra:
                failures.append({"role": role, "error": "unexpected oxide fixture records", "fixtures": sorted(extra)})
                mapping = {name: record for name, record in mapping.items() if name in expected}
            oxide_pools[role] = mapping
        except (ValueError, KeyError) as error:
            failures.append({"role": role, "error": f"invalid oxide report: {error}"})

    pooled_width = native_report.get("pooled_width", 0)
    output_width = native_report.get("output_width", 0)
    if pooled_width <= 0 or output_width <= 0:
        failures.append({"error": "invalid Native output widths"})

    for name, fixture in expected.items():
        native_row = native.get(name)
        if native_row is None:
            continue
        sequences = len(fixture["cumulative"]) - 1
        tokens = len(fixture["ids"])
        pooled_len = sequences * pooled_width
        raw_len = tokens * output_width
        native_output = resolve_output(native_row["output"], Path(args.native).parent)
        if not native_output.is_file():
            failures.append({"fixture": name, "error": "missing Native output file"})
            continue
        native_values = read_f32(native_output)
        if len(native_values) != pooled_len + raw_len:
            failures.append({"fixture": name, "error": "Native output length is not exact", "values": len(native_values), "expected": pooled_len + raw_len})
            continue
        native_raw = native_values[pooled_len : pooled_len + raw_len]

        for role, label in POOLS.items():
            mapping = oxide_pools.get(role)
            if mapping is None:
                continue
            oxide = mapping.get(name)
            if oxide is None:
                failures.append({"fixture": name, "pool": label, "error": "missing oxide fixture"})
                continue
            if (
                oxide.get("width") != pooled_width
                or oxide.get("sequences") != sequences
                or oxide.get("cumulative") != fixture["cumulative"]
                or oxide.get("values") != pooled_len
            ):
                failures.append({"fixture": name, "pool": label, "error": "invalid oxide schema/geometry", "record": oxide})
                continue
            right_path = resolve_output(oxide["path"], args.oxide / role)
            if not right_path.is_file() or right_path.stat().st_size != pooled_len * 4:
                failures.append({"fixture": name, "pool": label, "error": "missing or wrong-length oxide output"})
                continue
            native_pool = native_output.parent / f"{name}.pool{role[-1]}.f32"
            if not native_pool.is_file() or native_pool.stat().st_size != pooled_len * 4:
                failures.append({"fixture": name, "pool": label, "error": "missing or wrong-length Native pool output"})
                continue
            left = read_f32(native_pool)
            right = read_f32(right_path)
            score = metrics(left, right)
            threshold = THRESHOLDS["pooled"]
            passed = (
                score.get("finite") is True
                and score.get("geometry") is True
                and score["max_abs"] <= threshold["max_abs"]
                and score["relative_l2"] <= threshold["relative_l2"]
                and score["cosine"] >= threshold["cosine"]
            )
            if not passed:
                failures.append({"fixture": name, "pool": label, "metrics": score})
            results.append({"fixture": name, "kind": "pooled", "pool": label, "metrics": score, "pass": passed})

        if args.allow_raw:
            mapping = oxide_pools.get("raw")
            if mapping is None:
                continue
            oxide = mapping.get(name)
            if oxide is None:
                failures.append({"fixture": name, "kind": "raw", "error": "missing oxide fixture"})
                continue
            if (
                oxide.get("width") != output_width
                or oxide.get("sequences") != sequences
                or oxide.get("cumulative") != fixture["cumulative"]
                or oxide.get("values") != raw_len
            ):
                failures.append({"fixture": name, "kind": "raw", "error": "invalid oxide schema/geometry", "record": oxide})
                continue
            right_path = resolve_output(oxide["path"], args.oxide / "raw")
            if not right_path.is_file() or right_path.stat().st_size != raw_len * 4:
                failures.append({"fixture": name, "kind": "raw", "error": "missing or wrong-length oxide raw output"})
                continue
            right = read_f32(right_path)
            rows = raw_len // output_width
            row_metrics = []
            for row in range(rows):
                begin = row * output_width
                row_metrics.append(
                    metrics(native_raw[begin : begin + output_width], right[begin : begin + output_width])
                )
            score = {
                "rows": rows,
                "width": output_width,
                "finite": all(item.get("finite") for item in row_metrics),
                "max_abs": max(item.get("max_abs", float("inf")) for item in row_metrics),
                "max_relative_l2": max(item.get("relative_l2", float("inf")) for item in row_metrics),
                "min_cosine": min(item.get("cosine", 0.0) for item in row_metrics),
            }
            threshold = THRESHOLDS["raw"]
            passed = (
                score["finite"]
                and score["max_abs"] <= threshold["max_abs"]
                and score["max_relative_l2"] <= threshold["relative_l2"]
                and score["min_cosine"] >= threshold["cosine"]
            )
            if not passed:
                failures.append({"fixture": name, "kind": "raw", "metrics": score})
            results.append({"fixture": name, "kind": "raw", "metrics": score, "pass": passed})

    report = {
        "schema": "oxide-qwen3-vl-coupled-quality/v1",
        "native": str(args.native),
        "oxide": str(args.oxide),
        "fixtures": len(fixtures),
        "required_roles": required_roles,
        "thresholds": THRESHOLDS,
        "results": results,
        "failures": failures,
        "status": "passed" if not failures else "failed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "failures": len(failures), "results": len(results)}))
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
