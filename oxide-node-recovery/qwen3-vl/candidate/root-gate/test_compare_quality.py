#!/usr/bin/env python3
"""CPU-only strict parser/serialization/comparator tests (no model/GPU)."""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPARE = ROOT / "compare_quality.py"


def write_f32(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(struct.pack("<f", v) for v in values))


def record(name, path, values, width, sequences, cumulative):
    return {"fixture": name, "kind": "pooled", "width": width, "sequences": sequences,
            "values": len(values), "cumulative": cumulative, "sha256": "x",
            "path": path, "finite": True}


class Workspace:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.native = self.root / "native"
        self.oxide = self.root / "oxide"
        self.native.mkdir()
        write_f32(self.native / "a.f32", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        for code, values in [("0", [1.0, 2.0]), ("1", [1.5, 2.5]), ("2", [1.0, 2.0])]:
            write_f32(self.native / f"a.pool{code}.f32", values)
        (self.native / "results.json").write_text(json.dumps({
            "pooled_width": 2, "output_width": 2,
            "observations": [{"fixture": "a", "output": "/remote/native/a.f32"}],
        }))
        for code, values in [("0", [1.0, 2.0]), ("1", [1.5, 2.5]), ("2", [1.0, 2.0])]:
            run = self.oxide / f"pool{code}"
            write_f32(run / "a.f32", values)
            (run / "oxide-consumer.json").write_text(json.dumps({
                "records": [record("a", f"/remote/oxide/pool{code}/a.f32", values, 2, 1, [0, 2])]}))
        raw = self.oxide / "raw"
        write_f32(raw / "a.f32", [3.0, 4.0, 5.0, 6.0])
        (raw / "oxide-consumer.json").write_text(json.dumps({
            "records": [record("a", "/remote/oxide/raw/a.f32", [3.0, 4.0, 5.0, 6.0], 2, 1, [0, 2])]}))
        self.fixtures = self.root / "fixtures.json"
        self.fixtures.write_text(json.dumps({"fixtures": [{"name": "a", "ids": [1, 2], "cumulative": [0, 2]}]}))
        self.out = self.root / "quality.json"

    def run(self, allow_raw=True):
        cmd = [sys.executable, str(COMPARE), "--native", str(self.native / "results.json"),
               "--oxide", str(self.oxide), "--fixtures", str(self.fixtures), "--out", str(self.out)]
        if allow_raw:
            cmd.append("--allow-raw")
        return subprocess.run(cmd, capture_output=True, text=True)

    def close(self):
        self.tmp.cleanup()


class CompareQualityTests(unittest.TestCase):
    def test_valid_all_roles_pass_and_retrieved_paths_resolve(self):
        w = Workspace()
        try:
            result = w.run()
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(w.out.read_text())
            self.assertEqual(report["status"], "passed", report)
            self.assertEqual(len(report["results"]), 4)
        finally:
            w.close()

    def test_missing_pool_or_raw_fails(self):
        for missing in ("pool0", "raw"):
            w = Workspace()
            try:
                for path in sorted((w.oxide / missing).glob("*")):
                    path.unlink()
                (w.oxide / missing).rmdir()
                result = w.run()
                self.assertNotEqual(result.returncode, 0, missing)
                self.assertEqual(json.loads(w.out.read_text())["status"], "failed")
            finally:
                w.close()

    def test_duplicate_oxide_record_fails_before_metrics(self):
        w = Workspace()
        try:
            run = w.oxide / "pool0"
            data = json.loads((run / "oxide-consumer.json").read_text())
            data["records"].append(dict(data["records"][0]))
            (run / "oxide-consumer.json").write_text(json.dumps(data))
            result = w.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(any("duplicate" in str(f) for f in json.loads(w.out.read_text())["failures"]))
        finally:
            w.close()

    def test_wrong_width_sequences_cumulative_fail(self):
        for field, value in [("width", 999), ("sequences", 999), ("cumulative", [0, 999])]:
            w = Workspace()
            try:
                run = w.oxide / "pool1"
                data = json.loads((run / "oxide-consumer.json").read_text())
                data["records"][0][field] = value
                (run / "oxide-consumer.json").write_text(json.dumps(data))
                result = w.run()
                self.assertNotEqual(result.returncode, 0, field)
                self.assertEqual(json.loads(w.out.read_text())["status"], "failed")
            finally:
                w.close()

    def test_unexpected_extra_native_observation_fails(self):
        w = Workspace()
        try:
            data = json.loads((w.native / "results.json").read_text())
            data["observations"].append({"fixture": "UNEXPECTED", "output": "/remote/native/u.f32"})
            (w.native / "results.json").write_text(json.dumps(data))
            result = w.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(any("unexpected Native" in str(f) for f in json.loads(w.out.read_text())["failures"]))
        finally:
            w.close()

    def test_unexpected_extra_oxide_record_fails(self):
        w = Workspace()
        try:
            run = w.oxide / "pool0"
            data = json.loads((run / "oxide-consumer.json").read_text())
            data["records"].append(record("UNEXPECTED", "/remote/oxide/pool0/u.f32", [1.0, 2.0], 2, 1, [0, 2]))
            (run / "oxide-consumer.json").write_text(json.dumps(data))
            result = w.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(any("unexpected oxide" in str(f) for f in json.loads(w.out.read_text())["failures"]))
        finally:
            w.close()

    def test_trailing_native_f32_word_fails_exact_length(self):
        w = Workspace()
        try:
            write_f32(w.native / "a.f32", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            result = w.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(any("not exact" in str(f) for f in json.loads(w.out.read_text())["failures"]))
        finally:
            w.close()

    def test_duplicate_native_observation_fails(self):
        w = Workspace()
        try:
            data = json.loads((w.native / "results.json").read_text())
            data["observations"].append(dict(data["observations"][0]))
            (w.native / "results.json").write_text(json.dumps(data))
            result = w.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(w.out.read_text())["status"], "failed")
        finally:
            w.close()


if __name__ == "__main__":
    unittest.main()
