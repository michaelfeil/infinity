#!/usr/bin/env python3
"""CPU-only tests for controller_v3 argv/env/pins/retrieval. No Docker/GPU."""
import json
import unittest
from pathlib import Path
from unittest import mock

import controller_v4 as c


class FakeDone:
    returncode = 0
    stdout = ""
    stderr = ""


class ControllerV3Tests(unittest.TestCase):
    def test_quality_env_has_fa4_and_kernel_pins(self):
        quality = " ".join(c.container_env("quality"))
        self.assertIn("CUDA_VISIBLE_DEVICES=0", quality)
        self.assertIn(f"INFINITY_FA4_ROOT={c.FA4_ROOT}", quality)
        self.assertIn(f"INFINITY_OXIDE_KERNEL_MODULE={c.UNION124}", quality)
        cpu = " ".join(c.container_env("cpu"))
        self.assertIn("CUDA_VISIBLE_DEVICES=", cpu)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0", cpu)

    def test_plan_has_vectors_lifecycle_compare_and_missing_packets(self):
        out = Path("/tmp/qv3-quality-plan-output")
        plan = c.plan("unit", out)
        jobs = {job["job"] for job in plan["jobs"]}
        self.assertIn("quality-vectors", jobs)
        self.assertIn("quality-lifecycle-smoke", jobs)
        self.assertIn("quality-compare", jobs)
        skipped = {item.get("name"): item.get("status") for item in plan["skipped"]}
        self.assertEqual(skipped.get("colqwen3-4b"), "native_packet_missing")
        self.assertEqual(skipped.get("nemotron-4b"), "native_packet_missing")
        for job in plan["jobs"]:
            if job["job"] == "cpu-bindings":
                self.assertIn("--library-sha", job["argv"])
                self.assertIn("verify_b66_v4.py", " ".join(map(str, job["argv"])))
            if job["job"] == "cpu-native-whole":
                self.assertIn("--library-sha", job["argv"])
        self.assertFalse(any("<checkpoint>" in " ".join(map(str, job["argv"])) for job in plan["jobs"]))

    def test_retrieval_maps_all_roles_and_fails_on_docker_error(self):
        calls = []
        def ok(*args, capture=False, timeout=300):
            calls.append(tuple(map(str, args))); return FakeDone()
        with mock.patch.object(c, "docker", ok):
            for job in ("cpu-bindings", "cpu-native-whole", "cpu-oxide-batch",
                        "quality-native", "quality-vectors", "quality-lifecycle-smoke"):
                c.retrieve(Path("/remote"), Path("/tmp/qv3-retrieve"), {"job": job, "case": "qwen3-vl-8b"})
        joined = [" ".join(call) for call in calls]
        self.assertTrue(any("b66-binding-proof.json" in call for call in joined))
        self.assertTrue(any("oxide-quality-vectors" in call for call in joined))
        def fail(*args, capture=False, timeout=300):
            done = FakeDone(); done.returncode = 1; done.stderr = "missing"; return done
        with mock.patch.object(c, "docker", fail):
            with self.assertRaises(RuntimeError):
                c.retrieve(Path("/remote"), Path("/tmp/qv3-retrieve"), {"job": "cpu-bindings", "case": "x"})

    def test_pins_are_truthful_and_quality_consumer_built(self):
        pins = c.pins()
        self.assertEqual(pins["source"]["source_commit"], c.E7_COMMIT)
        quality = pins["host_checks"]["quality_consumer"]
        if quality["pending_build"]:
            self.assertIsNone(quality["sha256"])
        else:
            self.assertTrue(quality["present_on_host"])
            self.assertEqual(quality["sha256"], c.QUALITY_CONSUMER_SHA)
        self.assertTrue(pins["fixture_pins"])
        self.assertTrue(pins["no_weight_copy"])


if __name__ == "__main__":
    unittest.main()
