#!/usr/bin/env python3
"""CPU-only argv/env/pin tests for controller_v2. No Docker/GPU/Torch/network."""
import json
import unittest
from pathlib import Path
from unittest import mock

import controller_v2 as c


class FakeDone:
    returncode = 0
    stdout = ""
    stderr = ""


class ControllerV2Tests(unittest.TestCase):
    def test_phase_env_is_explicit(self):
        cpu = c.container_env("cpu")
        quality = c.container_env("quality")
        self.assertIn("CUDA_VISIBLE_DEVICES=", cpu)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0", cpu)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", quality)
        self.assertTrue(any(x.startswith("LD_PRELOAD=") for x in quality))

    def test_plan_uses_real_remote_guard_taskset_and_bench_paths(self):
        plan = c.plan("unit-plan")
        self.assertTrue(plan["no_weight_copy"])
        self.assertTrue(all(path.startswith("/bench/") for path in plan["read_only_models"]))
        for job in plan["jobs"]:
            argv = " ".join(map(str, job["argv"]))
            self.assertIn("remote_guard.py", argv)
            self.assertIn("taskset", argv)
            self.assertNotIn("<checkpoint>", argv)
            if job["phase"] == "cpu":
                self.assertIn("CUDA_VISIBLE_DEVICES=", argv)
                self.assertNotIn("CUDA_VISIBLE_DEVICES=0", argv)
                self.assertIn("taskset -c 160-167", argv)
            else:
                self.assertIn("CUDA_VISIBLE_DEVICES=0", argv)
                self.assertIn("taskset -c 184-191", argv)
        jobs = {job["job"] for job in plan["jobs"]}
        self.assertIn("cpu-native-whole", jobs)
        self.assertIn("quality-lifecycle", jobs)

    def test_stage_mocks_docker_and_never_copies_checkpoints(self):
        calls = []
        def fake_docker(*args, capture=False, timeout=300):
            calls.append(tuple(map(str, args)))
            return FakeDone()
        remote = Path(c.REMOTE_BASE) / "unit-stage"
        with mock.patch.object(c, "docker", fake_docker):
            self.assertTrue(c.stage(remote))
        flattened = [" ".join(call) for call in calls]
        self.assertTrue(any(call and call[0] == "cp" for call in calls))
        self.assertFalse(any("/bench/" in call for call in flattened))
        self.assertFalse(any("libtei_torch.so" in call for call in flattened))

    def test_host_pins_and_real_manifest_hashes(self):
        pins = c.pins()
        self.assertEqual(pins["source"]["source_commit"], c.E7_COMMIT)
        self.assertEqual(pins["host_checks"]["host_consumer"]["sha256"], c.HOST_CONSUMER_SHA)
        self.assertTrue(pins["fixture_pins"])
        self.assertTrue(pins["no_weight_copy"])
        fixture = json.loads(c.FIXTURES_MANIFEST.read_text())
        self.assertEqual(len(fixture["fixtures"]), 8)


if __name__ == "__main__":
    unittest.main()
