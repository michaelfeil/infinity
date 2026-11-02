"""Real harmless CPU guard/record-path checks; never invokes Docker or Torch."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import root_control as rc
import native_gold as gold


class GuardReadiness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="q35-native-owned-cpu-")
        self.root = Path(self.temp.name)
        shutil.copy2(Path(__file__).with_name("remote_guard.py"), self.root / "remote_guard.py")

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_staging_has_every_required_file(self):
        with patch.object(rc, "HOST_BASE", self.root):
            staged, reused = rc.stage_host("cpu-stage")
        self.assertFalse(reused)
        for name in rc.IMPL_FILES + ("native_gold.py",) + rc.REQUIRED_INPUT_FILES:
            self.assertTrue((staged / name).is_file(), name)
        self.assertTrue((staged / ".owner").is_file())

    def test_missing_imported_source_rejected_before_staging(self):
        source = self.root / "incomplete-input"
        source.mkdir()
        for path in rc.SOURCE_INPUT.iterdir():
            if path.is_file() and path.name != "native_raw_probe.py":
                shutil.copy2(path, source / path.name)
        with patch.object(rc, "HOST_BASE", self.root), patch.object(rc, "SOURCE_INPUT", source):
            with self.assertRaisesRegex(rc.ControlError, "native_raw_probe.py"):
                rc.stage_host("must-not-be-created")
        self.assertFalse((self.root / "must-not-be-created").exists())

    def test_gold_missing_source_cannot_be_stably_none(self):
        with patch.object(gold, "ROOT", self.root), patch.object(gold, "SOURCE_INPUT", self.root):
            with self.assertRaisesRegex(gold.GoldError, "required source is missing"):
                gold.source_hashes()

    def test_proxy_values_are_not_serialized(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "sentinel-private-value"}):
            argv = rc.docker_exec(str(self.root), ["cpu-command"], {})
        self.assertIn("HTTP_PROXY", argv)
        self.assertFalse(any("sentinel-private-value" in value for value in argv))
        with self.assertRaises(rc.ControlError):
            rc.docker_exec(str(self.root), ["cpu-command"], {"UNLISTED": "value"})
        with self.assertRaises(rc.ControlError):
            rc.docker_exec(str(self.root), ["cpu-command"], {"LD_LIBRARY_PATH": "not-the-fixed-vendor"})

    def test_quality_rejects_semantic_only_numeric_pass(self):
        (self.root / "native-gold.json").write_text(json.dumps({
            "status": "ok", "identity_stable": True, "sources_stable": True}))
        (self.root / "compare.json").write_text(json.dumps({
            "status": "UNCHANGED", "unchanged": True, "numeric_qualified": False,
            "float_count": 0}))
        with self.assertRaisesRegex(rc.ControlError, "nonempty numeric"):
            rc.check_phase(SimpleNamespace(phase="quality"), self.root)

    def test_actual_bounded_result_record_serialization(self):
        step = {"label": "harmless", "env": {"CUDA_VISIBLE_DEVICES": ""},
                "argv": [sys.executable, "-c", "print('owned-cpu-ready')"]}
        # Substitute only the transport; execute the actual guard/timeout/record path.
        with patch.object(rc, "CONTAINER_PYTHON", sys.executable), patch.object(
                rc, "docker_exec", side_effect=lambda directory, argv, env: argv):
            intended = rc.guard_argv(step, str(self.root), 10)
            actual = rc.run_bounded(step, self.root, str(self.root), 10)
        merged = rc.merge_record({"argv": intended, "label": "harmless"}, actual)
        restored = json.loads(json.dumps(merged))
        self.assertEqual(restored["argv"], intended)
        self.assertEqual(restored["returncode"], 0)
        self.assertFalse(restored["timed_out"])
        self.assertEqual(restored["stdout"].strip(), "owned-cpu-ready")
        self.assertTrue((self.root / "guard-harmless.json").is_file())
        with self.assertRaises(rc.ControlError):
            rc.merge_record({"argv": ["wrong"]}, actual)

    def test_actual_guard_stop_checks_owned_process_identity(self):
        state = self.root / "sleep.json"
        guard = self.root / "remote_guard.py"
        process = subprocess.Popen(
            [sys.executable, str(guard), "--state", str(state), "--seconds", "60",
             "--", sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        try:
            deadline = time.monotonic() + 5
            while not state.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(state.exists())
            original = json.loads(state.read_text())
            self.assertEqual(original["pid"], process.pid)
            # A stale identity must never signal even this owned dummy.
            stale = dict(original, start_ticks="deliberately-stale")
            state.write_text(json.dumps(stale))
            check = subprocess.run([sys.executable, str(guard), "--state", str(state),
                                    "--stop"], capture_output=True, text=True, timeout=10)
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertFalse(json.loads(check.stdout)["group_signal_sent"])
            self.assertIsNone(process.poll())
            state.write_text(json.dumps(original))
            stopped = subprocess.run([sys.executable, str(guard), "--state", str(state),
                                      "--stop"], capture_output=True, text=True, timeout=10)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertTrue(json.loads(stopped.stdout)["group_signal_sent"])
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                process.communicate(timeout=10)

    def test_actual_guard_rejects_unowned_state_directory(self):
        outside = self.root / "unowned" / "state.json"
        outside.parent.mkdir()
        check = subprocess.run(
            [sys.executable, str(self.root / "remote_guard.py"), "--state", str(outside),
             "--seconds", "1", "--", sys.executable, "-c", "raise SystemExit(0)"],
            capture_output=True, text=True, timeout=10)
        self.assertNotEqual(check.returncode, 0)
        self.assertIn("owned run root", check.stderr)
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
