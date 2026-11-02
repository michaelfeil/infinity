#!/usr/bin/env python3
"""Real CPU shared-library load/maps/symbol regression (no mocks, no SDK host)."""
import shutil
import tempfile
import unittest
from pathlib import Path

from load_probe import load_and_assert_mapped, mapped_paths

CANDIDATES = [
    (Path("/lib/x86_64-linux-gnu/libuuid.so.1"), "uuid_generate"),
    (Path("/lib/x86_64-linux-gnu/liblz4.so.1"), "LZ4_versionNumber"),
]


def temp_copy(path: Path) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="qv3-load-probe-"))
    destination = directory / ("probe-" + path.name)
    shutil.copy2(path, destination)
    return destination


class LoadProbeTests(unittest.TestCase):
    def _absent_candidate(self):
        before = mapped_paths()
        for path, symbol in CANDIDATES:
            if str(path.resolve()) not in before:
                return path, symbol
        source, symbol = CANDIDATES[0]
        return temp_copy(source), symbol

    def test_library_absent_before_load_is_mapped_after_and_symbol_owned(self):
        path, symbol = self._absent_candidate()
        before = mapped_paths()
        self.assertNotIn(str(path.resolve()), before)
        handle = load_and_assert_mapped(path, symbol)
        self.assertIn(str(path.resolve()), mapped_paths())
        self.assertIsNotNone(handle)

    def test_load_helper_rejects_already_mapped_target(self):
        path, symbol = temp_copy(CANDIDATES[0][0]), CANDIDATES[0][1]
        handle = load_and_assert_mapped(path, symbol)
        self.assertIsNotNone(handle)
        with self.assertRaises(RuntimeError):
            load_and_assert_mapped(path, symbol)


if __name__ == "__main__":
    unittest.main()
