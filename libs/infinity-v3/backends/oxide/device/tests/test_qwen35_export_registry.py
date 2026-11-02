# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""CPU-only checks for the additive Qwen3.5 export registry."""

from pathlib import Path
import re
import unittest

DEVICE = Path(__file__).resolve().parents[1]
BUILD = DEVICE / "build.sh"
MAIN = DEVICE / "src/main.rs"
REGISTRY = DEVICE / "qwen35_exports.txt"

QWEN35_MODULES = ("packed_mean_bf16", "qwen35_norm", "qwen35_attention",
                  "qwen35_copy")


def kernel_pattern(name):
    return re.compile(r"#\[kernel\]\s*pub unsafe fn " + re.escape(name) + r"\s*\(")


def registered_exports():
    names = []
    for block in re.findall(
            r"symbols\s*=\s*sorted\((?:symbols\s*\+\s*)?\[(.*?)\]\)",
            BUILD.read_text(), re.S):
        names.extend(re.findall(r'"([^"]+)"', block))
    for path in sorted(DEVICE.glob("*_exports.txt")):
        names.extend(line.strip() for line in path.read_text().splitlines()
                     if line.strip())
    return names


class Qwen35ExportRegistryTests(unittest.TestCase):
    def test_registry_entries_are_unique_and_have_kernels(self):
        entries = [line.strip() for line in REGISTRY.read_text().splitlines()
                   if line.strip()]
        self.assertEqual(len(entries), 9)
        self.assertEqual(len(set(entries)), len(entries))
        main = MAIN.read_text()
        for entry in entries:
            with self.subTest(entry=entry):
                self.assertIsNotNone(kernel_pattern(entry).search(main))

    def test_export_union_is_99_unique_kernels(self):
        names = registered_exports()
        self.assertEqual(len(names), 99)
        self.assertEqual(len(set(names)), len(names))
        main = MAIN.read_text()
        missing = [name for name in names
                   if kernel_pattern(name).search(main) is None]
        self.assertEqual(missing, [])

    def test_registry_sources_exist_and_are_declared(self):
        main = MAIN.read_text()
        for module in QWEN35_MODULES:
            with self.subTest(module=module):
                self.assertTrue((DEVICE / "src" / f"{module}.rs").is_file())
                self.assertRegex(main, rf"(?m)^mod {module};$")

    def test_registry_is_wired_into_manifest_and_snapshot(self):
        build = BUILD.read_text()
        self.assertIn('(root / "qwen35_exports.txt")', build)
        self.assertIn('manifest["sources"]["qwen35_exports.txt"]', build)
        self.assertIn('"qwen35_exports.txt"', build)
        self.assertIn('root.parent / "NOTICE.qwen35"', build)
        for module in QWEN35_MODULES:
            self.assertIn(f'src/{module}.rs', build)


if __name__ == "__main__":
    unittest.main()
