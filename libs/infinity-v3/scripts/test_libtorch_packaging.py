# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Runtime packaging regressions; requires a C compiler and readelf, not CUDA."""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "packager", Path(__file__).with_name("package-libtorch-runtime.py"))
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sdk = self.root / "sdk"
        self.lib = self.sdk / "torch/lib"
        self.lib.mkdir(parents=True)
        self.compile("libinner.so", "int inner(void) { return 7; }")
        self.compile("libouter.so", "extern int inner(void); int outer(void) { return inner(); }",
                     "-L" + str(self.lib), "-linner")
        self.server = self.compile(
            "server.so", "extern int outer(void); int serve(void) { return outer(); }",
            "-L" + str(self.lib), "-louter")
        for name in ("libcudnn.so.9", "libcudnn_engines_runtime_compiled.so.9",
                     "libcaffe2_nvrtc.so", "libtorch_cuda_linalg.so", "libnvrtc-builtins.so.13.0",
                     "libnvvm.so.4", "libnvrtc.alt.so.13", "libtorch_python.so"):
            shutil.copy2(self.lib / "libinner.so", self.lib / name)

    def compile(self, name, source, *link):
        output = self.lib / name
        subprocess.run(["cc", "-x", "c", "-", "-x", "none", "-shared", "-fPIC",
                        "-Wl,-soname," + name, "-o", str(output), *link],
                       input=source, text=True, check=True, capture_output=True)
        return output

    def test_linked_and_lazy_libraries_only(self):
        selected = {path.name for path in packager.runtime_libraries([self.server], self.sdk)}
        self.assertEqual(selected, {
            "libinner.so", "libouter.so", "libcudnn.so.9",
            "libcudnn_engines_runtime_compiled.so.9", "libcaffe2_nvrtc.so",
            "libtorch_cuda_linalg.so", "libnvrtc-builtins.so.13.0",
        })

    def test_missing_dependency_is_not_hidden_by_sdk_stubs(self):
        stubs = self.sdk / "nvidia/stubs"
        stubs.mkdir(parents=True)
        (self.lib / "libinner.so").rename(stubs / "libinner.so")
        with self.assertRaisesRegex(RuntimeError, "Unresolved runtime dependency libinner.so"):
            packager.runtime_libraries([self.server], self.sdk)

    def test_missing_lazy_library_fails(self):
        (self.lib / "libnvrtc-builtins.so.13.0").unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing runtime-loaded SDK library"):
            packager.runtime_libraries([self.server], self.sdk)

    def test_duplicate_names_must_have_identical_contents(self):
        other = self.sdk / "nvidia/lib"
        other.mkdir(parents=True)
        duplicate = other / "libinner.so"
        shutil.copy2(self.lib / "libinner.so", duplicate)
        packager.runtime_libraries([self.server], self.sdk)
        duplicate.write_bytes(b"conflicting SDK version")
        with self.assertRaisesRegex(RuntimeError, "Conflicting runtime libraries: libinner.so"):
            packager.runtime_libraries([self.server], self.sdk)

    def test_driver_is_external_but_python_is_not(self):
        driver = self.compile("libcuda.so.1", "int driver(void) { return 1; }")
        caller = self.compile("driver-user.so",
                              "extern int driver(void); int f(void) { return driver(); }",
                              str(driver))
        selected = packager.runtime_libraries([caller], self.sdk)
        self.assertNotIn("libcuda.so.1", {path.name for path in selected})
        caller = self.compile("python-user.so",
                              "extern int inner(void); int f(void) { return inner(); }",
                              str(self.compile("libtorch_python.so", "int inner(void) { return 1; }")))
        with self.assertRaisesRegex(RuntimeError, "Unresolved runtime dependency libtorch_python.so"):
            packager.runtime_libraries([caller], self.sdk)

    def bundles(self, different_runtime=False):
        root = self.root / "bundles"
        for sm in ("80", "90"):
            bundle = root / ("sm_" + sm)
            bundle.mkdir(parents=True)
            for name in ("libinfinityfa4.so", "libcute_dsl_runtime.so", "libtvm_ffi.so"):
                content = name + (sm if name == "libinfinityfa4.so" or different_runtime else "")
                (bundle / name).write_text(content)
            (bundle / "export.o").write_bytes(b"build-only")
            manifest = {
                "name": "InfinityFA4", "native_abi_version": 1,
                "architecture": "sm_" + sm + ("a" if sm == "90" else ""),
                "libraries": {path.name: packager.digest(path) for path in bundle.glob("*.so")},
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
        return root

    def test_shared_runtimes_survive_relocation(self):
        destination = self.root / "package"
        packager.package_fa4(self.bundles(), destination, {"80", "90a"})
        self.assertTrue((destination / "sm_90/libtvm_ffi.so").is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(destination / "sm_90/libtvm_ffi.so")))
        self.assertFalse((destination / "sm_90/libinfinityfa4.so").is_symlink())
        self.assertFalse(list(destination.rglob("*.o")))
        relocated = self.root / "relocated"
        shutil.copytree(destination, relocated, symlinks=True)
        shutil.rmtree(destination)
        for manifest_path in relocated.glob("*/manifest.json"):
            for name, expected in json.loads(manifest_path.read_text())["libraries"].items():
                self.assertEqual(packager.digest(manifest_path.parent / name), expected)

    def test_different_runtime_versions_are_not_deduplicated(self):
        destination = self.root / "package"
        packager.package_fa4(self.bundles(different_runtime=True), destination, {"80", "90"})
        self.assertFalse((destination / "sm_90/libtvm_ffi.so").is_symlink())

    def test_bundle_hashes_and_requested_architectures_are_checked(self):
        root = self.bundles()
        with self.assertRaisesRegex(RuntimeError, "Missing InfinityFA4 bundle"):
            packager.package_fa4(root, self.root / "missing", {"120"})
        (root / "sm_90/libtvm_ffi.so").write_text("changed after export")
        with self.assertRaisesRegex(RuntimeError, "does not match its manifest"):
            packager.package_fa4(root, self.root / "changed", {"90"})

    def test_optional_fa4_and_clean_output(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(packager.package_fa4(self.root / "absent", self.root / "out", {"90"}), [])
        with patch.dict(os.environ, {"INFINITY_FA4_ROOT": "/requested"}), \
                self.assertRaisesRegex(RuntimeError, "installed bundles are missing"):
            packager.package_fa4(self.root / "absent", self.root / "out", {"90"})
        with self.assertRaisesRegex(RuntimeError, "Runtime output must be empty"):
            packager.main(self.root / "unused.jsonl", self.sdk, self.lib)


if __name__ == "__main__":
    unittest.main()
