# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Unit tests for the public raw8/aux3 cold GDN bundle builder."""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_gdn as gdn


def make_elf(symbols, torch_symbols=(), helper=True):
    strings = bytearray(b"\0")

    def add(name):
        offset = len(strings)
        strings.extend(name.encode("utf-8") + b"\0")
        return offset

    entries = []
    for name in symbols:
        entries.append((add(name), (1 << 4) | 2, 0x10))
    if helper:
        entries.append((add("_Z12internal_helperv"), 2, 0))
    for name in torch_symbols:
        entries.append((add(name), (1 << 4) | 2, 0))
    symtab = bytearray(24)
    for name, info, other in entries:
        symtab.extend(struct.pack("<IBBHQQ", name, info, other, 1, 0, 0))
    shstr = b"\0.symtab\0.strtab\0.shstrtab\0"

    def aligned(value):
        return (value + 7) // 8 * 8

    strtab_offset = 64
    symtab_offset = aligned(strtab_offset + len(strings))
    shstr_offset = aligned(symtab_offset + len(symtab))
    section_offset = aligned(shstr_offset + len(shstr))
    image = bytearray(section_offset + 4 * 64)
    ident = b"\x7fELF" + bytes((2, 1, 1, 0)) + b"\0" * 8
    struct.pack_into("<16sHHIQQQIHHHHHH", image, 0, ident, 2, 190, 1, 0, 0,
                     section_offset, 0, 64, 0, 0, 64, 4, 3)
    image[strtab_offset:strtab_offset + len(strings)] = strings
    image[symtab_offset:symtab_offset + len(symtab)] = symtab
    image[shstr_offset:shstr_offset + len(shstr)] = shstr

    def section(index, name, kind, offset, size, link=0, size_align=1, entry=0):
        struct.pack_into("<IIQQQQIIQQ", image, section_offset + index * 64, name, kind,
                         0, 0, offset, size, link, 1, size_align, entry)

    section(1, 9, 3, strtab_offset, len(strings))
    section(2, 1, 2, symtab_offset, len(symtab), link=1, size_align=8, entry=24)
    section(3, 17, 3, shstr_offset, len(shstr))
    return bytes(image)


class BuilderTests(unittest.TestCase):
    def test_raw_retains_full_v3_license_without_changing_three_source_abi(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root)/'raw', pathlib.Path(root)/'aux'
            with mock.patch.object(gdn.subprocess, 'run', side_effect=self._runner([], [])):
                self.assertEqual(gdn.main(['--raw-output', str(raw), '--aux-output', str(aux)]), 0)
            manifest = json.loads((raw/'manifest.json').read_text())
            self.assertEqual(set(manifest['sources']), {'qwen35_gdn_native.cu', 'NOTICE', 'LICENSE.candle-MIT'})
            self.assertEqual((raw/'LICENSE').read_bytes(), gdn.REPO_LICENSE.read_bytes())
            self.assertEqual(manifest['additional_licenses']['LICENSE'], hashlib.sha256((raw/'LICENSE').read_bytes()).hexdigest())

    def test_nested_torch_namespaces_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as root:
            cubin = pathlib.Path(root)/'bad.cubin'
            for name in ('_ZN5other4callEN2at6TensorE', '_ZN5other4callEN3c1010TensorImplE'):
                cubin.write_bytes(make_elf(gdn.RAW_EXPORTS, (name,)))
                with self.assertRaisesRegex(gdn.BuildError, 'Torch'): gdn._validate_cubin(cubin, gdn.RAW_EXPORTS, 'raw')

    def test_cleanup_failure_is_reported_with_primary_error_and_residual(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root)/'raw', pathlib.Path(root)/'aux'
            replace = gdn.os.replace; rmtree = gdn.shutil.rmtree
            def publish(source, dest):
                if pathlib.Path(dest) == aux: raise OSError('aux-primary-failure')
                return replace(source, dest)
            def cleanup(path, *args, **kwargs):
                if pathlib.Path(path) == raw: raise PermissionError('rollback-denied')
                return rmtree(path, *args, **kwargs)
            err = io.StringIO()
            with mock.patch.object(gdn.subprocess, 'run', side_effect=self._runner([], [])), \
                 mock.patch.object(gdn.os, 'replace', side_effect=publish), \
                 mock.patch.object(gdn.shutil, 'rmtree', side_effect=cleanup), contextlib.redirect_stderr(err):
                self.assertEqual(gdn.main(['--raw-output', str(raw), '--aux-output', str(aux)]), 2)
            self.assertTrue(raw.exists()); self.assertFalse(aux.exists())
            self.assertIn('aux-primary-failure', err.getvalue()); self.assertIn('cleanup incomplete', err.getvalue())
            self.assertIn(str(raw), err.getvalue()); self.assertIn('rollback-denied', err.getvalue())

    def test_malformed_elf_never_publishes_a_bundle(self):
        base = bytearray(make_elf(gdn.RAW_EXPORTS))
        shoff = struct.unpack_from('<Q', base, 40)[0]
        symoff = struct.unpack_from('<Q', base, shoff+2*64+24)[0]
        stroff, strsize = struct.unpack_from('<QQ', base, shoff+64+24)
        mutations = []
        mutations.append(b'\x7fELF')
        bad = bytearray(base); struct.pack_into('<Q', bad, 40, len(base)*2); mutations.append(bad)
        bad = bytearray(base); struct.pack_into('<Q', bad, shoff+2*64+56, 23); mutations.append(bad)
        bad = bytearray(base); struct.pack_into('<Q', bad, shoff+64+32, len(base)*2); mutations.append(bad)
        bad = bytearray(base); struct.pack_into('<I', bad, symoff+24, len(base)*2); mutations.append(bad)
        bad = bytearray(base); bad[stroff+strsize-1] = 1; mutations.append(bad)
        for index, image in enumerate(mutations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as root:
                raw, aux = pathlib.Path(root)/'raw', pathlib.Path(root)/'aux'
                def run(argv, **kwargs):
                    if argv[1] == '--version': return subprocess.CompletedProcess(argv, 0, 'release13.1', '')
                    pathlib.Path(argv[-1]).write_bytes(image)
                    return subprocess.CompletedProcess(argv, 0, '', '')
                with mock.patch.object(gdn.subprocess, 'run', side_effect=run), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(gdn.main(['--raw-output', str(raw), '--aux-output', str(aux)]), 2)
                self._assert_no_outputs(root, raw, aux)

    def test_malformed_path_is_actionable_before_compiler(self):
        err = io.StringIO()
        with mock.patch.object(gdn.subprocess, 'run') as run, contextlib.redirect_stderr(err):
            self.assertEqual(gdn.main(['--raw-output', 'bad\0path', '--aux-output', 'unused']), 2)
        run.assert_not_called(); self.assertIn('error:', err.getvalue())

    def test_same_and_nested_path_aliases_reject_before_compiler(self):
        with tempfile.TemporaryDirectory() as root:
            base = pathlib.Path(root)
            for left, right in ((base / 'raw', base / 'unused' / '..' / 'raw'),
                                (base / 'raw', base / 'raw' / 'nested')):
                with mock.patch.object(gdn.subprocess, 'run') as run:
                    self.assertEqual(gdn.main(['--raw-output', str(left), '--aux-output', str(right)]), 2)
                    run.assert_not_called()

    def test_cuda_entrypoint_tags_and_machine_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            cubin = pathlib.Path(root) / 'kernels.cubin'
            image = bytearray(make_elf(gdn.RAW_EXPORTS))
            struct.pack_into('<H', image, 18, 62)
            cubin.write_bytes(image)
            with self.assertRaises(gdn.BuildError): gdn._validate_cubin(cubin, gdn.RAW_EXPORTS, 'raw')
            image = bytearray(make_elf(gdn.RAW_EXPORTS))
            shoff = struct.unpack_from('<Q', image, 40)[0]
            symoff = struct.unpack_from('<Q', image, shoff + 2 * 64 + 24)[0]
            # Clearing the entry flag or replacing STT_FUNC with the old fake
            # type10 cannot let a host/helper symbol stand in for a kernel.
            for change in ('other', 'type'):
                bad = bytearray(image)
                bad[symoff + 24 + (5 if change == 'other' else 4)] = 0 if change == 'other' else 0x1a
                cubin.write_bytes(bad)
                with self.assertRaises(gdn.BuildError): gdn._validate_cubin(cubin, gdn.RAW_EXPORTS, 'raw')

    def _runner(self, calls, compiled, symbols=None, mutate=None):
        def run(argv, **kwargs):
            calls.append((list(argv), kwargs))
            if argv[1] == "--version":
                return subprocess.CompletedProcess(
                    argv, 0, "Cuda compilation tools, release 12.8, V12.8.61\n", "")
            architecture = next(item.split("=", 1)[1]
                                for item in argv if item.startswith("-arch="))
            source = pathlib.Path(argv[-3])
            cubin = pathlib.Path(argv[-1])
            self.assertEqual(cubin, source.parent / architecture / "kernels.cubin")
            kind = "aux" if source.name == gdn.AUX_SOURCE_NAME else "raw"
            exports = gdn.AUX_EXPORTS if kind == "aux" else gdn.RAW_EXPORTS
            if symbols is not None:
                exports = symbols(kind, architecture)
            compiled.append((kind, source.name,
                             hashlib.sha256(source.read_bytes()).hexdigest()))
            cubin.write_bytes(make_elf(exports))
            if mutate is not None:
                mutate(source)
            return subprocess.CompletedProcess(argv, 0, "", "")
        return run

    def _assert_no_outputs(self, root, raw, aux):
        self.assertFalse(os.path.lexists(raw))
        self.assertFalse(os.path.lexists(aux))
        self.assertEqual(list(pathlib.Path(root).iterdir()), [])

    def test_synthetic_elf_registers_only_cuda_kernels(self):
        with tempfile.TemporaryDirectory() as root:
            cubin = pathlib.Path(root) / "kernels.cubin"
            cubin.write_bytes(make_elf(gdn.RAW_EXPORTS, ("torch_add",), helper=True))
            kernels, torch = gdn._elf_kernels(cubin)
        self.assertEqual(kernels, list(gdn.RAW_EXPORTS))
        self.assertEqual(torch, ["torch_add"])

    def test_builds_both_bundles_with_exact_commands_and_manifests(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
            calls, compiled = [], []
            with mock.patch.object(gdn.subprocess, "run",
                                   side_effect=self._runner(calls, compiled)):
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "/opt/cuda/bin/nvcc"])
            self.assertEqual(result, 0)
            versions = [call for call in calls if call[0][1] == "--version"]
            compiles = [call for call in calls if call[0][1] != "--version"]
            self.assertEqual(versions[0][0], ["/opt/cuda/bin/nvcc", "--version"])
            self.assertEqual(compiles[0][1], {
                "capture_output": True, "text": True, "check": False})
            self.assertEqual(len(compiles), 6)
            expected_order = [("raw", "qwen35_gdn_native.cu", arch)
                              for arch in gdn.ARCHITECTURES]
            expected_order += [("aux", "qwen35_gdn_aot.cu", arch)
                               for arch in gdn.ARCHITECTURES]
            actual_order = []
            for (argv, _), (kind, source, _) in zip(compiles, compiled):
                architecture = next(item.split("=", 1)[1]
                                    for item in argv if item.startswith("-arch="))
                self.assertEqual(argv[0], "/opt/cuda/bin/nvcc")
                self.assertEqual(argv[1:10], list(gdn.REQUIRED_FLAGS))
                self.assertEqual(pathlib.Path(argv[-3]).name, source)
                actual_order.append((kind, "qwen35_gdn_native.cu" if kind == "raw"
                                     else "qwen35_gdn_aot.cu", architecture))
                self.assertEqual(actual_order[-1][2], architecture)
            self.assertEqual(actual_order, expected_order)

            for kind, destination, exports, operator in (
                    ("raw", raw, gdn.RAW_EXPORTS, "qwen35_gdn_native"),
                    ("aux", aux, gdn.AUX_EXPORTS, "qwen35_gdn_aot_pack")):
                manifest = json.loads((destination / "manifest.json").read_text())
                self.assertEqual(manifest["abi_version"], 1)
                self.assertEqual(manifest["operator"], operator)
                self.assertTrue(manifest["compile_only"])
                self.assertFalse(manifest["device_calls"])
                self.assertEqual(manifest["architectures"], list(gdn.ARCHITECTURES))
                self.assertEqual(manifest["exports"], list(exports))
                self.assertEqual(manifest["flags"], list(gdn.REQUIRED_FLAGS))
                self.assertEqual(manifest["toolkit"], {"version": "12.8"})
                self.assertEqual(manifest["torch_symbols"], [])
                for name, digest in manifest["sources"].items():
                    self.assertEqual(
                        hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
                    compiled_hashes = [value for got_kind, source, value in compiled
                                       if got_kind == kind and source == name]
                    if compiled_hashes:
                        self.assertEqual(compiled_hashes[0], digest)
                self.assertEqual(len(manifest["compilations"]), 3)
                for record, architecture in zip(manifest["compilations"],
                                                gdn.ARCHITECTURES):
                    self.assertEqual(record["architecture"], architecture)
                    self.assertEqual(record["cubin"], f"{architecture}/kernels.cubin")
                    cubin = destination / record["cubin"]
                    self.assertEqual(record["cubin_bytes"], cubin.stat().st_size)
                    self.assertGreater(record["cubin_bytes"], 0)
                    self.assertEqual(record["cubin_sha256"],
                                     hashlib.sha256(cubin.read_bytes()).hexdigest())
                    self.assertEqual(record["exports"], list(exports))
                    self.assertEqual(record["torch_symbols"], [])
                expected_sources = ({name for name, _ in gdn._raw_sources()} if kind == "raw"
                                    else {name for name, _ in gdn._aux_sources()})
                self.assertEqual(set(manifest["sources"]), expected_sources)

    def test_missing_license_fails_before_compiler(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
            missing = pathlib.Path(root) / "missing-LICENSE"
            with mock.patch.object(gdn, "REPO_LICENSE", missing), \
                    mock.patch.object(gdn.subprocess, "run") as run:
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "nvcc"])
            run.assert_not_called()
            self.assertNotEqual(result, 0)
            self._assert_no_outputs(root, raw, aux)

    def test_missing_extra_and_torch_symbols_fail_without_output(self):
        for label, symbols in (
                ("missing", lambda kind, arch: gdn.RAW_EXPORTS[:-1]),
                ("extra", lambda kind, arch: gdn.RAW_EXPORTS + ("extra_kernel",)),
                ("torch", lambda kind, arch: gdn.RAW_EXPORTS)):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
                calls, compiled = [], []

                def runner(argv, **kwargs):
                    if argv[1] == "--version":
                        return subprocess.CompletedProcess(argv, 0, "release 12.8", "")
                    source = pathlib.Path(argv[-3])
                    exports = symbols("raw", "sm_90")
                    torch_symbols = ("torch_add",) if label == "torch" else ()
                    pathlib.Path(argv[-1]).write_bytes(make_elf(exports, torch_symbols))
                    return subprocess.CompletedProcess(argv, 0, "", "")

                stderr = io.StringIO()
                with mock.patch.object(gdn.subprocess, "run", side_effect=runner), \
                        contextlib.redirect_stderr(stderr):
                    result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                        "--nvcc", "nvcc"])
                self.assertNotEqual(result, 0)
                self.assertIn("error:", stderr.getvalue())
                if label == "torch":
                    self.assertIn("Torch", stderr.getvalue())
                else:
                    self.assertIn("kernel symbols do not match", stderr.getvalue())
                self._assert_no_outputs(root, raw, aux)

    def test_failed_compile_leaves_no_output_or_staging(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"

            def runner(argv, **kwargs):
                if argv[1] == "--version":
                    return subprocess.CompletedProcess(argv, 0, "release 12.8", "")
                return subprocess.CompletedProcess(argv, 1, "", "nvcc fatal: synthetic")

            with mock.patch.object(gdn.subprocess, "run", side_effect=runner):
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "nvcc"])
            self.assertNotEqual(result, 0)
            self._assert_no_outputs(root, raw, aux)

    def test_success_without_cubin_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"

            def runner(argv, **kwargs):
                if argv[1] == "--version":
                    return subprocess.CompletedProcess(argv, 0, "release 12.8", "")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with mock.patch.object(gdn.subprocess, "run", side_effect=runner):
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "nvcc"])
            self.assertNotEqual(result, 0)
            self._assert_no_outputs(root, raw, aux)

    def test_existing_and_symlink_outputs_are_rejected_before_compiler(self):
        for label in ("directory", "file", "symlink"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
                if label == "directory":
                    raw.mkdir()
                elif label == "file":
                    raw.write_text("occupied")
                else:
                    target = pathlib.Path(root) / "target"
                    target.write_text("occupied")
                    os.symlink(target, raw)
                with mock.patch.object(gdn.subprocess, "run") as run:
                    result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                        "--nvcc", "nvcc"])
                run.assert_not_called()
                self.assertNotEqual(result, 0)
                self.assertTrue(os.path.lexists(raw))
                self.assertFalse(aux.exists())

    def test_source_change_after_snapshot_invalidates_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
            calls, compiled = [], []
            with mock.patch.object(
                    gdn.subprocess, "run",
                    side_effect=self._runner(calls, compiled,
                                             mutate=lambda path: path.write_text("changed"))):
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "nvcc"])
            self.assertNotEqual(result, 0)
            self._assert_no_outputs(root, raw, aux)

    def test_aux_publish_failure_rolls_back_raw_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            raw, aux = pathlib.Path(root) / "raw", pathlib.Path(root) / "aux"
            calls, compiled = [], []
            real_replace = os.replace

            def replace(source, destination):
                if pathlib.Path(destination) == aux:
                    raise OSError("synthetic publish failure")
                return real_replace(source, destination)

            with mock.patch.object(gdn.subprocess, "run",
                                   side_effect=self._runner(calls, compiled)), \
                    mock.patch.object(gdn.os, "replace", side_effect=replace):
                result = gdn.main(["--raw-output", str(raw), "--aux-output", str(aux),
                                    "--nvcc", "nvcc"])
            self.assertNotEqual(result, 0)
            self._assert_no_outputs(root, raw, aux)

    def test_nvcc_defaults_to_toolkit_then_path(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gdn._default_nvcc(None), "nvcc")
        with mock.patch.dict(os.environ, {"CUDA_TOOLKIT_PATH": "/cuda"}, clear=True):
            self.assertEqual(gdn._default_nvcc(None), "/cuda/bin/nvcc")
        self.assertEqual(gdn._default_nvcc("/custom/nvcc"), "/custom/nvcc")


if __name__ == "__main__":
    unittest.main()
