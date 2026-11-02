#!/usr/bin/env python3
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""CPU-only guards for the development-only selected-symbol FA4 export.

No CUDA, torch, cutlass, network, or real compile is used. Subprocess checks
stop before heavy imports; the mocked tests replace the compiler/torch boundary
and execute build.py's real argument, filtering, validation, and manifest code.
"""
import json
import os
import pathlib
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

VENDOR = pathlib.Path(__file__).resolve().parent
BUILD = VENDOR / "build.py"
sys.path.insert(0, str(VENDOR))
import build_contract  # noqa: E402
import portable_host  # noqa: E402


def run_build(args, extra_env=None):
    env = {name: value for name, value in os.environ.items()
           if name not in ("CUTE_DSL_PTXAS_PATH", "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(BUILD), *[str(value) for value in args]],
        cwd=VENDOR, env=env, text=True, capture_output=True)


class GuardTests(unittest.TestCase):
    def test_requires_compile_export_and_only_symbol_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            for index, args in enumerate((
                    ["--compile-only", "--only-symbol", "fa4_bert_bf16"],
                    ["--export-only", "--only-symbol", "fa4_bert_bf16"],
                    ["--only-symbol", "fa4_bert_bf16"],
                    ["--link-only", "--only-symbol", "fa4_bert_bf16"],
            )):
                output = pathlib.Path(temporary) / f"out-{index}"
                result = run_build([output, "--arch", "sm_90a", *args])
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("--only-symbol requires --compile-only and --export-only",
                              result.stderr)
                self.assertFalse(output.exists(), "refusal must happen before mkdir")

    def test_unknown_symbol_fails_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            sentinel = root / "torch-imported"
            fake = root / "fake"
            fake.mkdir()
            (fake / "torch.py").write_text(
                "import os, pathlib\n"
                "pathlib.Path(os.environ['TORCH_SENTINEL']).write_text('imported')\n"
                "raise ImportError('blocked heavy torch import')\n")
            result = run_build(
                [output, "--arch", "sm_90a", "--compile-only", "--export-only",
                 "--only-symbol", "fa4_nope"],
                {"PYTHONPATH": str(fake), "TORCH_SENTINEL": str(sentinel)})
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unknown --only-symbol", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(sentinel.exists(), "torch was imported before validation")

    def test_existing_canonical_bundle_is_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            sentinel = root / "torch-imported"
            for index, canonical_name in enumerate(("manifest.json", "libinfinityfa4.so")):
                with self.subTest(canonical=canonical_name):
                    output = root / f"out-{index}"
                    output.mkdir()
                    canonical = output / canonical_name
                    canonical.write_text('{"name": "InfinityFA4"}\n')
                    keep = output / "keep.txt"
                    keep.write_text("unrelated\n")
                    result = run_build(
                        [output, "--arch", "sm_90a", "--compile-only", "--export-only",
                         "--only-symbol", "fa4_bert_bf16"],
                        {"TORCH_SENTINEL": str(sentinel)})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(
                        "refusing development-only export into an existing canonical bundle",
                        result.stderr)
                    self.assertEqual(canonical.read_text(),
                                     '{"name": "InfinityFA4"}\n')
                    self.assertEqual(keep.read_text(), "unrelated\n")
                    self.assertFalse((output / "development-manifest.json").exists())
            self.assertFalse(sentinel.exists())

    def test_failed_probe_removes_stale_dev_manifest_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "development-manifest.json").write_text('{"stale": true}\n')
            keep = output / "keep.txt"
            keep.write_text("unrelated\n")
            result = run_build(
                [output, "--arch", "sm_90a", "--compile-only", "--export-only",
                 "--only-symbol", "fa4_bert_bf16"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical26 export requires --runtime-dir and --ffi-root",
                          result.stderr)
            self.assertFalse((output / "development-manifest.json").exists())
            self.assertFalse((output / "manifest.json").exists())
            self.assertEqual(keep.read_text(), "unrelated\n")

    def test_link_only_rejects_development_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "development-manifest.json").write_text("{}\n")
            result = run_build(
                [output, "--arch", "sm_90a", "--link-only",
                 "--runtime-dir", root / "runtime", "--ffi-root", root / "ffi"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing --link-only on a development-only export",
                          result.stderr)

    def test_multiple_symbols_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_build(
                [pathlib.Path(temporary) / "out", "--arch", "sm_90a",
                 "--compile-only", "--export-only", "--only-symbol",
                 "fa4_bert_bf16", "fa4_bert_fp16"])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("canonical26 export requires --runtime-dir and --ffi-root",
                          result.stderr)
            self.assertNotIn("unknown --only-symbol", result.stderr)

    def test_helper_symbols_use_existing_flags(self):
        cases = (
            (["--deberta"], "fa4_deberta_bf16"),
            (["--deberta", "--deberta-exact-mul"], "fa4_deberta_exact_mul_both_bf16"),
            (["--alibi"], "fa4_alibi_model_bias_fp16"),
            (["--jinacode"], "fa4_jinacode_flash_alibi_bf16"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for index, (flags, symbol) in enumerate(cases):
                rejected = run_build(
                    [root / f"no-flag-{index}", "--arch", "sm_90a",
                     "--compile-only", "--export-only", "--only-symbol", symbol])
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("unknown --only-symbol", rejected.stderr)

                accepted = run_build(
                    [root / f"flagged-{index}", "--arch", "sm_90a",
                     "--compile-only", "--export-only", "--only-symbol", symbol,
                     *flags])
                self.assertNotEqual(accepted.returncode, 0)
                self.assertIn("canonical26 export requires --runtime-dir and --ffi-root",
                              accepted.stderr)
                self.assertNotIn("unknown --only-symbol", accepted.stderr)


def run_mocked_export(root, args, validate_side_effect=None,
                      export_failure_symbol=None):
    stage = root / "stage"
    interface_path = stage / "flash_attn" / "cute" / "interface.py"
    interface_path.parent.mkdir(parents=True)
    interface_path.write_text("# staged interface\n")

    state = {"calls": []}
    exported = []

    class FakeTensor:
        def __getitem__(self, item):
            return self

        def view(self, *shape):
            return self

    class Cache:
        def __init__(self):
            self.cache = {}

    class FakeFwd:
        def __init__(self):
            self.compile_cache = Cache()

        def __call__(self, *args, **kwargs):
            self.compile_cache.cache["entry"] = object()
            state["calls"].append({"engine_is_none": True,
                                   "real_IR_present": True})

    class FakeInterface:
        def __init__(self):
            self._flash_attn_fwd = FakeFwd()
            self.cute = types.SimpleNamespace(compile=None)

        def flash_attn_varlen_func(self, *args, **kwargs):
            self._flash_attn_fwd.compile_cache.cache["entry"] = object()
            state["calls"].append({"engine_is_none": True,
                                   "real_IR_present": True})

    class FakeTensorMode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_interface = FakeInterface()
    fake_interface.__file__ = str(interface_path)
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "fp16"
    fake_torch.bfloat16 = "bf16"
    fake_torch.int32 = "int32"
    fake_torch.float32 = "f32"
    fake_torch.zeros = lambda *args, **kwargs: FakeTensor()
    fake_torch.tensor = lambda *args, **kwargs: object()
    fake_torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
    fake_subclasses = types.ModuleType("torch._subclasses")
    fake_subclasses.__path__ = []
    fake_fake_tensor = types.ModuleType("torch._subclasses.fake_tensor")
    fake_fake_tensor.FakeTensorMode = FakeTensorMode
    fake_subclasses.fake_tensor = fake_fake_tensor
    fake_torch._subclasses = fake_subclasses

    fake_flash_attn = types.ModuleType("flash_attn")
    fake_flash_attn.__path__ = []
    fake_cute = types.ModuleType("flash_attn.cute")
    fake_cute.__path__ = []
    fake_cute.interface = fake_interface
    fake_flash_attn.cute = fake_cute
    fake_format = types.ModuleType("flash_attn.cute.interface")
    fake_format.__file__ = str(interface_path)

    lock = json.loads((VENDOR / "fa4_source_lock.json").read_text())
    fake_fa4_source = types.ModuleType("fa4_source")
    fake_fa4_source.stage_fa4_sources = lambda: (
        types.SimpleNamespace(name=str(stage)), lock["softmax_policy"])

    fake_compile_only = types.ModuleType("compile_only")
    fake_compile_only.install_export_only_compile = lambda cute: state

    def score_module(name, **values):
        module = types.ModuleType(name)
        for key, value in values.items():
            setattr(module, key, value)
        return module

    fake_deberta = score_module("deberta_score",
                                deberta_score=lambda *a, **k: None)
    fake_deberta_mul = score_module(
        "deberta_mul_score",
        deberta_exact_mul_both=lambda *a, **k: None,
        deberta_exact_mul_single=lambda *a, **k: None)
    fake_alibi = score_module(
        "alibi_score",
        alibi_model_bias_bf16=lambda *a, **k: None,
        alibi_model_bias_fp16=lambda *a, **k: None)
    fake_jinacode = score_module(
        "alibi12_native_score",
        alibi12_native_flash_bf16=lambda *a, **k: None,
        alibi12_native_flash_fp16=lambda *a, **k: None)

    modules = {
        "torch": fake_torch,
        "torch._subclasses": fake_subclasses,
        "torch._subclasses.fake_tensor": fake_fake_tensor,
        "flash_attn": fake_flash_attn,
        "flash_attn.cute": fake_cute,
        "flash_attn.cute.interface": fake_format,
        "fa4_source": fake_fa4_source,
        "compile_only": fake_compile_only,
        "deberta_score": fake_deberta,
        "deberta_mul_score": fake_deberta_mul,
        "alibi_score": fake_alibi,
        "alibi12_native_score": fake_jinacode,
    }

    def fake_export(compiled, path, symbol):
        if symbol == export_failure_symbol:
            raise RuntimeError(f"mocked export failure for {symbol}")
        path.write_bytes(symbol.encode())
        exported.append(symbol)

    env = {"CUDA_VISIBLE_DEVICES": "",
           "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0"}
    old_argv = sys.argv
    try:
        sys.argv = [str(BUILD)] + [str(value) for value in args]
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.dict(sys.modules, modules), \
                mock.patch.object(build_contract, "verify_source_identity",
                                  side_effect=lambda staged, capability, lock:
                                  {"files": 1, "capability": capability}), \
                mock.patch.object(build_contract, "runtime_pins",
                                  return_value={"libtvm_ffi.so": "a" * 64,
                                                "libcute_dsl_runtime.so": "b" * 64}), \
                mock.patch.object(portable_host, "export_portable",
                                  side_effect=fake_export), \
                mock.patch.object(portable_host, "validate_host_object",
                                  side_effect=validate_side_effect), \
                mock.patch("importlib.metadata.version", return_value="4.7.1"):
            try:
                runpy.run_path(str(BUILD), run_name="__main__")
            except SystemExit as error:
                return exported, state, error.code
            return exported, state, None
    finally:
        sys.argv = old_argv


class MockedCompileTests(unittest.TestCase):
    def test_selected_symbol_compiles_only_its_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "keep.txt").write_text("unrelated\n")
            exported, state, code = run_mocked_export(root, [
                output, "--arch", "sm_90a", "--compile-only", "--export-only",
                "--only-symbol", "fa4_bert_bf16",
                "--runtime-dir", root / "runtime", "--ffi-root", root / "ffi"])
            self.assertEqual(code, 0)
            self.assertEqual(exported, ["fa4_bert_bf16"])
            self.assertEqual(len(state["calls"]), 1)
            self.assertTrue((output / "fa4_bert_bf16.o").exists())
            self.assertFalse((output / "fa4_bert_fp16.o").exists())
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "libinfinityfa4.so").exists())
            self.assertEqual((output / "keep.txt").read_text(), "unrelated\n")

            development = json.loads((output / "development-manifest.json").read_text())
            self.assertEqual(development["kind"], "development-only")
            self.assertIs(development["development_only"], True)
            self.assertIs(development["production_admissible"], False)
            self.assertEqual(development["selected_symbols"], ["fa4_bert_bf16"])
            self.assertEqual(development["descriptors"][0]["canonical_mode"], 0)
            self.assertEqual(development["descriptors"][0]["object"], "fa4_bert_bf16.o")
            self.assertEqual(development["compile_cache_isolation"]["exports"], 1)
            for production_field in ("name", "native_abi_version", "abi_version",
                                     "registry", "canonical26_inventory",
                                     "link_object_order", "libraries"):
                self.assertNotIn(production_field, development)

    def test_nonportable_selected_object_leaves_no_development_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()

            def reject(path):
                raise RuntimeError(f"Nonportable AVX host instructions in {path}")

            with self.assertRaises(RuntimeError):
                run_mocked_export(root, [
                    output, "--arch", "sm_90a", "--compile-only", "--export-only",
                    "--only-symbol", "fa4_bert_bf16",
                    "--runtime-dir", root / "runtime", "--ffi-root", root / "ffi"],
                    validate_side_effect=reject)
            self.assertTrue((output / "fa4_bert_bf16.o").exists())
            self.assertFalse((output / "development-manifest.json").exists())
            self.assertFalse((output / "manifest.json").exists())

    def test_canonical_full_export_clears_stale_dev_marker_after_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "development-manifest.json").write_text("{}\n")
            exported, state, code = run_mocked_export(root, [
                output, "--arch", "sm_90a", "--compile-only", "--export-only",
                "--deberta", "--deberta-exact-mul", "--alibi", "--jinacode",
                "--runtime-dir", root / "runtime", "--ffi-root", root / "ffi"])
            self.assertEqual(code, 0)
            self.assertEqual(len(exported), 60)
            self.assertEqual(len(state["calls"]), 60)
            self.assertTrue((output / "manifest.json").exists())
            self.assertFalse((output / "development-manifest.json").exists())

    def test_canonical_full_export_keeps_stale_dev_marker_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            output = root / "out"
            output.mkdir()
            (output / "development-manifest.json").write_text("{}\n")
            with self.assertRaises(RuntimeError):
                run_mocked_export(root, [
                    output, "--arch", "sm_90a", "--compile-only", "--export-only",
                    "--deberta", "--deberta-exact-mul", "--alibi", "--jinacode",
                    "--runtime-dir", root / "runtime", "--ffi-root", root / "ffi"],
                    export_failure_symbol="fa4_bert_bf16")
            self.assertTrue((output / "development-manifest.json").exists())
            self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
