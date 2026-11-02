#!/usr/bin/env python3
"""Pure-CPU contract tests for the Qwen3.5 GDN AOT packing module.

These tests never create a CUDA context or run a CUDA kernel. They test the
layout helpers that are textually compiled from the actual .cu source with a
host C++ compiler, plus the compile-only manifest and the preserved raw-module
provenance records.

The C++ harness inside build.py compares the production helpers against
independent reference loops for identity, permuted, and duplicate source
gathers, B=0, and 64-bit offsets beyond 2**32. This file adds manifest,
architecture, source-contract, and raw-module preservation checks.
"""

import importlib.util
import json
import pathlib
import re
import tempfile
import unittest

TASK = pathlib.Path(__file__).resolve().parent
WORKTREE = pathlib.Path(
    "/tmp/infinity-oxide-qwen35-gdn-aot-worktree"
)
MODULE_DIR = (
    WORKTREE
    / "libs/infinity-v3/backends/oxide/device/native_gdn_aot"
)
SOURCE = MODULE_DIR / "qwen35_gdn_aot.cu"
BUNDLE = TASK / "kernels"


def load_build():
    spec = importlib.util.spec_from_file_location(
        "qwen35_gdn_aot_build", MODULE_DIR / "build.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IndexContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = load_build()

    def test_index_contract_harness_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.build.run_index_test(
                SOURCE, pathlib.Path(directory), "/usr/bin/c++"
            )
        self.assertEqual(result["output"], "index-contract checks passed")

    def test_architectures_must_be_exact(self):
        self.assertEqual(
            self.build.parse_architectures("sm_90;sm_100;sm_120"),
            ["sm_90", "sm_100", "sm_120"],
        )
        for bad in (
            "sm_90;sm_100",
            "sm_100;sm_90;sm_120",
            "sm_90;sm_100;sm_120;sm_120",
            "sm_80;sm_90;sm_100",
            "",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.build.parse_architectures(bad)

    def test_output_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "existing").write_text("x")
            with self.assertRaises(RuntimeError):
                self.build.prepare_output(root)

    def test_source_contract_is_half_split_and_bit_preserving(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("gate * value_heads + head", text)
        self.assertIn("__float2bfloat16_rn", text)
        self.assertIn("const uint32_t* source", text)
        self.assertIn("uint32_t* destination", text)
        self.assertIn("source_indices ? source_indices[batch_index] : batch_index", text)
        self.assertIn("2 * key_heads * kHeadWidth", text)
        for name in self.build.EXPORTS:
            self.assertIn(name, text)
        for name in self.build.RAW_MODULE_EXPORTS:
            self.assertNotIn(name, text)
        self.assertEqual(set(self.build.EXPORTS) & set(self.build.RAW_MODULE_EXPORTS), set())

    def test_manifest_matches_output_and_sources(self):
        if not (BUNDLE / "manifest.json").is_file():
            self.skipTest(f"bundle not built at {BUNDLE}")
        manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["abi_version"], 1)
        self.assertEqual(manifest["operator"], "qwen35_gdn_aot_pack")
        self.assertTrue(manifest["compile_only"])
        self.assertFalse(manifest["device_calls"])
        self.assertEqual(manifest["architectures"], ["sm_90", "sm_100", "sm_120"])
        self.assertEqual(tuple(manifest["exports"]), tuple(self.build.EXPORTS))
        self.assertEqual(tuple(manifest["flags"]), tuple(self.build.FLAGS))
        self.assertEqual(manifest["torch_symbols"], [])
        self.assertEqual(
            manifest["sources"]["qwen35_gdn_aot.cu"],
            self.build.sha256(SOURCE),
        )
        self.assertEqual(len(manifest["compilations"]), 3)
        for record in manifest["compilations"]:
            architecture = record["architecture"]
            cubin = BUNDLE / record["cubin"]
            self.assertTrue(cubin.is_file())
            self.assertEqual(cubin.stat().st_size, record["cubin_bytes"])
            self.assertEqual(self.build.sha256(cubin), record["cubin_sha256"])
            self.assertEqual(tuple(record["exports"]), tuple(self.build.EXPORTS))
            self.assertEqual(record["torch_symbols"], [])
            self.assertEqual(record["arch_number"], int(architecture.split("_")[1]))
            self.assertIn(f"-arch={architecture}", record["compile_command"])
            self.assertEqual(record["compile_command"][-1], str(SOURCE))
        self.assertEqual(
            set(manifest["abi"]),
            {"pack_bf16", "state_kv_to_vk", "state_vk_to_kv"},
        )
        self.assertIn(
            "alpha[0:value_heads) then beta[value_heads:2*value_heads)",
            manifest["abi"]["pack_bf16"]["arguments"][2][2],
        )

    def test_manifest_preserves_raw_module(self):
        if not (BUNDLE / "manifest.json").is_file():
            self.skipTest(f"bundle not built at {BUNDLE}")
        manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        raw = manifest["raw_module"]
        self.assertTrue(raw["untouched_by_this_build"])
        self.assertEqual(tuple(raw["exports_preserved"]), tuple(self.build.RAW_MODULE_EXPORTS))
        sibling = MODULE_DIR.parent / "native_gdn"
        for name, digest in raw["files_sha256"].items():
            self.assertEqual(self.build.sha256(sibling / name), digest)
        self.assertEqual(len(raw["exports_preserved"]), 8)
        # Old raw source is untouched and still has its distinct eight exports.
        raw_source = (sibling / "qwen35_gdn_native.cu").read_text(encoding="utf-8")
        self.assertIn("infinity_qwen35_gdn_cached_lanes8", raw_source)
        self.assertNotIn("infinity_qwen35_gdn_aot_pack_bf16", raw_source)

    def test_manifest_records_required_licenses(self):
        if not (BUNDLE / "manifest.json").is_file():
            self.skipTest(f"bundle not built at {BUNDLE}")
        manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
        for name in ("NOTICE", "LICENSE", "LICENSE.candle-MIT"):
            self.assertTrue((BUNDLE / name).is_file())
            self.assertIn(name, manifest["sources"])
            self.assertEqual(self.build.sha256(BUNDLE / name), manifest["sources"][name])
        notice = (BUNDLE / "NOTICE").read_text(encoding="utf-8")
        for required in (
            "Infinity Restricted MIT License 1.0",
            "qwen35_kernels.cu",
            "pack_gdn_inputs",
            "flashinfer_gdn.cpp",
            "qwen35_models.cpp",
            "LICENSE.candle-MIT",
        ):
            self.assertRegex(notice, re.escape(required))


if __name__ == "__main__":
    unittest.main()
