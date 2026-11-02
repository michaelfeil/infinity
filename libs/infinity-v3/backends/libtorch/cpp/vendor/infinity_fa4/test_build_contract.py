#!/usr/bin/env python3
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""CPU-only canonical-prefix contract guards for the shared FA4 producer."""
import hashlib
import math
import pathlib
import sys
import unittest

VENDOR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(VENDOR))
import build_contract as contract  # noqa: E402


def _helper_kernels():
    order = [
        "fa4_deberta_{dtype}",
        "fa4_alibi_model_bias_{dtype}",
        "fa4_jinacode_flash_alibi_{dtype}",
        "fa4_deberta_exact_mul_both_bf16",
        "fa4_deberta_exact_mul_single_bf16",
    ]
    kernels = []
    for pattern in order:
        if "{dtype}" in pattern:
            for dtype in contract.DTYPES:
                kernels.append(dict(contract.HELPER_CONTRACTS[pattern.format(dtype=dtype)]))
        else:
            kernels.append(dict(contract.HELPER_CONTRACTS[pattern]))
    return kernels


def manifest_for(count):
    rows = contract.canonical_rows_for(count)
    kernels = []
    for dtype in contract.DTYPES:
        for mode, (name, heads, kv_heads, d, causal, local) in enumerate(rows):
            window = contract.canonical_window(mode, name, causal, local)
            kernels.append(contract.profile_kernel(
                mode, name, heads, kv_heads, d, causal, window, dtype))
    kernels.extend(_helper_kernels())
    symbols = [kernel["symbol"] for kernel in kernels]
    names = [row[0] for row in rows]
    flags = {f"include_{family}": True for family in
             ("deberta", "deberta_exact_mul", "alibi", "jinacode")}
    link_order = [f"{symbol}.o" for symbol in
                  contract.expected_link_symbols(names, **flags)]
    return {
        "registry": contract.canonical_registry_for(count),
        "kernels": kernels,
        "canonical31_inventory": {"exact_order": True, "symbols": symbols},
        "link_object_order": link_order,
        "objects": {name: "0" * 64 for name in link_order},
    }


def key(mode, row):
    name, heads, kv_heads, d, causal, local = row
    g = math.gcd(heads, kv_heads)
    return ((heads // g, kv_heads // g), d, causal,
            contract.canonical_window(mode, name, causal, local))


class CanonicalPrefixTests(unittest.TestCase):
    def test_active_canonical38_registry_and_fingerprint_are_coherent(self):
        rows = contract.validate_profiles_inc(VENDOR / "profiles.inc")
        self.assertEqual(len(rows), 38)
        self.assertEqual(contract.CANONICAL38_REGISTRY["namespace"], "canonical38")
        self.assertEqual(contract.CANONICAL38_REGISTRY["mode_count"], 38)
        self.assertEqual(contract.CANONICAL38_REGISTRY["mode_mask"],
                         "0x3fffffffff")
        self.assertEqual(contract.PROFILE_SEQUENCE_SHA256,
                         "5308427d3c2f149dba100ef7235843df3693b11dd125ccb78288616bd7dcd99d")

    def test_legacy_and_extended_manifests_validate(self):
        for count in (19, 20, 25, 26, 31, 38):
            with self.subTest(count=count):
                manifest = manifest_for(count)
                inventory = contract.validate_manifest_inventory(manifest)
                self.assertEqual(inventory["order_kind"], "canonical")
                self.assertEqual(manifest["registry"]["mode_count"], count)
                expected_mask = f"{(1 << count) - 1:#x}"
                self.assertEqual(manifest["registry"]["mode_mask"], expected_mask)

    def test_bridge_exports_legacy_and_extended_mode_queries(self):
        bridge = (VENDOR / "bridge.cpp").read_text()
        self.assertIn("infinity_fa4_modes_v1()", bridge)
        self.assertIn("infinity_fa4_modes_v2()", bridge)
        self.assertIn("uint64_t infinity_fa4_modes_v2()", bridge)
        self.assertIn("attention_modes >= 32 ? -1", bridge)

    def test_all_dtypes_and_sm_targets_are_declared_from_one_source(self):
        self.assertEqual(contract.DTYPES, ("fp16", "bf16"))
        rows = contract.canonical_rows_for(38)
        names = [row[0] for row in rows]
        flags = {f"include_{family}": True for family in
                 ("deberta", "deberta_exact_mul", "alibi", "jinacode")}
        symbols = contract.expected_kernel_symbols(names, **flags)
        self.assertEqual(len(symbols), 38 * 2 + 8)
        for dtype in contract.DTYPES:
            for name in names:
                self.assertIn(f"fa4_{name}_{dtype}", symbols)
        build_all = (VENDOR / "build_all.py").read_text()
        self.assertIn("--architectures", build_all)
        self.assertIn("90;100;120", build_all)

    def test_legacy_mode_masks_are_unchanged(self):
        self.assertEqual(contract.canonical_registry_for(19)["mode_mask"], "0x7ffff")
        self.assertEqual(contract.canonical_registry_for(20)["mode_mask"], "0xfffff")
        self.assertEqual(contract.canonical_registry_for(25)["mode_mask"], "0x1ffffff")
        self.assertEqual(contract.CANONICAL26_REGISTRY["mode_mask"], "0x3ffffff")
        self.assertEqual(contract.CANONICAL31_REGISTRY["mode_mask"], "0x7fffffff")
        self.assertEqual(contract.canonical_registry_for(38)["mode_mask"],
                         "0x3fffffffff")

    def test_canonical38_keys_are_unique_after_real_shape_deduplication(self):
        rows = contract.canonical_rows_for(38)
        keys = [key(mode, row) for mode, row in enumerate(rows)]
        self.assertEqual(len(keys), len(set(keys)))
        by_name = {row[0]: row for row in rows}
        # Qwen3 16/8 d128 noncausal already shares published voyage geometry.
        mode_of = {row[0]: mode for mode, row in enumerate(rows)}
        self.assertEqual(key(3, ("qwen3_16kv8_d128_noncausal",
                                 16, 8, 128, False, False)),
                         key(mode_of["voyage"], by_name["voyage"]))
        # Rune 26B Q16/KV8 d256 local1024 deduplicates onto Gemma3 mode 35.
        rune_key = key(35, ("gemma3_local1024_d256_gqa2",
                            16, 8, 256, True, True))
        self.assertEqual(key(35, by_name["gemma3_local1024_d256_gqa2"]), rune_key)
        # Causal narrowing never equates the two d64 decoder families.
        self.assertNotEqual(key(31, by_name["qwen2_q14kv2_d64_causal"]),
                            key(33, by_name["llama_q32kv8_d64_noncausal"]))

    def test_new_windows_are_canonical_and_old_windows_are_preserved(self):
        self.assertEqual(contract.canonical_window(21, "gemma3_local512", True, True),
                         (511, 0))
        self.assertEqual(contract.canonical_window(23, "embedding_gemma2_local256",
                                                   False, True), (512, 512))
        self.assertEqual(contract.canonical_window(30, "qwen35_moe_gqa8_noncausal",
                                                   False, False), (None, None))
        for mode in (35, 36, 37):
            name = contract.CANONICAL38_APPENDED[mode][0]
            self.assertEqual(contract.canonical_window(mode, name, True, True),
                             (1023, 0))
        for mode in (31, 32, 33, 34):
            name = contract.CANONICAL38_APPENDED[mode][0]
            self.assertEqual(contract.canonical_window(mode, name,
                                                       mode not in (33,), False),
                             (None, None))

    def test_registry_tampering_is_rejected_for_active_and_legacy(self):
        for count in (19, 20, 25, 26, 31, 38):
            with self.subTest(count=count, field="mode_mask"):
                manifest = manifest_for(count)
                manifest["registry"]["mode_mask"] = "0x1"
                with self.assertRaises(RuntimeError):
                    contract.validate_manifest_inventory(manifest)
            with self.subTest(count=count, field="namespace"):
                manifest = manifest_for(count)
                manifest["registry"]["namespace"] = "canonical29"
                with self.assertRaises(RuntimeError):
                    contract.validate_manifest_inventory(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
