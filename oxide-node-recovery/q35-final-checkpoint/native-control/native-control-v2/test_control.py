#!/usr/bin/env python3
"""Source-only tests for the actual Rust v2 quality report shape (v4).

Constructs metadata.before/after, quality.cases, flat requested, valid/invalid
cases, top-level pool, case.shape, file batch_sha256, and bits_u32 exactly as
the copied Rust source does. No execution.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "input"))

import native_light as nl
import root_control as rc
import strict_compare as sc


def bits_of(values):
    return [struct.unpack("<I", struct.pack("<f", float(value)))[0] for value in values]


def layer_types():
    return ["full_attention"] * 40


def config_facts():
    return {
        "model_type": "qwen3_5_moe",
        "text_model_type": "qwen3_5_moe_text",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "class": ["Qwen3_5MoeForConditionalGeneration"],
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 256,
        "layer_types": layer_types(),
        "layer_order_sha256": sc.compact_sha(layer_types()),
        "dtype": "bfloat16",
    }


def native_config(sha="a" * 64):
    return {
        "architecture_class": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "layer_count": 40,
        "layer_types": layer_types(),
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "source_dtype": "bfloat16",
        "config_sha256": sha,
        "layer_order_sha256": sc.compact_sha(layer_types()),
        "dense_intermediate_size": 0,
    }


def batch():
    return {"input_ids": [1, 2, 3], "position_ids": [0, 1, 2],
            "token_type_ids": [0, 0, 0], "cumulative_seq_lengths": [0, 3],
            "pooled_indices": [0], "raw_indices": [], "max_length": 3}


def native_case(name, outputs, batch_value=None, pool="mean", width=2048):
    batch_value = batch_value or batch()
    return {"case_id": name, "pool": pool, "batch": batch_value,
            "batch_sha256": sc.canonical_batch_sha(batch_value), "width": width,
            "outputs": outputs}


def rust_valid(name, outputs, batch_value=None, file_sha="d" * 64):
    return {
        "name": name, "valid": True, "status": "ok",
        "batch_file": f"fixtures/{name}.json", "batch_sha256": file_sha,
        "batch_fingerprint_sha256": "e" * 64, "submission_type": "Ready",
        "backend_duration_ns": 1, "wall_duration_ns": 1,
        "shape": {"D": 2048, "H": 2048}, "requested": batch_value or batch(),
        "output": {"outputs": outputs, "facts": {"entry_count": len(outputs)}},
    }


def rust_invalid(name, expected="Inference", actual=None, status="ok",
                 file_sha="f" * 64):
    requested = {key: value for key, value in batch().items() if key != "token_type_ids"}
    requested["special"] = None
    return {
        "name": name, "valid": False, "status": status,
        "batch_file": f"fixtures/{name}.json", "batch_sha256": file_sha,
        "batch_fingerprint_sha256": "e" * 64,
        "expected_error_variant": expected,
        "actual_error_variant": expected if actual is None else actual,
        "error": "typed rejection", "requested": requested,
    }


def native_pooled(values=None):
    values = values if values is not None else [0.5] * 2048
    return {"index": 0, "kind": "pooled", "shape": [2048],
            "values": list(values), "f32_bits": bits_of(values)}


def rust_pooled(values=None, bits=None):
    values = values if values is not None else [0.5] * 2048
    return {"index": 0, "kind": "pooled", "shape": [2048],
            "values": list(values), "bits_u32": list(bits or bits_of(values))}


def native_report(cases, sha="a" * 64):
    return {"schema": "oxide-q35moe-native-gold/v1", "status": "ok", "identity_stable": True, "sources_stable": True, "errors": [], "config_identity": native_config(sha), "cases": cases}


def metadata(sha="a" * 64, config=None):
    before = {
        "model": "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b",
        "config_sha256": sha, "config": config or config_facts(),
        "safetensors_headers": [], "indices": [], "module_identity_sha256": {},
        "provider": {"source": "x"}, "collector_created_engine": False,
        "collector_created_context": False, "tensor_payload_read_bytes": 0,
    }
    after = json.loads(json.dumps(before))
    return {"before": before, "after": after,
            "before_after_equal": True, "payload_hashed": False}


def rust_report(cases, config=None, sha="a" * 64, pool="mean", meta=None):
    return {
        "schema": sc.RUST_REPORT_SCHEMA, "mode": "quality", "status": "ok",
        "started_unix_seconds": 1,
        "model": "/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b",
        "pool": pool, "product_source": "x", "product_freeze_sha256": "y",
        "normal_factory_source": "z", "canonical31_provider": "p",
        "root_checkpoint": "r", "engine_count": 1, "normal_factory_route": True,
        "submission_type": "Ready", "phases": [],
        "quality": {
            "route": "normal-service-factory",
            "normal_factory": "infinity_v3_backend::Backend::new_selected",
            "pool": pool, "shape": {"D": 2048, "H": 2048}, "cases": cases,
            "failure_count": 0, "numeric_empty_outputs_are_semantic_only": True,
        },
        "life": None, "metadata": meta or metadata(sha, config),
        "errors": [], "typed_errors": [],
    }


class ActualSchemaTests(unittest.TestCase):
    def assert_changed(self, result, needle=None):
        self.assertEqual(result["status"], "CHANGED")
        self.assertFalse(result["unchanged"])
        if needle:
            self.assertTrue(any(needle in reason for reason in result["reasons"]),
                            result["reasons"])

    def test_actual_shape_unchanged(self):
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])]))
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["diff_count"], 0)
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["invalid_case_count"], 0)

    def test_native_failed_status_rejected(self):
        native = native_report([native_case("t7_a", [native_pooled()])])
        native["status"] = "failed"
        self.assert_changed(sc.compare(native, rust_report([rust_valid("t7_a", [rust_pooled()])])), "schema/status")

    def test_native_changed_identity_rejected(self):
        native = native_report([native_case("t7_a", [native_pooled()])])
        native["identity_stable"] = False
        self.assert_changed(sc.compare(native, rust_report([rust_valid("t7_a", [rust_pooled()])])), "identity/source")

    def test_native_layer_digest_rejected(self):
        native = native_report([native_case("t7_a", [native_pooled()])])
        native["config_identity"]["layer_order_sha256"] = "0" * 64
        self.assert_changed(sc.compare(native, rust_report([rust_valid("t7_a", [rust_pooled()])])), "layer_order_sha256")

    def test_metadata_flat_config_rejected(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["metadata"] = {"config": config_facts(), "config_sha256": "a" * 64}
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "metadata.config")

    def test_metadata_before_after_required_equal(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["metadata"]["before_after_equal"] = False
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "before_after_equal")

    def test_metadata_after_snapshot_mismatch(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["metadata"]["after"]["config"]["head_dim"] = 128
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "snapshots differ")

    def test_metadata_payload_hashed_rejected(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["metadata"]["payload_hashed"] = True
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "payload_hashed")

    def test_report_status_required(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["status"] = "running"
        self.assert_changed(sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                                       rust), "status must be ok")

    def test_normal_factory_route_required(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["normal_factory_route"] = False
        self.assert_changed(sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                                       rust), "normal_factory_route")

    def test_engine_count_required(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["engine_count"] = 2
        self.assert_changed(sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                                       rust), "engine_count")

    def test_errors_and_typed_errors_required_empty(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()])])
        rust["typed_errors"] = [{"stage": "x"}]
        self.assert_changed(sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                                       rust), "typed_errors")

    def test_invalid_cases_excluded_after_validation(self):
        cases = [rust_valid("t7_a", [rust_pooled()]),
                 rust_invalid("empty", expected="Inference", actual="Inference"),
                 rust_valid("t1", [rust_pooled()])]
        native = native_report([native_case("t7_a", [native_pooled()]),
                                native_case("t1", [native_pooled()])])
        result = sc.compare(native, rust_report(cases),
                            manifest={"t7_a": "d" * 64, "t1": "d" * 64, "empty": "f" * 64})
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["invalid_case_count"], 1)

    def test_invalid_variant_mismatch(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()]),
                            rust_invalid("empty", expected="Inference", actual="Start")])
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "expected/actual variant")

    def test_invalid_status_mismatch(self):
        rust = rust_report([rust_valid("t7_a", [rust_pooled()]),
                            rust_invalid("empty", status="variant-mismatch")])
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]), rust)
        self.assert_changed(result, "status must be ok")

    def test_valid_case_status_required(self):
        case = rust_valid("t7_a", [rust_pooled()])
        case["status"] = "deferred"
        self.assert_changed(sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                                       rust_report([case])), "valid Rust case")

    def test_flat_requested_required(self):
        case = rust_valid("t7_a", [rust_pooled()])
        case["requested"] = {"pool": "mean", "batch": batch()}
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([case]))
        self.assert_changed(result, "flat seven fields")

    def test_top_level_pool_drives_comparison(self):
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()], pool="mean")]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])], pool="cls"))
        self.assert_changed(result, "pool")

    def test_case_shape_must_be_2048(self):
        case = rust_valid("t7_a", [rust_pooled()])
        case["shape"] = {"D": 1024, "H": 1024}
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([case]))
        self.assert_changed(result, "shape H/D")

    def test_file_sha_is_retained_not_canonical(self):
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()], file_sha="9" * 64)]))
        self.assertEqual(result["status"], "UNCHANGED")
        self.assertEqual(result["retained_file_sha256"], ["9" * 64])
        self.assertNotEqual(result["retained_file_sha256"][0],
                            result["canonical_batch_sha256"][0])

    def test_manifest_file_sha_validated(self):
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()], file_sha="9" * 64)]),
                            manifest={"t7_a": "8" * 64})
        self.assert_changed(result, "manifest")

    def test_canonical_batch_derived_from_requested(self):
        native = native_report([native_case("t7_a", [native_pooled()])])
        requested = batch()
        requested["input_ids"] = [9, 9, 9]
        rust = rust_report([rust_valid("t7_a", [rust_pooled()], batch_value=requested)])
        self.assert_changed(sc.compare(native, rust), "batch.input_ids")

    def test_no_torch_dtype_guess(self):
        config = config_facts()
        config["torch_dtype"] = config.pop("dtype")
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])], config))
        self.assert_changed(result, "dtype")

    def test_text_model_type_mismatch(self):
        config = config_facts()
        config["text_model_type"] = "other"
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])], config))
        self.assert_changed(result, "text_model_type")

    def test_architecture_class_mismatch(self):
        config = config_facts()
        config["architectures"] = ["Other"]
        config["class"] = ["Other"]
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])], config))
        self.assert_changed(result, "architecture_class")

    def test_layer_order_sha_mismatch(self):
        config = config_facts()
        config["layer_order_sha256"] = "0" * 64
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [rust_pooled()])], config))
        self.assert_changed(result, "layer_order_sha256")

    def test_f32_bits_alias_rejected(self):
        entry = rust_pooled()
        entry["f32_bits"] = entry.pop("bits_u32")
        result = sc.compare(native_report([native_case("t7_a", [native_pooled()])]),
                            rust_report([rust_valid("t7_a", [entry])]))
        self.assert_changed(result, "bits_u32")

    def test_signed_zero(self):
        native_values = [0.0, 1.0] + [0.5] * 2046
        rust_values = list(native_values)
        rust_values[0] = -0.0
        result = sc.compare(
            native_report([native_case("t7_a",
                                       [{"index": 0, "kind": "pooled", "shape": [2048],
                                         "values": native_values,
                                         "f32_bits": bits_of(native_values)}])]),
            rust_report([rust_valid("t7_a", [rust_pooled(rust_values)])]))
        self.assert_changed(result, "strict f32 word equality")
        self.assertEqual(result["signed_zero_diffs"], 1)

    def test_nan_and_missing_vectors(self):
        values = [0.5] * 2048
        values[3] = float("nan")
        self.assert_changed(sc.compare(
            native_report([native_case("t7_a", [native_pooled()])]),
            rust_report([rust_valid("t7_a", [rust_pooled(values)])])), "not finite")
        self.assert_changed(sc.compare(
            native_report([native_case("t7_a", [native_pooled()])]),
            rust_report([rust_valid("t7_a", [])])), "entry count")

    def test_reordered_valid_names(self):
        native = native_report([native_case("t7_a", [native_pooled()]),
                                native_case("t1", [native_pooled()])])
        rust = rust_report([rust_valid("t1", [rust_pooled()]),
                            rust_valid("t7_a", [rust_pooled()])])
        self.assert_changed(sc.compare(native, rust), "name order")

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema": "x", "schema": "y"}')
            with self.assertRaises(sc.CompareError):
                sc.load_json_strict(path)


class LightFactsTests(unittest.TestCase):
    def test_actual_metadata_constants(self):
        self.assertEqual(nl.EXPECTED_TENSORS, 1811)
        self.assertEqual(nl.EXPECTED_DTYPE_COUNTS, {"BF16": 1751, "F32": 60})
        self.assertEqual(nl.EXPECTED_SHARDS, 14)
        self.assertEqual(nl.EXPECTED_MODEL_BYTES, 71903878016)

    def test_actual_config_facts(self):
        raw = {"model_type": "qwen3_5_moe",
               "architectures": ["Qwen3_5MoeForConditionalGeneration"],
               "text_config": {"model_type": "qwen3_5_moe_text",
                               "hidden_size": 2048, "num_hidden_layers": 40,
                               "num_attention_heads": 16, "num_key_value_heads": 2,
                               "head_dim": 256, "num_experts": 256,
                               "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                               "vocab_size": 151936, "max_position_embeddings": 32768,
                               "layer_types": layer_types(), "dtype": "bfloat16"}}
        config = nl.validate_q35_config(raw)
        self.assertEqual(config["text_model_type"], "qwen3_5_moe_text")
        self.assertEqual(config["source_dtype"], "bfloat16")


class ControllerV4Tests(unittest.TestCase):
    def test_native_only_phases(self):
        self.assertEqual(rc.PHASES, ("light", "starter-native", "quality"))
        self.assertNotIn("starter-rust", rc.PHASES)
        self.assertNotIn("life", rc.PHASES)

    def test_t7_a_selection(self):
        fixture = {"cases": [
            {"case_id": "t1", "pool": "mean", "batch": batch()},
            {"case_id": "t7_a", "pool": "mean", "batch": batch()},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.json"
            path.write_text(json.dumps(fixture))
            selected = rc.select_t7_a(path)
        self.assertEqual(selected, {"cases": [fixture["cases"][1]]})


if __name__ == "__main__":
    unittest.main()
