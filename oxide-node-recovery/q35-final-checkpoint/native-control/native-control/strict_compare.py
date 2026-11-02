#!/usr/bin/env python3
"""Strict CPU comparator for Native gold vs the actual Rust v2 quality report.

Explicit actual schema (copied source):
  top: schema/mode/status/pool/engine_count/normal_factory_route/errors/typed_errors
  metadata: {before:{config,config_sha256,...}, after:{...}, before_after_equal, payload_hashed}
  quality: {route, normal_factory, pool, shape:{D,H}, cases, failure_count,
            numeric_empty_outputs_are_semantic_only}
  valid case: name, valid=true, status=ok, batch_sha256 (file bytes digest),
              shape:{D,H}, requested (FLAT seven fields), output.outputs[]
  invalid case: name, valid=false, status=ok, expected_error_variant,
                actual_error_variant (must match expected)
  entry: index, kind, shape, values, bits_u32

No metadata.config in quality. No requested.pool/requested.batch. No aliases.
Canonical batch digest is derived from requested; file SHA is retained and
checked only against an explicit manifest mapping, never against canonical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_INPUT = ROOT / "input"
SOURCE_INPUT = LOCAL_INPUT if LOCAL_INPUT.is_dir() else (
    ROOT.parent / "input" if (ROOT.parent / "input").is_dir()
    else Path("/tmp/oxide-q35moe-native-runtime-control-v1/input"))
if SOURCE_INPUT.is_dir():
    sys.path.insert(0, str(SOURCE_INPUT))

import native_batch as nb  # noqa: E402

SCHEMA = "oxide-q35moe-strict-compare/v4"
RUST_REPORT_SCHEMA = "oxide-q35moe-unified-runtime-report/v1"
BATCH_FIELDS = list(nb.BATCH_FIELDS)
ENTRY_KEYS = {"index", "kind", "shape", "values", "f32_bits"}
CASE_KEYS = {"case_id", "pool", "batch", "batch_sha256", "width", "outputs"}
CONFIG_KEYS = {
    "architecture_class", "model_type", "text_model_type", "hidden_size",
    "layer_count", "layer_types", "num_attention_heads", "num_key_value_heads",
    "head_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "dense_intermediate_size", "vocab_size", "max_position_embeddings",
    "source_dtype", "config_sha256", "layer_order_sha256",
}
POOLS = {"cls", "mean", "last-token"}
DEFAULT_MAX_ABS = 0.25
DEFAULT_RELATIVE_L2 = 0.02
DEFAULT_MIN_COS = 0.999


class CompareError(ValueError):
    pass


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise CompareError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def load_json_strict(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompareError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise CompareError(f"{path} must contain a JSON object")
    return value


def canonical_batch_sha(batch: dict) -> str:
    return hashlib.sha256(
        json.dumps(batch, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compact_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _src(obj: dict, key: str, where: str):
    if not isinstance(obj, dict) or key not in obj:
        raise CompareError(f"{where} is missing required field {key!r}")
    return obj[key]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_dtype(value, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompareError(f"{where}.dtype must be a nonempty string")
    value = value.lower()
    return "bfloat16" if value == "bf16" else value


def _config_from_rust(metadata: dict) -> dict:
    if "config" in metadata:
        raise CompareError("quality metadata must not use metadata.config; use metadata.before")
    before = _src(metadata, "before", "rust.metadata")
    after = _src(metadata, "after", "rust.metadata")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise CompareError("rust.metadata.before/after must be objects")
    if metadata.get("before_after_equal") is not True:
        raise CompareError("rust.metadata.before_after_equal must be true")
    if before != after:
        raise CompareError("rust.metadata.before and after snapshots differ")
    if metadata.get("payload_hashed") is not False:
        raise CompareError("rust.metadata.payload_hashed must be false")
    config = _src(before, "config", "rust.metadata.before")
    config_sha = _src(before, "config_sha256", "rust.metadata.before")
    if not isinstance(config, dict):
        raise CompareError("rust.metadata.before.config must be an object")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise CompareError("rust.metadata.before.config_sha256 must be a sha256")
    architectures = _src(config, "architectures", "rust.metadata.before.config")
    class_value = config.get("class")
    if class_value is not None and class_value != architectures:
        raise CompareError("rust config architectures and class differ")
    layer_types = _src(config, "layer_types", "rust.metadata.before.config")
    if not isinstance(layer_types, list):
        raise CompareError("rust config layer_types must be a list")
    layer_order_sha = config.get("layer_order_sha256")
    if layer_order_sha is not None and layer_order_sha != compact_sha(layer_types):
        raise CompareError("rust config layer_order_sha256 does not match layer_types")
    result = {
        "architecture_class": architectures,
        "model_type": _src(config, "model_type", "rust.metadata.before.config"),
        "text_model_type": config.get("text_model_type"),
        "hidden_size": _src(config, "hidden_size", "rust.metadata.before.config"),
        "layer_count": _src(config, "num_hidden_layers", "rust.metadata.before.config"),
        "layer_types": layer_types,
        "num_attention_heads": _src(config, "q_heads", "rust.metadata.before.config"),
        "num_key_value_heads": _src(config, "kv_heads", "rust.metadata.before.config"),
        "head_dim": _src(config, "head_dim", "rust.metadata.before.config"),
        "num_experts": _src(config, "num_experts", "rust.metadata.before.config"),
        "num_experts_per_tok": _src(config, "num_experts_per_tok", "rust.metadata.before.config"),
        "moe_intermediate_size": _src(config, "moe_intermediate_size",
                                      "rust.metadata.before.config"),
        "dense_intermediate_size": 0,
        "vocab_size": _src(config, "vocab_size", "rust.metadata.before.config"),
        "max_position_embeddings": _src(config, "max_position_embeddings",
                                        "rust.metadata.before.config"),
        "source_dtype": _normalize_dtype(
            _src(config, "dtype", "rust.metadata.before.config"),
            "rust.metadata.before.config"),
        "config_sha256": config_sha,
        "layer_order_sha256": compact_sha(layer_types),
    }
    return result


def _config_from_native(identity: dict) -> dict:
    if not isinstance(identity, dict):
        raise CompareError("native.config_identity must be an object")
    result = {key: _src(identity, key, "native.config_identity") for key in CONFIG_KEYS}
    if not isinstance(result["layer_types"], list):
        raise CompareError("native config layer_types must be a list")
    if not isinstance(result["config_sha256"], str) or len(result["config_sha256"]) != 64:
        raise CompareError("native config_sha256 is not a sha256")
    result["source_dtype"] = _normalize_dtype(result["source_dtype"], "native.config_identity")
    if result["layer_order_sha256"] != compact_sha(result["layer_types"]):
        raise CompareError("native layer_order_sha256 does not match layer_types")
    return result


def _file_sha(value, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CompareError(f"{where} must be a sha256 file digest")
    if any(c not in "0123456789abcdefABCDEF" for c in value):
        raise CompareError(f"{where} must be hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise CompareError(f"{where} must be hexadecimal") from error
    return value


def _entry(value: dict, where: str) -> dict:
    if not isinstance(value, dict):
        raise CompareError(f"{where} must be an object")
    required = {"index", "kind", "shape", "values", "bits_u32"}
    if set(value) != required:
        missing = sorted(required - set(value))
        raise CompareError(f"{where} is missing {missing}")
    return {"index": value["index"], "kind": value["kind"], "shape": value["shape"],
            "values": value["values"], "f32_bits": value["bits_u32"]}


def _manifest_sha(manifest, name: str):
    if manifest is None:
        return None
    if not isinstance(manifest, dict) or name not in manifest:
        raise CompareError(f"manifest lacks fixture {name!r}")
    return _file_sha(manifest[name], f"manifest[{name!r}]")


def normalize_rust(report: dict, manifest=None) -> dict:
    if report.get("schema") != RUST_REPORT_SCHEMA:
        raise CompareError(f"Rust report schema must be {RUST_REPORT_SCHEMA}")
    if report.get("mode") != "quality":
        raise CompareError("Rust report mode must be quality")
    if report.get("status") != "ok":
        raise CompareError("Rust report status must be ok")
    if report.get("normal_factory_route") is not True:
        raise CompareError("Rust report normal_factory_route must be true")
    if report.get("engine_count") != 1:
        raise CompareError("Rust report engine_count must be 1")
    if report.get("errors") != [] or report.get("typed_errors") != []:
        raise CompareError("Rust report errors/typed_errors must be empty")
    pool = report.get("pool")
    if pool not in POOLS:
        raise CompareError("Rust report pool is missing/unsupported")
    metadata = _src(report, "metadata", "rust")
    config = _config_from_rust(metadata)
    quality = _src(report, "quality", "rust")
    if quality.get("route") != "normal-service-factory":
        raise CompareError("Rust quality.route must be normal-service-factory")
    if quality.get("pool") != pool:
        raise CompareError("Rust quality.pool differs from top-level report.pool")
    if quality.get("failure_count") != 0:
        raise CompareError("Rust quality.failure_count must be 0")
    if quality.get("numeric_empty_outputs_are_semantic_only") is not True:
        raise CompareError("Rust quality numeric-empty semantic flag must be true")
    cases = _src(quality, "cases", "rust.quality")
    if not isinstance(cases, list) or not cases:
        raise CompareError("Rust quality.cases must be nonempty")
    valid, invalid, names = [], [], set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise CompareError(f"rust.quality.cases[{index}] must be an object")
        name = _src(case, "name", f"rust.quality.cases[{index}]")
        if not isinstance(name, str) or not name:
            raise CompareError(f"rust.quality.cases[{index}].name must be a string")
        if name in names:
            raise CompareError(f"duplicate Rust case name {name!r}")
        names.add(name)
        valid_flag = _src(case, "valid", f"rust.quality.cases[{index}]")
        if not isinstance(valid_flag, bool):
            raise CompareError(f"rust.quality.cases[{index}].valid must be bool")
        file_sha = _file_sha(_src(case, "batch_sha256",
                                  f"rust.quality.cases[{index}]"),
                             f"rust.case[{name}].batch_sha256")
        expected_manifest = _manifest_sha(manifest, name)
        if expected_manifest is not None and file_sha != expected_manifest:
            raise CompareError(f"rust case {name!r} batch_sha256 differs from manifest")
        if not valid_flag:
            if case.get("status") != "ok":
                raise CompareError(f"invalid Rust case {name!r} status must be ok")
            expected = _src(case, "expected_error_variant", f"rust.case[{name}]")
            actual = _src(case, "actual_error_variant", f"rust.case[{name}]")
            if not isinstance(expected, str) or not isinstance(actual, str):
                raise CompareError(f"invalid Rust case {name!r} variants must be strings")
            if expected != actual:
                raise CompareError(f"invalid Rust case {name!r} expected/actual variant differs")
            invalid.append({"name": name, "expected_error_variant": expected,
                            "actual_error_variant": actual, "status": "ok",
                            "batch_file_sha256": file_sha})
            continue
        if case.get("status") != "ok":
            raise CompareError(f"valid Rust case {name!r} status must be ok")
        requested = _src(case, "requested", f"rust.case[{name}]")
        if not isinstance(requested, dict) or set(requested) != set(BATCH_FIELDS):
            raise CompareError(f"valid Rust case {name!r} requested must be the flat seven fields")
        batch = nb.validate_batch(requested)
        shape = _src(case, "shape", f"rust.case[{name}]")
        if not isinstance(shape, dict) or shape.get("H") != 2048 or shape.get("D") != 2048:
            raise CompareError(f"valid Rust case {name!r} shape H/D must both be 2048")
        output = _src(case, "output", f"rust.case[{name}]")
        outputs = _src(output, "outputs", f"rust.case[{name}].output")
        if not isinstance(outputs, list):
            raise CompareError(f"rust.case[{name}].output.outputs must be a list")
        entries = [_entry(entry, f"rust.case[{name}].output[{position}]")
                   for position, entry in enumerate(outputs)]
        valid.append({
            "case_id": name, "pool": pool, "batch": batch,
            "batch_sha256": canonical_batch_sha(batch),
            "width": shape["H"], "outputs": entries,
            "batch_file_sha256": file_sha,
            "batch_fingerprint_sha256": case.get("batch_fingerprint_sha256"),
            "batch_file": case.get("batch_file"),
        })
    if not valid:
        raise CompareError("Rust quality has no valid cases")
    return {"schema": report.get("schema"), "config_identity": config,
            "pool": pool, "cases": valid, "invalid_cases": invalid}


def normalize_native(report: dict) -> dict:
    if report.get("schema") != "oxide-q35moe-native-gold/v1" or report.get("status") != "ok":
        raise CompareError("Native schema/status must describe successful actual gold")
    if report.get("identity_stable") is not True or report.get("sources_stable") is not True or report.get("errors") != []:
        raise CompareError("Native identity/source/errors evidence failed")
    config = _config_from_native(_src(report, "config_identity", "native"))
    cases = _src(report, "cases", "native")
    if not isinstance(cases, list) or not cases:
        raise CompareError("native.cases must be a nonempty list")
    normalized = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise CompareError(f"native.cases[{index}] must be an object")
        item = {key: _src(case, key, f"native.cases[{index}]") for key in CASE_KEYS}
        if not isinstance(item["outputs"], list):
            raise CompareError(f"native.cases[{index}].outputs must be a list")
        for position, entry in enumerate(item["outputs"]):
            if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
                raise CompareError(
                    f"native.cases[{index}].outputs[{position}] keys must be "
                    f"{sorted(ENTRY_KEYS)}")
        normalized.append(item)
    return {"schema": report.get("schema"), "config_identity": config,
            "cases": normalized}


def _entries(case: dict, side: str, reasons: list, index: int):
    outputs = case.get("outputs")
    entries, previous, seen = [], None, set()
    for position, entry in enumerate(outputs):
        where = f"{side}: case {index} entry {position}"
        if set(entry) != ENTRY_KEYS:
            reasons.append(f"{where} keys are not exactly {sorted(ENTRY_KEYS)}")
            continue
        item = entry["index"]
        if not _is_int(item) or item < 0:
            reasons.append(f"{where} index is not a nonnegative integer")
            continue
        if item in seen:
            reasons.append(f"{where} duplicate entry index {item}")
            continue
        seen.add(item)
        if previous is not None and item <= previous:
            reasons.append(f"{where} indices are not strictly increasing")
        previous = item
        kind = entry["kind"]
        if kind not in ("pooled", "raw"):
            reasons.append(f"{where} kind {kind!r} is not pooled/raw")
            continue
        shape = entry["shape"]
        if (not isinstance(shape, list) or not shape
                or not all(_is_int(d) and d > 0 for d in shape)):
            reasons.append(f"{where} shape is invalid")
            continue
        values, bits = entry["values"], entry["f32_bits"]
        if not isinstance(values, list) or not isinstance(bits, list):
            reasons.append(f"{where} values/f32_bits must be lists")
            continue
        count = math.prod(shape)
        if len(values) != count or len(bits) != count:
            reasons.append(f"{where} shape product {count} != values {len(values)} "
                           f"/ f32_bits {len(bits)}")
            continue
        ok, squared = True, 0.0
        for offset, (value, word) in enumerate(zip(values, bits)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"{where}[{offset}] value is not a JSON number")
                ok = False
                break
            numeric = float(value)
            if not math.isfinite(numeric):
                reasons.append(f"{where}[{offset}] value is not finite")
                ok = False
                break
            if not _is_int(word) or word < 0 or word > (1 << 32) - 1:
                reasons.append(f"{where}[{offset}] f32_bits is not a u32 word")
                ok = False
                break
            try:
                packed = struct.unpack("<I", struct.pack("<f", numeric))[0]
            except (OverflowError, struct.error):
                reasons.append(f"{where}[{offset}] value is outside f32")
                ok = False
                break
            if packed != word:
                reasons.append(f"{where}[{offset}] value/f32_bits disagree")
                ok = False
                break
            squared += numeric * numeric
        if not ok:
            continue
        if squared == 0.0:
            reasons.append(f"{where} is a vacuous all-zero vector")
            continue
        entries.append({"index": item, "kind": kind, "shape": list(shape),
                        "values": [float(v) for v in values], "f32_bits": list(bits)})
    return entries


def model_preflight(native: dict, rust: dict, reasons: list) -> None:
    for side, report in (("native", native), ("rust", rust)):
        identity = report.get("config_identity")
        if not isinstance(identity, dict) or set(identity) != CONFIG_KEYS:
            reasons.append(f"{side}: config identity keys are incomplete")
            continue
        pins = {"hidden_size": 2048, "layer_count": 40, "num_attention_heads": 16,
                "num_key_value_heads": 2, "head_dim": 256, "num_experts": 256,
                "num_experts_per_tok": 8, "moe_intermediate_size": 512}
        for key, value in pins.items():
            if identity.get(key) != value:
                reasons.append(f"{side}: {key} is not {value}")
        layers = identity.get("layer_types")
        if (not isinstance(layers, list) or len(layers) != 40
                or not all(item in ("full_attention", "linear_attention") for item in layers)):
            reasons.append(f"{side}: layer order is not the Qwen3.5 40-layer order")
        if identity.get("layer_order_sha256") != compact_sha(layers):
            reasons.append(f"{side}: layer_order_sha256 is not the known layer order")
        if identity.get("source_dtype") != "bfloat16":
            reasons.append(f"{side}: source dtype is not bfloat16")
        if not identity.get("architecture_class"):
            reasons.append(f"{side}: architecture class is missing")
        if not identity.get("model_type"):
            reasons.append(f"{side}: model type is missing")
    n, r = native.get("config_identity"), rust.get("config_identity")
    if isinstance(n, dict) and isinstance(r, dict):
        for key in sorted(CONFIG_KEYS):
            if n.get(key) != r.get(key):
                reasons.append(f"config_identity.{key} differs: {n.get(key)!r} != {r.get(key)!r}")


def preflight(native: dict, rust: dict, reasons: list) -> list[dict]:
    model_preflight(native, rust, reasons)
    native_cases, rust_cases = native.get("cases"), rust.get("cases")
    if len(native_cases) != len(rust_cases):
        reasons.append(f"valid case count differs: {len(native_cases)} != {len(rust_cases)}")
        return []
    pairs = []
    rust_pool = rust.get("pool")
    for index, (nc, rc) in enumerate(zip(native_cases, rust_cases)):
        if nc.get("case_id") != rc.get("case_id"):
            reasons.append(f"case {index} name order differs: "
                           f"{nc.get('case_id')!r} != {rc.get('case_id')!r}")
        if nc.get("pool") != rust_pool:
            reasons.append(f"native case {index} pool {nc.get('pool')!r} != report pool {rust_pool!r}")
        if nc.get("width") != 2048 or rc.get("width") != 2048:
            reasons.append(f"case {index} width is not 2048")
        nbatch = nc.get("batch")
        rbatch = rc.get("batch")
        try:
            nbatch = nb.validate_batch(nbatch)
            rbatch = nb.validate_batch(rbatch)
        except Exception as error:
            reasons.append(f"case {index} invalid batch: {error}")
            continue
        for field in BATCH_FIELDS:
            if nbatch.get(field) != rbatch.get(field):
                reasons.append(f"case {index} batch.{field} differs")
        canonical = canonical_batch_sha(nbatch)
        if nc.get("batch_sha256") != canonical:
            reasons.append(f"native case {index} batch_sha256 is not canonical")
        if rc.get("batch_sha256") != canonical_batch_sha(rbatch):
            reasons.append(f"rust case {index} requested canonical digest differs")
        nentries = _entries(nc, "native", reasons, index)
        rentries = _entries(rc, "rust", reasons, index)
        for side, batch_value, entries in (("native", nbatch, nentries),
                                           ("rust", rbatch, rentries)):
            selections = bool(batch_value["pooled_indices"] or batch_value["raw_indices"])
            if selections and not entries and not any(
                    f"{side}: case {index}" in reason for reason in reasons):
                reasons.append(f"{side}: case {index} has selections but no vectors")
            if entries and not selections:
                reasons.append(f"{side}: case {index} has vectors but no selections")
        if len(nentries) != len(rentries):
            reasons.append(f"case {index} entry count differs: {len(nentries)} != {len(rentries)}")
            continue
        if [e["index"] for e in nentries] != [e["index"] for e in rentries]:
            reasons.append(f"case {index} canonical entry indices differ")
            continue
        for ne, re_ in zip(nentries, rentries):
            if ne["kind"] != re_["kind"]:
                reasons.append(f"case {index} entry {ne['index']} kind differs")
            if ne["shape"] != re_["shape"]:
                reasons.append(f"case {index} entry {ne['index']} shape differs")
        pairs.append({"case_id": nc.get("case_id"), "pool": rust_pool,
                      "empty": not nentries, "native": nentries, "rust": rentries})
    return pairs


def compare(native_report: dict, rust_report: dict, manifest=None,
            max_abs: float = DEFAULT_MAX_ABS, relative_l2: float = DEFAULT_RELATIVE_L2,
            min_cos: float = DEFAULT_MIN_COS, word_gate: str = "strict") -> dict:
    reasons: list[str] = []
    try:
        native = normalize_native(native_report)
        rust = normalize_rust(rust_report, manifest)
    except CompareError as error:
        return {"schema": SCHEMA, "status": "CHANGED", "unchanged": False,
                "reasons": [f"contract error: {error}"], "diff_count": None,
                "max_abs": None, "relative_l2": None, "min_cos": None}
    pairs = preflight(native, rust, reasons)
    vector_pairs = [pair for pair in pairs if not pair["empty"]]
    diff_count = signed_zero = float_count = 0
    max_abs_seen = diff_sq = native_sq = 0.0
    min_cos_seen = None
    for pair in vector_pairs:
        for ne, re_ in zip(pair["native"], pair["rust"]):
            a, b = ne["values"], re_["values"]
            abits, bbits = ne["f32_bits"], re_["f32_bits"]
            if len(a) != len(b) or len(abits) != len(bbits):
                reasons.append(f"case {pair['case_id']} entry {ne['index']} length differs")
                continue
            dot = na = nb_ = 0.0
            for x, y, wx, wy in zip(a, b, abits, bbits):
                delta = x - y
                max_abs_seen = max(max_abs_seen, abs(delta))
                diff_sq += delta * delta
                native_sq += x * x
                dot += x * y
                na += x * x
                nb_ += y * y
                float_count += 1
                if wx != wy:
                    diff_count += 1
                    if x == 0.0 and y == 0.0 and (wx >> 31) != (wy >> 31):
                        signed_zero += 1
            if na == 0.0 or nb_ == 0.0:
                reasons.append(f"case {pair['case_id']} entry {ne['index']} has a zero norm")
                continue
            cosine = dot / (math.sqrt(na) * math.sqrt(nb_))
            min_cos_seen = cosine if min_cos_seen is None else min(min_cos_seen, cosine)
    relative = math.sqrt(diff_sq) / math.sqrt(native_sq) if native_sq > 0.0 else None
    if vector_pairs and word_gate == "strict" and diff_count:
        reasons.append(f"strict f32 word equality failed for {diff_count} words")
    if vector_pairs:
        if not (math.isfinite(max_abs_seen) and max_abs_seen <= max_abs):
            reasons.append(f"max_abs {max_abs_seen!r} exceeds {max_abs}")
        if relative is None or not (math.isfinite(relative) and relative <= relative_l2):
            reasons.append(f"relative_l2 {relative!r} exceeds {relative_l2}")
        if min_cos_seen is None or not (math.isfinite(min_cos_seen) and min_cos_seen >= min_cos):
            reasons.append(f"min_cos {min_cos_seen!r} is below {min_cos}")
    unchanged = not reasons
    return {
        "schema": SCHEMA, "status": "UNCHANGED" if unchanged else "CHANGED",
        "unchanged": unchanged, "word_gate": word_gate,
        "rust_schema": rust.get("schema"), "native_schema": native.get("schema"),
        "invalid_case_count": len(rust.get("invalid_cases", [])),
        "invalid_cases": rust.get("invalid_cases", []),
        "retained_file_sha256": [case.get("batch_file_sha256")
                                 for case in rust.get("cases", [])],
        "canonical_batch_sha256": [case.get("batch_sha256")
                                   for case in rust.get("cases", [])],
        "guards": {"max_abs": max_abs, "relative_l2": relative_l2, "min_cos": min_cos},
        "numeric_qualified": unchanged and float_count > 0,
        "case_count": len(pairs), "vector_case_count": len(vector_pairs),
        "empty_case_count": len(pairs) - len(vector_pairs),
        "empty_semantic_only": bool(pairs) and not vector_pairs,
        "float_count": float_count, "diff_count": diff_count,
        "signed_zero_diffs": signed_zero,
        "max_abs": max_abs_seen if vector_pairs else None,
        "relative_l2": relative, "min_cos": min_cos_seen, "reasons": reasons,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-abs", type=float, default=DEFAULT_MAX_ABS)
    parser.add_argument("--relative-l2", type=float, default=DEFAULT_RELATIVE_L2)
    parser.add_argument("--min-cos", type=float, default=DEFAULT_MIN_COS)
    parser.add_argument("--word-gate", choices=("strict", "off"), default="strict")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        native = load_json_strict(args.native)
        rust = load_json_strict(args.rust)
        manifest = load_json_strict(args.manifest) if args.manifest else None
        result = compare(native, rust, manifest, args.max_abs, args.relative_l2,
                         args.min_cos, args.word_gate)
    except CompareError as error:
        result = {"schema": SCHEMA, "status": "CHANGED", "unchanged": False,
                  "reasons": [f"contract error: {error}"], "diff_count": None,
                  "max_abs": None, "relative_l2": None, "min_cos": None}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0 if result.get("unchanged") else 2


if __name__ == "__main__":
    sys.exit(main())
