#!/opt/torch-sdk/bin/python3.12
"""Root-GPU-container native Qwen3.5 MoE gold (v3).

One engine per model/pool; caller's seven batch fields are passed to the real
19-argument tei_forward. The actual source run_physical_calls return records are
kept verbatim. Light SDK/GOT proof must pass first and GOT is repeated in
process before tei_create. No whole-model CPU forward/repack. Source-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

import model_checkpoint as mc  # noqa: E402
import native_batch as nb  # noqa: E402
import native_light as nl  # noqa: E402

SCHEMA = "oxide-q35moe-native-gold/v1"
POOL_CODES = {"cls": 0, "mean": 1, "last-token": 2}
GRAPH_DEFAULTS = {"INFINITY_TORCH_CUDA_GRAPHS": "1",
                  "INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES": "8",
                  "INFINITY_TORCH_CUDA_GRAPH_ENTRIES": "4",
                  "INFINITY_TORCH_CUDA_GRAPH_REUSE": "3",
                  "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES": "512MiB",
                  "INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS": "4096"}
SOURCE_FILES = ("native_light.py", "native_gold.py", "native_base.py",
                "native_raw_probe.py", "native_batch.py", "model_checkpoint.py",
                "container_sdk_got.py", "bridge.h", "bridge.cpp", "qwen35_models.cpp")


class GoldError(ValueError):
    pass


def maybe_sha(path: Path):
    path = Path(path)
    try:
        return nl.sha256_file(path) if path.is_file() else None
    except OSError:
        return None


def source_hashes() -> dict:
    result = {}
    for name in SOURCE_FILES:
        digest = maybe_sha(ROOT / name) or maybe_sha(SOURCE_INPUT / name)
        if digest is None:
            raise GoldError(f"required source is missing: {name}")
        result[name] = digest
    return result


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise GoldError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def load_case_file(path: Path) -> list[dict]:
    data = Path(path).read_bytes()
    raw = json.loads(data, object_pairs_hook=unique_object)
    if isinstance(raw, dict) and "cases" in raw:
        cases = raw["cases"]
    elif isinstance(raw, dict) and "batch" in raw:
        cases = [raw]
    elif isinstance(raw, list):
        cases = raw
    else:
        raise GoldError("case JSON must be a case object, a list, or {cases:[...]}")
    if not isinstance(cases, list) or not cases:
        raise GoldError("at least one physical case is required")
    result, seen = [], set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise GoldError(f"case {index} must be an object")
        case_id = case.get("case_id", index)
        if isinstance(case_id, bool) or not isinstance(case_id, (int, str)):
            raise GoldError(f"case {index} case_id must be int or string")
        if case_id in seen:
            raise GoldError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        pool = case.get("pool")
        if pool not in POOL_CODES:
            raise GoldError(f"case {case_id!r} pool must be one of {sorted(POOL_CODES)}")
        batch = nb.validate_batch(case.get("batch"))
        result.append({"case_id": case_id, "pool": pool, "batch": batch,
                       "batch_sha256": hashlib.sha256(
                           json.dumps(batch, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest(),
                       "layout": nb.physical_layout(batch, nl.Q35["hidden_size"])})
    return result


def require_minimal_t7(cases: list[dict]) -> None:
    if len(cases) != 1 or cases[0]["case_id"] != "t7_a" or cases[0]["pool"] != "mean":
        raise GoldError("minimal starter requires exactly one t7_a mean case")


def check_light_proof(path: Path, model: Path, library: Path,
                      bridge_header: Path, snapshot: dict) -> dict:
    report = json.loads(Path(path).read_text())
    if report.get("schema") != nl.SCHEMA or not report.get("ok"):
        raise GoldError("light SDK/GOT proof did not pass")
    if report.get("engine_created") or report.get("engine_or_forward_called"):
        raise GoldError("light proof is not CPU-only")
    if report.get("identity_stable") is not True:
        raise GoldError("light proof identity was not stable")
    if report.get("config_identity") != snapshot["config_identity"]:
        raise GoldError("light proof uses a different config identity")
    header = report.get("header_identity") or {}
    if header.get("header_digest_sha256") != snapshot["header_identity"]["header_digest_sha256"]:
        raise GoldError("light proof uses different headers")
    checks = report.get("sdk_proof_checks") or {}
    required = ("sdk_sha256", "sdk_loaded", "tei_forward_symbol", "tei_forward_args",
                "got_targets", "got_vendor_130200", "cublaslt_version", "torch_prefix",
                "torch_git", "cuda_not_initialized", "cuda_visible_devices",
                "ld_bind_now", "no_engine")
    failed = [name for name in required if checks.get(name) is not True]
    if failed:
        raise GoldError(f"light proof checks failed: {failed}")
    if report.get("library") != str(Path(library).resolve()):
        raise GoldError("light proof library differs")
    if report.get("model") != str(Path(model).resolve()):
        raise GoldError("light proof model differs")
    if report.get("bridge", {}).get("sha256") != maybe_sha(bridge_header):
        raise GoldError("light proof bridge differs")
    if maybe_sha(library) != nl.EXPECTED_SDK_SHA256:
        raise GoldError("gold library is not the exact b66 SDK")
    return report


def in_process_got(library: Path, vendor: Path) -> dict:
    import container_sdk_got as csg

    targets = csg.resolved_relocations(Path(library).resolve(), "cublas")
    vendor_path = Path(vendor).resolve()
    for target in targets:
        resolved = target.get("resolved_realpath")
        target["vendor_130200"] = bool(resolved and Path(resolved).parent == vendor_path)
        if not target["pointer_nonzero"] or not target["vendor_130200"]:
            raise GoldError(f"cublas GOT target is not vendor 130200: {target}")
    return {"cublas_got_target_count": len(targets), "targets": targets,
            "all_vendor_130200": True}


def derived_capi_config(raw: dict) -> dict:
    text = nl.text_config(raw)
    return {"hidden_size": nl.Q35["hidden_size"],
            "num_attention_heads": nl.Q35["num_attention_heads"],
            "num_hidden_layers": nl.Q35["num_hidden_layers"],
            "intermediate_size": int(text.get("intermediate_size", 0) or 0),
            "vocab_size": int(text["vocab_size"]),
            "max_position_embeddings": int(text["max_position_embeddings"]),
            "type_vocab_size": int(text.get("type_vocab_size", 1)),
            "layer_norm_eps": float(text.get("layer_norm_eps", text.get("rms_norm_eps", 1e-6))),
            "hidden_act": text.get("hidden_act", text.get("hidden_activation", "silu"))}


def build_pool_options(raw: dict, pool: str) -> dict:
    import native_base

    for name, value in GRAPH_DEFAULTS.items():
        os.environ.setdefault(name, value)
    options = native_base.build_options(raw, pool)
    options["_fp8_dynamic"] = False
    for key, value in raw.items():
        if options.get(key) != value:
            raise GoldError(f"raw option {key} was flattened")
    return options


def canonical_entries(layout: dict, flat: bytes) -> list[dict]:
    last = {segment["index"]: segment for segment in layout["segments"]}
    entries = []
    for index in sorted(last):
        segment = last[index]
        count = segment["length_floats"]
        values = struct.unpack_from("<%df" % count, flat, segment["offset_floats"] * 4)
        bits = struct.unpack_from("<%dI" % count, flat, segment["offset_floats"] * 4)
        entries.append({"index": index, "kind": segment["kind"],
                        "shape": list(segment["shape"]), "values": list(values),
                        "f32_bits": list(bits)})
    return entries


def run_case(native_raw_probe, library, handle, case: dict, repeat: int) -> dict:
    layout = case["layout"]
    empty = layout["physical_float_count"] == 0
    if empty:
        records, outputs, facts, flat, result_call = [], [], {
            "entry_count": 0, "float_count": 0, "raw_overrides_pooled": [],
            "duplicate_selection_indices": []}, b"", None
    else:
        records, outputs, facts, flat, result_call = native_raw_probe.run_physical_calls(
            library, handle, case["batch"], layout, POOL_CODES[case["pool"]], repeat)
    entries = canonical_entries(layout, flat) if flat else []
    if any(not math.isfinite(value) for entry in entries for value in entry["values"]):
        raise GoldError(f"non-finite Native output in {case['case_id']}")
    if [entry["values"] for entry in entries] != [entry["values"] for entry in outputs]:
        raise GoldError(f"canonical reconstruction disagrees for {case['case_id']}")
    return {"case_id": case["case_id"], "pool": case["pool"],
            "pool_code": POOL_CODES[case["pool"]], "batch": case["batch"],
            "batch_sha256": case["batch_sha256"], "batch_facts": nb.batch_facts(case["batch"]),
            "width": nl.Q35["hidden_size"], "layout": layout, "empty": empty,
            "empty_semantic_only": empty, "outputs": entries, "facts": facts,
            "finite": True, "guards_ok": True, "repeat": repeat,
            "result_call_index": result_call,
            "first_flat_sha256": records[0]["physical_sha256"] if records else None,
            "physical_sha256": hashlib.sha256(flat).hexdigest() if flat else None,
            "repeat_bit_identity": all(item["bit_identical_to_first"] for item in records),
            "source_call_keys": sorted(records[0]) if records else [],
            "calls": records,
            "graphs": {"requested": bool(case.get("graphs_options", {}).get("_cuda_graphs")),
                       "observed_replay": any(item["observed_replay"] for item in records),
                       "observed_capture": any(item["observed_capture"] for item in records),
                       "runtime_graph_proof": False,
                       "note": "actual tei_create handle counters; never forced"}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", action="store_true")
    parser.add_argument("--model", type=Path, default=nl.DEFAULT_MODEL)
    parser.add_argument("--library", type=Path, default=nl.DEFAULT_LIBRARY)
    parser.add_argument("--vendor", type=Path, default=nl.DEFAULT_VENDOR)
    parser.add_argument("--bridge-header", type=Path, default=SOURCE_INPUT / "bridge.h")
    parser.add_argument("--sdk-proof", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--minimal-t7", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.container:
        raise SystemExit("native_gold.py is Root-GPU-container-only; pass --container")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    cases = load_case_file(args.cases)
    if args.minimal_t7:
        require_minimal_t7(cases)
    before = nl.identity_snapshot(args.model)
    source_before = source_hashes()
    bridge = nl.check_bridge_header(args.bridge_header)
    light = check_light_proof(args.sdk_proof, args.model, args.library,
                              args.bridge_header, before)
    raw, derived = before["raw_config"], derived_capi_config(before["raw_config"])
    checkpoint = before["checkpoint"]

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import native_base
    import native_raw_probe

    if torch.cuda.is_initialized():
        raise GoldError("CUDA initialized before SDK/GOT proof")
    library = native_base.bind_library(Path(args.library))
    if len(library.tei_forward.argtypes) != nl.EXPECTED_TEI_FORWARD_ARGS:
        raise GoldError("tei_forward is not 19 arguments")
    got = in_process_got(Path(args.library), Path(args.vendor))
    if torch.cuda.is_initialized():
        raise GoldError("CUDA initialized during SDK/GOT proof")

    pool_order = []
    for case in cases:
        if case["pool"] not in pool_order:
            pool_order.append(case["pool"])
    reports = {case["case_id"]: None for case in cases}
    engines = []
    for pool in pool_order:
        options = build_pool_options(raw, pool)
        for case in cases:
            if case["pool"] == pool:
                case["graphs_options"] = options
        handle = None
        try:
            handle = native_base.create_engine(library, derived, options, args.device, 2, True)
            native_base.upload_weights(library, handle, checkpoint)
            native_base.check_status(library, library.tei_ready(handle), "tei_ready")
            if (int(library.tei_output_width(handle)) != nl.Q35["hidden_size"]
                    or int(library.tei_pooled_width(handle)) != nl.Q35["hidden_size"]):
                raise GoldError("native widths are not 2048")
            for case in cases:
                if case["pool"] == pool:
                    reports[case["case_id"]] = run_case(
                        native_raw_probe, library, handle, case, args.repeat)
        finally:
            if handle is not None:
                library.tei_destroy(handle)
        engines.append({"pool": pool, "device": args.device, "dtype_code": 2,
                        "dtype": "bfloat16", "graphs_options": options,
                        "raw_options_sha256": hashlib.sha256(
                            json.dumps(options, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                        "raw_options": options, "dynamic_fp8": False,
                        "weights_admitted": True})
    after = nl.identity_snapshot(args.model)
    source_after = source_hashes()
    identity_stable = (before["identity_sha256"] == after["identity_sha256"]
                       and before["config_identity"] == after["config_identity"])
    sources_stable = source_before == source_after
    if not identity_stable or not sources_stable:
        raise GoldError("model/source/header identity changed during gold")
    ordered = [reports[case["case_id"]] for case in cases]
    status = "ok" if all(case["repeat_bit_identity"] for case in ordered) else "repeat-bit-difference"
    report = {"schema": SCHEMA, "stage": "native-gold", "status": status,
              "backend": "native-capi-direct-tei_forward", "container_only": True,
              "device": args.device, "dtype": "bfloat16", "dtype_code": 2,
              "dynamic_fp8": False, "model": str(Path(args.model).resolve()),
              "library": str(Path(args.library).resolve()),
              "library_sha256": nl.sha256_file(args.library), "bridge": bridge,
              "sdk_proof_path": str(Path(args.sdk_proof).resolve()),
              "sdk_proof_identity_sha256": light.get("identity_sha256_before"),
              "sdk_got_in_process": got, "config_identity": before["config_identity"],
              "header_identity": before["header_identity"],
              "identity_sha256_before": before["identity_sha256"],
              "identity_sha256_after": after["identity_sha256"],
              "identity_stable": identity_stable,
              "source_sha256_before": source_before,
              "source_sha256_after": source_after, "sources_stable": sources_stable,
              "derived_capi_config": derived, "engines": engines,
              "engine_per_pool": True, "cases": ordered,
              "no_retokenize_rebatch_padding": True,
              "all_seven_batch_fields_preserved": True,
              "minimal_t7": args.minimal_t7, "gpu_fit_proven": False,
              "fit_note": "72 GiB admission unproven by this source-only delivery",
              "timing_claims": False, "errors": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": status, "case_count": len(ordered), "pools": pool_order,
                      "entry_count": sum(len(case["outputs"]) for case in ordered),
                      "minimal_t7": args.minimal_t7, "output": str(args.output)}))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
