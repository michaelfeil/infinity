#!/usr/bin/env python3
"""CPU-only Native b66 contract control for Nemotron-3.5-Rerank-8B (Mistral3).

This script never imports Torch, never loads a shared library, never creates or
queries a CUDA context, never uses the network, and never reads tensor payloads.
It reads only config.json, model.safetensors.index.json and each safetensors
header (the 8-byte little-endian length plus the JSON header bytes) using the
Python standard library.

Run from this directory with:

    CUDA_VISIBLE_DEVICES='' python3 native_b66_control.py

A future Root GPU/Docker run is a separate step. Nothing in this file claims
that a GPU run happened.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("INFINITY_MISTRAL3_CONTROL_OUT", str(ROOT)))
CHECKPOINT = Path(
    os.environ.get(
        "INFINITY_MISTRAL3_MODEL",
        "/workspace/model-performance/michaelfeil/benchmark-overview/models/"
        "nvidia--Nemotron-3.5-Rerank-8B-BF16",
    )
)
SDK = Path(
    os.environ.get(
        "INFINITY_MISTRAL3_SDK",
        "/tmp/infinity-fast-weight-reader/libs/infinity-v3/target/debug/build/"
        "infinity-backend-libtorch-0ee14ee2449a0322/out/build/libtei_torch.so",
    )
)
EXPECTED_SDK_PREFIX = "b66"
CUBLASLT_EXPECTED = 130200
SCHEMA = "oxide-native-b66-mistral3-cpu-control/v1"
INVENTORY_SCHEMA = "oxide-native-b66-mistral3-tensor-inventory/v1"

DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
}
DTYPE_C_API = {
    "F32": 0,
    "F16": 1,
    "BF16": 2,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open("rb") as handle:
        return json.loads(handle.read())


def read_safetensors_header(path: Path):
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"short safetensors length prefix: {path}")
        (header_bytes,) = struct.unpack("<Q", raw_size)
        if header_bytes == 0 or header_bytes > 100_000_000:
            raise ValueError(f"invalid safetensors header length {header_bytes}: {path}")
        raw_header = handle.read(header_bytes)
        if len(raw_header) != header_bytes:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(raw_header)
    metadata = header.get("__metadata__", {})
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = info["dtype"]
        shape = [int(dim) for dim in info["shape"]]
        offsets = [int(value) for value in info["data_offsets"]]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        if len(shape) != len(info["shape"]) or len(offsets) != 2:
            raise ValueError(f"malformed tensor header for {name}")
        expected = DTYPE_BYTES[dtype]
        for dim in shape:
            if dim < 0:
                raise ValueError(f"negative dimension for {name}")
            expected *= dim
        if offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] - offsets[0] != expected:
            raise ValueError(f"offset/length mismatch for {name}")
        tensors[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": offsets,
            "nbytes": expected,
            "c_api_dtype": DTYPE_C_API.get(dtype),
        }
    data_start = header_bytes + 8
    data_length = file_size - data_start
    if data_length < 0:
        raise ValueError(f"safetensors header extends past end of file: {path}")
    payload_end = max((item["data_offsets"][1] for item in tensors.values()), default=0)
    if payload_end != data_length:
        raise ValueError(
            f"safetensors payload length mismatch for {path}: "
            f"header end {payload_end} != file data length {data_length}"
        )
    return {
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "data_start": data_start,
        "data_length": data_length,
        "file_size": file_size,
        "metadata": metadata,
        "tensors": tensors,
    }


def inventory():
    index_path = CHECKPOINT / "model.safetensors.index.json"
    index = read_json(index_path)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError("model.safetensors.index.json lacks weight_map")
    weight_map = index["weight_map"]
    index_meta = index.get("metadata", {})
    index_total_size = index_meta.get("total_size")
    if not isinstance(index_total_size, int) or index_total_size <= 0:
        raise ValueError("model.safetensors.index.json lacks metadata.total_size")

    shard_names = sorted(set(weight_map.values()))
    all_tensors = {}
    shard_inventory = {}
    total_payload = 0
    for shard in shard_names:
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError(f"unsafe shard path in index: {shard}")
        header = read_safetensors_header(CHECKPOINT / shard)
        expected_names = {name for name, owner in weight_map.items() if owner == shard}
        actual_names = set(header["tensors"])
        if expected_names != actual_names:
            missing = sorted(expected_names - actual_names)[:8]
            extra = sorted(actual_names - expected_names)[:8]
            raise ValueError(
                f"index/header tensor set mismatch for {shard}: "
                f"missing={missing} extra={extra}"
            )
        shard_inventory[shard] = {
            "header_bytes": header["header_bytes"],
            "header_sha256": header["header_sha256"],
            "data_start": header["data_start"],
            "data_length": header["data_length"],
            "file_size": header["file_size"],
            "tensors": len(header["tensors"]),
            "metadata": header["metadata"],
            "index_and_header_tensor_sets_match": True,
        }
        total_payload += header["data_length"]
        for name, info in header["tensors"].items():
            if name in all_tensors:
                raise ValueError(f"duplicate tensor across shards: {name}")
            all_tensors[name] = {**info, "shard": shard}

    if set(all_tensors) != set(weight_map):
        raise ValueError("index and header tensor names disagree")
    if total_payload != index_total_size:
        raise ValueError(
            f"index metadata.total_size {index_total_size} != header payload {total_payload}"
        )
    return index, all_tensors, shard_inventory, total_payload


def categorize(names):
    vision = [name for name in names if name.startswith("model.vision_tower.")]
    projector = [name for name in names if name.startswith("model.multi_modal_projector.")]
    language = [name for name in names if name.startswith("model.language_model.")]
    score = [name for name in names if name == "language_model.score.weight"]
    classified = set(vision) | set(projector) | set(language) | set(score)
    unclassified = sorted(set(names) - classified)
    return vision, projector, language, score, unclassified


def shape_rules(tensors, config):
    vision = config["vision_config"]
    text = config["text_config"]
    v_hidden = int(vision["hidden_size"])
    v_layers = int(vision["num_hidden_layers"])
    v_intermediate = int(vision["intermediate_size"])
    patch = int(vision["patch_size"])
    merge = int(config["spatial_merge_size"])
    t_hidden = int(text["hidden_size"])
    t_layers = int(text["num_hidden_layers"])
    t_intermediate = int(text["intermediate_size"])
    t_heads = int(text["num_attention_heads"])
    t_kv = int(text["num_key_value_heads"])
    t_head_dim = int(text["head_dim"])
    vocab = int(text["vocab_size"])

    rules = []

    def add(rule, expected, names):
        actual = []
        failures = []
        for name in names:
            info = tensors.get(name)
            if info is None:
                failures.append({"tensor": name, "error": "missing"})
                continue
            actual_shape = info["shape"]
            ok = actual_shape == expected
            if not ok:
                failures.append(
                    {"tensor": name, "expected": expected, "actual": actual_shape}
                )
            actual.append({"tensor": name, "shape": actual_shape, "ok": ok})
        rules.append(
            {
                "rule": rule,
                "expected_shape": expected,
                "tensor_count": len(names),
                "passed": not failures,
                "failures": failures,
                "tensors": actual,
            }
        )

    add(
        "vision_tower_patch_conv",
        [v_hidden, 3, patch, patch],
        ["model.vision_tower.patch_conv.weight"],
    )
    add(
        "vision_tower_ln_pre",
        [v_hidden],
        ["model.vision_tower.ln_pre.weight"],
    )
    for index in range(v_layers):
        prefix = f"model.vision_tower.transformer.layers.{index}."
        add(
            f"vision_layer_{index}_attention_norm",
            [v_hidden],
            [prefix + "attention_norm.weight"],
        )
        add(
            f"vision_layer_{index}_ffn_norm",
            [v_hidden],
            [prefix + "ffn_norm.weight"],
        )
        add(
            f"vision_layer_{index}_qkv_o",
            [v_hidden, v_hidden],
            [prefix + "attention." + name + ".weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")],
        )
        add(
            f"vision_layer_{index}_gate_up",
            [v_intermediate, v_hidden],
            [prefix + "feed_forward." + name + ".weight" for name in ("gate_proj", "up_proj")],
        )
        add(
            f"vision_layer_{index}_down",
            [v_hidden, v_intermediate],
            [prefix + "feed_forward.down_proj.weight"],
        )
    add(
        "projector_norm",
        [v_hidden],
        ["model.multi_modal_projector.norm.weight"],
    )
    add(
        "projector_merging_layer",
        [v_hidden, v_hidden * merge * merge],
        ["model.multi_modal_projector.patch_merger.merging_layer.weight"],
    )
    add(
        "projector_linear_1",
        [t_hidden, v_hidden],
        ["model.multi_modal_projector.linear_1.weight"],
    )
    add(
        "projector_linear_2",
        [t_hidden, t_hidden],
        ["model.multi_modal_projector.linear_2.weight"],
    )
    add(
        "sequence_classification_score",
        [1, t_hidden],
        ["language_model.score.weight"],
    )
    add(
        "language_embed_tokens",
        [vocab, t_hidden],
        ["model.language_model.embed_tokens.weight"],
    )
    add(
        "language_final_norm",
        [t_hidden],
        ["model.language_model.norm.weight"],
    )
    for index in range(t_layers):
        prefix = f"model.language_model.layers.{index}."
        add(
            f"language_layer_{index}_norms",
            [t_hidden],
            [prefix + "input_layernorm.weight", prefix + "post_attention_layernorm.weight"],
        )
        add(
            f"language_layer_{index}_q",
            [t_heads * t_head_dim, t_hidden],
            [prefix + "self_attn.q_proj.weight"],
        )
        add(
            f"language_layer_{index}_kv",
            [t_kv * t_head_dim, t_hidden],
            [prefix + "self_attn." + name + ".weight" for name in ("k_proj", "v_proj")],
        )
        add(
            f"language_layer_{index}_o",
            [t_hidden, t_heads * t_head_dim],
            [prefix + "self_attn.o_proj.weight"],
        )
        add(
            f"language_layer_{index}_gate_up",
            [t_intermediate, t_hidden],
            [prefix + "mlp." + name + ".weight" for name in ("gate_proj", "up_proj")],
        )
        add(
            f"language_layer_{index}_down",
            [t_hidden, t_intermediate],
            [prefix + "mlp.down_proj.weight"],
        )

    bias_names = [
        name
        for name in tensors
        if name.startswith("model.multi_modal_projector.") and name.endswith(".bias")
    ]
    bias_rule = {
        "rule": "projector_bias_absent",
        "expected_shape": None,
        "tensor_count": 0,
        "passed": not bias_names,
        "failures": (
            [{"tensor": name, "error": "unexpected bias"} for name in bias_names]
            if bias_names
            else []
        ),
    }
    rules.append(bias_rule)

    failures = [failure for rule in rules for failure in rule["failures"]]
    return {
        "all_passed": not failures,
        "checks_total": sum(rule["tensor_count"] for rule in rules),
        "rules_total": len(rules),
        "rules_passed": sum(1 for rule in rules if rule["passed"]),
        "failures": failures,
        "rules": rules,
        "dimensions": {
            "vision_hidden": v_hidden,
            "vision_intermediate": v_intermediate,
            "vision_layers": v_layers,
            "vision_patch": patch,
            "text_hidden": t_hidden,
            "text_intermediate": t_intermediate,
            "text_layers": t_layers,
            "text_heads": t_heads,
            "text_kv_heads": t_kv,
            "text_head_dim": t_head_dim,
            "text_vocab": vocab,
            "spatial_merge_size": merge,
        },
    }


def sdk_record():
    if not SDK.is_file():
        return {
            "path": str(SDK),
            "sha256": None,
            "size_bytes": None,
            "expected_prefix": EXPECTED_SDK_PREFIX,
            "matches_b66_prefix": False,
            "cublaslt_version_expected": CUBLASLT_EXPECTED,
            "library_loaded": False,
        }
    digest = sha256_file(SDK)
    return {
        "path": str(SDK),
        "sha256": digest,
        "sha256_prefix": digest[:6],
        "size_bytes": SDK.stat().st_size,
        "expected_prefix": EXPECTED_SDK_PREFIX,
        "matches_b66_prefix": digest.startswith(EXPECTED_SDK_PREFIX),
        "cublaslt_version_expected": CUBLASLT_EXPECTED,
        "library_loaded": False,
        "load_performed": False,
        "c_api_symbols": [
            "tei_error",
            "tei_last_error_kind",
            "tei_device_preflight",
            "tei_create",
            "tei_weight",
            "tei_load_safetensors",
            "tei_option",
            "tei_ready",
            "tei_forward",
            "tei_forward_submit",
            "tei_output_wait",
            "tei_output_destroy",
            "tei_destroy",
        ],
    }


def build_cases(config):
    vision = config["vision_config"]
    text = config["text_config"]
    patch = int(vision["patch_size"])
    merge = int(config["spatial_merge_size"])
    patch_dim = 3 * patch * patch
    image_token_index = int(config["image_token_index"])

    def image_case(name, height, width, sequence_tokens, pool, token_start):
        rows = height * width
        token_count = (height // merge) * (width // merge)
        return {
            "name": name,
            "kind": "image_text_reference",
            "layout": "pooled",
            "pool": pool,
            "grid_thw": [1, height, width],
            "patch_dim": patch_dim,
            "patch_rows": rows,
            "pixel_values": rows * patch_dim,
            "merge_size": merge,
            "image_token_index": image_token_index,
            "image_token_start": token_start,
            "image_token_count": token_count,
            "sequence_tokens": sequence_tokens,
            "batch": 1,
            "pooled_rows": [0],
            "raw_rows": [],
            "expected_output_elements": int(text["hidden_size"]) if pool == 1 else 1,
            "pixel_seed": 0x4D495354,
        }

    return {
        "schema": "oxide-native-b66-mistral3-cases/v1",
        "note": "Reference inputs are deterministic synthetic tokens/pixels; no tokenizer or payload read occurs here.",
        "image_reference": [
            image_case("image-mean-1x4x6", 4, 6, 12, 1, 2),
            image_case("image-score-1x8x8", 8, 8, 24, 4, 2),
        ],
        "text_reference": [
            {
                "name": "text-mean-16",
                "kind": "text_reference",
                "layout": "pooled",
                "pool": 1,
                "tokens": 16,
                "batch": 1,
                "pooled_rows": [0],
                "raw_rows": [],
                "expected_output_elements": int(text["hidden_size"]),
                "token_seed": 0x4D495354,
            },
            {
                "name": "text-mean-128",
                "kind": "text_reference",
                "layout": "pooled",
                "pool": 1,
                "tokens": 128,
                "batch": 1,
                "pooled_rows": [0],
                "raw_rows": [],
                "expected_output_elements": int(text["hidden_size"]),
                "token_seed": 0x4D495354,
            },
            {
                "name": "text-score-32",
                "kind": "text_reference",
                "layout": "pooled",
                "pool": 4,
                "tokens": 32,
                "batch": 1,
                "pooled_rows": [0],
                "raw_rows": [],
                "expected_output_elements": 1,
                "token_seed": 0x4D495354,
            },
        ],
        "lifecycle": {
            "steps": [
                "create engine from the original byte-identical config; parse nested YaRN/llama_4_scaling in memory and emit tei_option keys without rewriting the checkpoint",
                "load all 531 tensors from 4 shards with tei_load_safetensors",
                "ready",
                "synchronous tei_forward image+text reference",
                "tei_forward_submit -> tei_output_wait -> tei_output_destroy",
                "tei_destroy",
            ],
            "expected_tensor_count": 531,
            "expected_shard_count": 4,
            "deferred_capacity_rule": "exact expected output element count",
        },
        "perf": {
            "warmup_iterations": 3,
            "measured_iterations": 20,
            "modes": ["synchronous", "deferred_submit_wait"],
            "report": ["min_ns", "max_ns", "mean_ns", "p50_ns", "output_checksum"],
            "claim": "No performance claim is made until Root runs on a real GPU.",
        },
    }


def main():
    if not CHECKPOINT.is_dir():
        raise SystemExit(f"checkpoint directory not found: {CHECKPOINT}")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    config_path = CHECKPOINT / "config.json"
    index_path = CHECKPOINT / "model.safetensors.index.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)

    index, tensors, shards, total_payload = inventory()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    index_sha = sha256_file(index_path)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "original-config.json").write_bytes(config_bytes)
    if sha256_file(OUTPUT / "original-config.json") != config_sha:
        raise ValueError("original-config.json copy is not byte-identical")

    names = sorted(tensors)
    vision, projector, language, score, unclassified = categorize(names)
    if unclassified:
        raise ValueError(f"unclassified tensors: {unclassified[:8]}")
    if len(vision) + len(projector) + len(language) + len(score) != len(names):
        raise ValueError("category counts do not account for every tensor")
    categories = {
        "vision": len(vision),
        "projector": len(projector),
        "language": len(language),
        "score": len(score),
        "retained": len(names),
        "unclassified": len(unclassified),
    }
    top_level = sorted(name for name in names if ".layers." not in name)
    validation = shape_rules(tensors, config)
    if not validation["all_passed"]:
        raise ValueError(
            "expected Mistral3 tensor shape validation failed: "
            + json.dumps(validation["failures"][:8])
        )

    inventory_doc = {
        "schema": INVENTORY_SCHEMA,
        "checkpoint": str(CHECKPOINT),
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "tensor_count": len(tensors),
        "shard_count": len(shards),
        "total_payload_bytes": total_payload,
        "index_metadata": index.get("metadata", {}),
        "categories": {name: sorted(group) for name, group in {
            "vision": vision,
            "projector": projector,
            "language": language,
            "score": score,
        }.items()},
        "shards": shards,
        "tensors": tensors,
    }
    inventory_text = json.dumps(inventory_doc, indent=2, sort_keys=True) + "\n"
    (OUTPUT / "tensor-inventory.json").write_text(inventory_text)
    inventory_sha = hashlib.sha256(inventory_text.encode()).hexdigest()

    cases = build_cases(config)
    sdk = sdk_record()
    report = {
        "schema": SCHEMA,
        "checkpoint": str(CHECKPOINT),
        "payload_read": False,
        "download_performed": False,
        "network_used": False,
        "torch_imported": False,
        "native_library_loaded": False,
        "cuda_runtime_called": False,
        "cuda_context_created": False,
        "cuda_visible_devices": cuda_visible,
        "cuda_hidden": cuda_visible == "",
        "config_sha256": config_sha,
        "index_sha256": index_sha,
        "config_preserved": config,
        "config_preserved_path": "original-config.json",
        "config_preserved_byte_identical": True,
        "text_config_sha256": hashlib.sha256(
            json.dumps(config["text_config"], sort_keys=True).encode()
        ).hexdigest(),
        "tensor_count": len(tensors),
        "shard_count": len(shards),
        "total_payload_bytes": total_payload,
        "index_and_headers_agree": True,
        "category_counts": categories,
        "vision_tensor_names": vision,
        "projector_tensor_names": projector,
        "score_tensor_names": score,
        "top_level_tensor_names": top_level,
        "tensor_inventory": "tensor-inventory.json",
        "tensor_inventory_sha256": inventory_sha,
        "tensor_inventory_doc_sha256": hashlib.sha256(
            json.dumps(tensors, sort_keys=True).encode()
        ).hexdigest(),
        "shard_header_inventory": shards,
        "expected_tensor_shape_validation": validation,
        "sdk": sdk,
        "cublaslt_version_expected": CUBLASLT_EXPECTED,
        "cases": cases,
        "root_gpu_followup": {
            "run_now": False,
            "requires": [
                "Root-provided real GPU host",
                "Root-provided CUDA and Torch environment",
                "frozen b66 libtei_torch.so (sha256 prefix b6607c) loaded by Root",
                "CUDA_VISIBLE_DEVICES explicitly set by Root (for example 0)",
            ],
            "steps": [
                "Root compiles native/include/mistral3_b66.h and native/src/mistral3_runner.cpp against libtei_torch.so.",
                "Root sets CUDA_VISIBLE_DEVICES=0 and runs the runner; it refuses an empty/unset guard.",
                "Create the engine from the original nested config and load all 531 tensors from 4 shards; no drops.",
                "Run text/image mean-pooled reference cases and classification score cases.",
                "Run synchronous and deferred submit/wait lifecycle cases and record perf counters.",
                "Publish no quality or performance claim until trained reference outputs and real GPU results exist.",
            ],
            "non_claims": [
                "This control did not import Torch.",
                "This control did not load libtei_torch.so.",
                "This control did not create a CUDA context or query a GPU.",
                "This control did not read tensor payload bytes.",
                "Shape checks are metadata checks, not numerical parity checks.",
            ],
        },
    }

    (OUTPUT / "CPU-control.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "tensor_count": len(tensors),
                "shard_count": len(shards),
                "categories": categories,
                "vision_tensor_names": len(vision),
                "projector_tensor_names": len(projector),
                "top_level_tensor_names": len(top_level),
                "shape_validation_passed": validation["all_passed"],
                "shape_checks": validation["checks_total"],
                "sdk_sha256": sdk["sha256"],
                "sdk_matches_b66": sdk["matches_b66_prefix"],
                "cuda_visible_devices": cuda_visible,
                "cuda_hidden": cuda_visible == "",
                "payload_read": False,
                "torch_imported": False,
                "native_library_loaded": False,
                "outputs": [
                    "original-config.json",
                    "tensor-inventory.json",
                    "CPU-control.json",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
