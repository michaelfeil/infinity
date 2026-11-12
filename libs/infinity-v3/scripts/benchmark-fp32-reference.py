#!/usr/bin/env python3
"""Independent HuggingFace FP32 numerical smoke reference generator.

Runs one local forward pass per pinned common-model probe and records the raw
float32 result. This is a numerical smoke reference for cross-runtime
comparison, not a task-accuracy or ranking-quality evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
ALLOWED_TASKS = ("embeddings", "rerank")
ALLOWED_DEVICES = ("cpu", "cuda:0")
REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
METADATA_SUFFIXES = frozenset(
    {".json", ".py", ".txt", ".model", ".tiktoken", ".jinja", ".yaml", ".yml",
     ".cfg", ".spm", ".md"}
)
POOLING_KEYS = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_lasttoken": "last",
}
POOLER_KEYS = frozenset({"pooler.dense.weight", "pooler.dense.bias"})
STANDARD_EMBEDDING_TYPES = {
    "bert": ("BertModel", "transformers.models.bert.modeling_bert"),
    "roberta": ("RobertaModel", "transformers.models.roberta.modeling_roberta"),
    "xlm-roberta": ("XLMRobertaModel",
                    "transformers.models.xlm_roberta.modeling_xlm_roberta"),
}
STANDARD_EMBEDDING_CLASSES = {
    name: module for name, module in STANDARD_EMBEDDING_TYPES.values()
}
COMMON_HELPER = HERE / "benchmark-common.py"
INTEGRITY_HELPER = HERE / "benchmark-integrity.py"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
LOADER_ENVIRONMENT_VARIABLES = (
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME")
EXPECTED_LOADER_POLICY = {
    "version": 1,
    "ld_library_path": "empty",
    "ld_preload": "unset",
    "ld_audit": "unset",
    "pythonpath": "unset",
    "pythonhome": "unset",
    "python_safe_path": True,
}


class SmokeError(Exception):
    pass


class UnavailableError(SmokeError):
    pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_sha256(path):
    path = Path(path)
    if not path.is_dir():
        return ""
    files = {}

    def visit(directory, prefix, ancestors):
        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            return
        if resolved in ancestors:
            return
        ancestors = ancestors | {resolved}
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return
        for item in entries:
            if item.name in {"__pycache__", ".git"}:
                continue
            relative = prefix + item.name
            if item.is_dir():
                visit(item, relative + "/", ancestors)
            elif item.is_file() and item.suffix.lower() in METADATA_SUFFIXES:
                files[relative] = _sha256_file(item)

    visit(path, "", frozenset())
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path, document):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".benchmark-fp32-reference-",
                                             dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_client_module():
    path = HERE / "benchmark-http.py"
    if not path.is_file():
        raise SmokeError(f"client helper not found: {path.name}")
    spec = importlib.util.spec_from_file_location("_benchmark_http_client", path)
    if spec is None or spec.loader is None:
        raise SmokeError(f"could not import client helper: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sibling_module(name, filename):
    path = HERE / filename
    if not path.is_file():
        raise SmokeError(f"helper not found: {filename}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SmokeError(f"could not import helper: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SourceGuard:
    def __init__(self, paths):
        self._paths = dict(paths)
        self._hashes = {name: _sha256_file(path)
                        for name, path in self._paths.items()}

    def hashes(self):
        return dict(self._hashes)

    def verify(self):
        for name, path in self._paths.items():
            if not path.is_file() or _sha256_file(path) != self._hashes[name]:
                raise SmokeError(f"{name} helper source changed during run")
        return dict(self._hashes)


def _load_helpers():
    guard = _SourceGuard({
        "checkpoint_helper": COMMON_HELPER,
        "integrity_helper": INTEGRITY_HELPER,
    })
    common = _load_sibling_module("_benchmark_common", COMMON_HELPER.name)
    integrity = _load_sibling_module("_benchmark_integrity", INTEGRITY_HELPER.name)
    guard.verify()
    return guard, common, integrity


def _validate_loader_environment(environ=None, flags=None):
    environ = os.environ if environ is None else environ
    flags = sys.flags if flags is None else flags
    if not bool(getattr(flags, "safe_path", False)):
        raise SmokeError("PYTHONSAFEPATH=1 is required for FP32 reference runs")
    present = [name for name in LOADER_ENVIRONMENT_VARIABLES if name in environ]
    if present:
        raise SmokeError("loader environment must leave unset: "
                         f"{', '.join(present)}")
    return True


def _loader_policy(common):
    policy = common.loader_policy("sglang")
    if policy != EXPECTED_LOADER_POLICY:
        raise SmokeError("sglang loader policy does not match the frozen reference "
                         "policy")
    return policy


def _validate_hash(value, label):
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise SmokeError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _portable_dependencies(items, label):
    if not isinstance(items, list):
        raise SmokeError(f"{label} must be a list")
    result = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise SmokeError(f"{label}[{position}] must be an object")
        identifier = item.get("id")
        revision = item.get("revision")
        if not isinstance(identifier, str) or not identifier:
            raise SmokeError(f"{label}[{position}] lacks a valid id")
        if not isinstance(revision, str) or not revision:
            raise SmokeError(f"{label}[{position}] lacks a valid revision")
        metadata = _validate_hash(item.get("metadata_sha256"),
                                  f"{label}[{position}].metadata_sha256")
        result.append({"id": identifier, "revision": revision,
                       "metadata_sha256": metadata})
    return result


def _dependency_declarations(items, label):
    if items is None:
        return []
    if not isinstance(items, list):
        raise SmokeError(f"{label} must be a list")
    result = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise SmokeError(f"{label}[{position}] must be an object")
        identifier = item.get("id")
        revision = item.get("revision")
        path = item.get("path")
        if not isinstance(identifier, str) or not identifier:
            raise SmokeError(f"{label}[{position}] lacks a valid id")
        if not isinstance(revision, str) or not revision:
            raise SmokeError(f"{label}[{position}] lacks a valid revision")
        if not isinstance(path, str) or not path:
            raise SmokeError(f"{label}[{position}] lacks a valid path")
        try:
            normalized = str(Path(path).resolve())
        except (OSError, RuntimeError) as exc:
            raise SmokeError(f"{label}[{position}] path cannot be resolved") from exc
        result.append({"id": identifier, "revision": revision,
                       "path": normalized})
    return result


def _validate_weight_map(weight_map, label, canonical):
    if not isinstance(weight_map, dict) or not weight_map:
        raise SmokeError(f"{label} must be a non-empty hash map")
    for name, digest in weight_map.items():
        if not isinstance(name, str) or not name:
            raise SmokeError(f"{label} contains an invalid file name")
        if os.path.isabs(name) or name.startswith(("/", "\\")) \
                or ".." in re.split(r"[\\/]+", name):
            raise SmokeError(f"{label} contains a non-portable file name")
        _validate_hash(digest, f"{label}[{name!r}]")
    return canonical(weight_map)


def _validated_checkpoint_identity(receipt, model, receipt_path, fresh, canonical):
    if not isinstance(fresh, dict):
        raise SmokeError("checkpoint identity result must be an object")
    raw = fresh.get("checkpoint")
    identity = fresh.get("identity")
    if not isinstance(raw, dict) or not isinstance(identity, dict):
        raise SmokeError("checkpoint identity result lacks checkpoint/identity")
    for label, record in (("checkpoint", raw), ("identity", identity)):
        if record.get("id") != receipt.get("id"):
            raise SmokeError(f"{label} id does not match the validated receipt")
        if str(record.get("revision", "")).lower() \
                != str(receipt.get("revision", "")).lower():
            raise SmokeError(f"{label} revision does not match the validated receipt")
    if identity["id"] != model["id"] \
            or str(identity["revision"]).lower() != model["revision"].lower():
        raise SmokeError("checkpoint identity does not match the manifest model")
    metadata = _validate_hash(identity.get("metadata_sha256"),
                              "identity.metadata_sha256")
    if raw.get("metadata_sha256") != metadata:
        raise SmokeError("checkpoint metadata identity mismatch")
    weight_map = raw.get("weight_files_sha256")
    aggregate = _validate_weight_map(weight_map, "checkpoint.weight_files_sha256",
                                     canonical)
    weights = _validate_hash(identity.get("weights_sha256"),
                             "identity.weights_sha256")
    if raw.get("weights_sha256") != weights or aggregate != weights:
        raise SmokeError("checkpoint weight identity mismatch")
    dependencies = _portable_dependencies(identity.get("dependencies"),
                                          "identity.dependencies")
    raw_dependencies = _portable_dependencies(raw.get("dependencies"),
                                               "checkpoint.dependencies")
    if dependencies != raw_dependencies:
        raise SmokeError("checkpoint dependency identity mismatch")
    declared = _dependency_declarations(receipt.get("dependencies"),
                                        "receipt.dependencies")
    raw_declared = _dependency_declarations(raw.get("dependencies"),
                                            "checkpoint.dependencies")
    if declared != raw_declared:
        raise SmokeError("checkpoint dependency path identity differs from receipt")
    if [(item["id"], item["revision"]) for item in dependencies] \
            != [(item["id"], item["revision"]) for item in declared]:
        raise SmokeError("checkpoint dependency identity differs from receipt")
    code_dependencies = _portable_dependencies(identity.get("code_dependencies"),
                                               "identity.code_dependencies")
    raw_code_dependencies = _portable_dependencies(
        raw.get("code_dependencies"), "checkpoint.code_dependencies")
    if code_dependencies != raw_code_dependencies:
        raise SmokeError("checkpoint code dependency identity mismatch")
    declared_code = _dependency_declarations(receipt.get("code_dependencies"),
                                             "receipt.code_dependencies")
    raw_declared_code = _dependency_declarations(
        raw.get("code_dependencies"), "checkpoint.code_dependencies")
    if declared_code != raw_declared_code:
        raise SmokeError("checkpoint code dependency path identity differs from "
                         "receipt")
    if [(item["id"], item["revision"]) for item in code_dependencies] \
            != [(item["id"], item["revision"]) for item in declared_code]:
        raise SmokeError("checkpoint code dependency identity differs from receipt")
    return {
        "id": identity["id"],
        "revision": identity["revision"],
        "status": receipt.get("status"),
        "receipt_sha256": _sha256_file(receipt_path),
        "metadata_sha256": metadata,
        "weights_sha256": weights,
        "weight_files_sha256": dict(sorted(weight_map.items())),
        "dependencies": dependencies,
        "code_dependencies": code_dependencies,
    }


def _environment_receipt(guard, integrity):
    guard.verify()
    receipt = integrity.fingerprint_runtime()
    guard.verify()
    if not isinstance(receipt, dict):
        raise SmokeError("python environment receipt must be an object")
    digest = _validate_hash(receipt.get("digest_sha256"),
                            "python_environment.digest_sha256")
    source = _validate_hash(receipt.get("integrity_source_sha256"),
                            "python_environment.integrity_source_sha256")
    if source != guard.hashes()["integrity_helper"]:
        raise SmokeError("python environment receipt helper source mismatch")
    interpreter = receipt.get("interpreter")
    if not isinstance(interpreter, dict) or interpreter.get("safe_path") is not True:
        raise SmokeError("python environment receipt does not bind safe_path")
    if "private_roots" in receipt or "private_executable" in receipt:
        raise SmokeError("python environment receipt contains private paths")
    return receipt


def _semantic_sha256(task, item):
    if task == "rerank":
        payload = {"query": item["query"], "document": item["document"]}
    elif task == "embeddings":
        payload = {"text": item["text"]}
    else:
        raise SmokeError(f"unsupported task {task!r}")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _pool_dense(hidden, attention_mask, pooling):
    if not torch.is_tensor(hidden) or hidden.dim() != 3:
        raise SmokeError("dense last_hidden_state must have shape [batch, tokens, hidden]")
    if hidden.shape[0] != 1:
        raise SmokeError("dense reference expects exactly one row")
    if hidden.shape[1] < 1:
        raise SmokeError("dense last_hidden_state must contain at least one position")
    if attention_mask is None:
        attention_mask = torch.ones(hidden.shape[:2], dtype=torch.long,
                                    device=hidden.device)
    if not torch.is_tensor(attention_mask) or attention_mask.dim() != 2:
        raise SmokeError("attention_mask must have shape [batch, tokens]")
    if tuple(attention_mask.shape) != tuple(hidden.shape[:2]):
        raise SmokeError("attention_mask shape does not match dense hidden states")
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype)
    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling == "mean":
        counts = mask.sum(dim=1)
        if bool((counts <= 0).any()):
            raise SmokeError("mean pooling requires at least one attended token")
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
    if pooling == "last":
        if bool((mask.sum(dim=1) <= 0).any()):
            raise SmokeError("last-token pooling requires at least one attended token")
        reversed_mask = torch.flip(mask, dims=(1,))
        offset = reversed_mask.argmax(dim=1)
        index = hidden.shape[1] - 1 - offset
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, index, :]
    raise SmokeError(f"unsupported dense pooling {pooling!r}")


def _normalize_l2(vector):
    if not torch.is_tensor(vector) or vector.numel() == 0:
        raise SmokeError("dense vector is empty")
    if not bool(torch.isfinite(vector).all()):
        raise SmokeError("dense vector contains a non-finite value")
    tensor = vector.reshape(1, -1).float()
    norm = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm <= 0).any()):
        raise SmokeError("dense vector has an invalid L2 norm")
    return tensor / norm


def _require_float32_tensor(tensor, label):
    if not torch.is_tensor(tensor):
        raise SmokeError(f"{label} is missing")
    if tensor.dtype != torch.float32:
        raise SmokeError(f"{label} dtype is {tensor.dtype}, expected float32")


def _assert_model_float32(model):
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise SmokeError(f"model parameter {name} dtype is {parameter.dtype}, "
                             "expected float32")


def _extract_rerank_scalar(logits):
    _require_float32_tensor(logits, "rerank logits")
    if logits.numel() != 1:
        raise SmokeError(f"rerank logits must be exactly one scalar, got shape "
                         f"{list(logits.shape)}")
    value = float(logits.detach().to(dtype=torch.float32, device="cpu").reshape(-1)[0])
    if not math.isfinite(value):
        raise SmokeError("rerank logit is non-finite")
    return value


def _rerank_reference(logits):
    value = _extract_rerank_scalar(logits)
    return {"values": [value], "shape": [1], "logits_shape": list(logits.shape)}


def _validate_model(model):
    if not isinstance(model, dict):
        raise SmokeError("manifest model entry must be an object")
    model_id = model.get("id")
    revision = model.get("revision")
    if not isinstance(model_id, str) or not model_id:
        raise SmokeError("manifest model id must be a non-empty string")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        raise SmokeError(f"{model_id}.revision must be a full 40-character hex string")
    task = model.get("task")
    if task not in ALLOWED_TASKS:
        raise SmokeError(f"{model_id}.task {task!r} is not one of {ALLOWED_TASKS}")
    lengths = model.get("lengths")
    if not isinstance(lengths, list) or not lengths:
        raise SmokeError(f"{model_id}.lengths must be a non-empty list")
    seen = set()
    for length in lengths:
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise SmokeError(f"{model_id}.lengths contains invalid value {length!r}")
        if length in seen:
            raise SmokeError(f"{model_id}.lengths contains duplicate {length}")
        seen.add(length)
    parameters = model.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, (int, float)) or isinstance(parameters, bool):
            raise SmokeError(f"{model_id}.parameters must be numeric or null")
        if parameters < 0:
            raise SmokeError(f"{model_id}.parameters must be non-negative")
    selection = model.get("selection")
    if selection is not None and not isinstance(selection, str):
        raise SmokeError(f"{model_id}.selection must be a string")
    pooling = model.get("pooling")
    if pooling is not None and pooling not in ("cls", "mean", "last"):
        raise SmokeError(f"{model_id}.pooling {pooling!r} is not cls/mean/last")
    pooling_source = model.get("pooling_source")
    if pooling_source is not None and not isinstance(pooling_source, str):
        raise SmokeError(f"{model_id}.pooling_source must be a string")


def _load_manifest(input_dir):
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SmokeError(f"manifest.json not found under {input_dir.name}")
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SmokeError("manifest.json must contain an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SmokeError(f"unsupported manifest schema_version "
                         f"{manifest.get('schema_version')!r}")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise SmokeError("manifest models must be a non-empty list")
    seen = set()
    for model in models:
        _validate_model(model)
        if model["id"] in seen:
            raise SmokeError(f"duplicate manifest model id {model['id']!r}")
        seen.add(model["id"])
    return manifest, manifest_path


def _selected_models(models, only):
    if not only:
        return models
    requested = [item.strip() for item in only.split(",") if item.strip()]
    if not requested:
        raise SmokeError("--only did not contain a model id")
    by_id = {model["id"]: model for model in models}
    missing = [item for item in requested if item not in by_id]
    if missing:
        raise SmokeError(f"--only contains unknown model id(s): {', '.join(missing)}")
    keep = set(requested)
    return [model for model in models if model["id"] in keep]


def _local_path(value, base):
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _dependency_entries(receipt, key):
    value = receipt.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise SmokeError(f"receipt {key} must be a list")
    entries = []
    for item in value:
        if not isinstance(item, dict):
            raise SmokeError(f"receipt {key} entries must be objects")
        entries.append(item)
    return entries


def _normalize_receipt_paths(input_dir, receipt):
    normalized = dict(receipt)
    normalized["path"] = str(_local_path(receipt["path"], input_dir))
    for key in ("dependencies", "code_dependencies"):
        value = receipt.get(key)
        if value is None:
            continue
        if not isinstance(value, list):
            raise SmokeError(f"receipt {key} must be a list")
        items = []
        for position, item in enumerate(value):
            if not isinstance(item, dict):
                raise SmokeError(f"receipt {key}[{position}] must be an object")
            entry = dict(item)
            raw = entry.get("path")
            if isinstance(raw, str) and raw:
                entry["path"] = str(_local_path(raw, input_dir))
            items.append(entry)
        normalized[key] = items
    return normalized


def _validated_receipt(input_dir, model):
    checkpoints = input_dir / "checkpoints"
    slug = model["id"].replace("/", "--")
    receipt_path = checkpoints / f"{slug}.json"
    if not receipt_path.is_file():
        raise UnavailableError(f"checkpoint receipt missing: {slug}.json")
    receipt = _load_json(receipt_path)
    if not isinstance(receipt, dict):
        raise SmokeError(f"checkpoint receipt {slug}.json must be an object")
    if receipt.get("id") != model["id"]:
        raise SmokeError(f"receipt id {receipt.get('id')!r} does not match "
                         f"{model['id']!r}")
    if str(receipt.get("revision", "")).lower() != model["revision"].lower():
        raise SmokeError(f"receipt revision {receipt.get('revision')!r} does not "
                         f"match {model['revision']!r}")
    if receipt.get("status") != "ready":
        raise UnavailableError(f"checkpoint status is {receipt.get('status')!r}")
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise UnavailableError("checkpoint receipt has no local path")
    receipt = _normalize_receipt_paths(input_dir, receipt)
    checkpoint_path = Path(receipt["path"])
    if not checkpoint_path.is_dir():
        raise UnavailableError("checkpoint path is not a local directory")
    dependencies = _dependency_entries(receipt, "dependencies")
    if dependencies:
        raw = dependencies[0].get("path")
        if isinstance(raw, str) and raw and not Path(raw).is_dir():
            raise UnavailableError("checkpoint dependency0 path is not local")
    code_dependencies = _dependency_entries(receipt, "code_dependencies")
    return receipt, checkpoint_path, dependencies, code_dependencies, receipt_path


def _tokenizer_path(checkpoint_path, dependencies):
    if dependencies and isinstance(dependencies[0], dict) \
            and dependencies[0].get("path"):
        return Path(dependencies[0]["path"])
    return checkpoint_path


def _dense_pooling(model, checkpoint_path):
    modules_path = checkpoint_path / "modules.json"
    if not modules_path.is_file():
        raise UnavailableError("missing sentence-transformers modules.json")
    modules = _load_json(modules_path)
    if not isinstance(modules, list) or len(modules) not in (2, 3):
        raise UnavailableError("modules.json must contain the standard 2/3-stage "
                               "sentence-transformers pipeline")
    expected = (
        ("sentence_transformers.models.Transformer", ""),
        ("sentence_transformers.models.Pooling", "1_Pooling"),
    )
    for index, (module_type, module_path) in enumerate(expected):
        module = modules[index]
        if not isinstance(module, dict) or module.get("type") != module_type \
                or module.get("path") != module_path:
            raise UnavailableError("unsupported sentence-transformers pipeline at "
                                   f"module {index}")
    if len(modules) == 3:
        module = modules[2]
        if not isinstance(module, dict) \
                or module.get("type") != "sentence_transformers.models.Normalize" \
                or module.get("path") != "2_Normalize":
            raise UnavailableError("unsupported sentence-transformers pipeline at "
                                   "module 2")
    pooling_path = checkpoint_path / "1_Pooling" / "config.json"
    if not pooling_path.is_file():
        raise UnavailableError("missing pinned 1_Pooling/config.json")
    config = _load_json(pooling_path)
    if not isinstance(config, dict):
        raise UnavailableError("1_Pooling/config.json must contain an object")
    enabled = sorted(key for key, value in config.items()
                     if isinstance(key, str) and key.startswith("pooling_mode")
                     and value is True)
    if len(enabled) != 1 or enabled[0] not in POOLING_KEYS:
        raise UnavailableError("unsupported pooling configuration "
                               f"{enabled or 'none'}")
    pooling = POOLING_KEYS[enabled[0]]
    declared = model.get("pooling")
    if declared is not None and declared != pooling:
        raise SmokeError(f"manifest pooling {declared!r} conflicts with pinned "
                         f"1_Pooling/config.json {pooling!r}")
    return pooling


def _prepare_probe(client, tokenizer, safe, specials, task, length):
    workload = client.Workload(tokenizer, safe, specials, task, length)
    item = workload.raw_item(0)
    sample_sha256 = _semantic_sha256(task, item)
    if sample_sha256 != item.get("sha256"):
        raise SmokeError("probe semantic hash does not match client raw_item(0)")
    if task == "rerank":
        encoded = tokenizer(item["query"], item["document"],
                            add_special_tokens=True, truncation=False,
                            padding=False, return_tensors="pt")
    else:
        encoded = tokenizer(item["text"], add_special_tokens=True,
                            truncation=False, padding=False, return_tensors="pt")
    if "input_ids" not in encoded or encoded["input_ids"].dim() != 2:
        raise SmokeError("tokenizer did not produce a single-row input_ids tensor")
    local_tokens = int(item["local_tokens"])
    if int(encoded["input_ids"].shape[1]) != local_tokens:
        raise SmokeError(f"tokenizer produced {int(encoded['input_ids'].shape[1])} "
                         f"tokens but probe declares {local_tokens}")
    return item, sample_sha256, local_tokens, encoded


def _standard_embedding_type(checkpoint_path):
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = _load_json(config_path)
    except Exception:
        return None
    if not isinstance(config, dict) or config.get("auto_map"):
        return None
    model_type = config.get("model_type")
    return model_type if model_type in STANDARD_EMBEDDING_TYPES else None


def _class_identity(model):
    return f"{type(model).__module__}.{type(model).__name__}"


def _is_standard_embedding_class(model):
    module = STANDARD_EMBEDDING_CLASSES.get(type(model).__name__)
    return module is not None and type(model).__module__ == module


def _validate_loading_info(task, model, info, add_pooling_layer, scrub):
    if not isinstance(info, dict):
        raise UnavailableError("from_pretrained did not return loading_info")

    def clean_set(name):
        return sorted({_scrub_text(str(item), scrub)
                       for item in (info.get(name) or [])})

    missing = clean_set("missing_keys")
    unexpected = clean_set("unexpected_keys")
    mismatched = clean_set("mismatched_keys")
    errors = [_scrub_text(str(item), scrub)
              for item in (info.get("error_msgs") or [])]
    allowed = set()
    if task == "embeddings" and _is_standard_embedding_class(model):
        allowed = set(POOLER_KEYS)
    rejected_missing = sorted(set(missing) - allowed)
    rejected_unexpected = sorted(set(unexpected) - allowed)
    if mismatched or errors or rejected_missing or rejected_unexpected:
        raise UnavailableError(
            "loading_info rejected: "
            f"model_class={_class_identity(model)}, "
            f"add_pooling_layer={add_pooling_layer}, "
            f"missing_keys={missing}, unexpected_keys={unexpected}, "
            f"mismatched_keys={mismatched}, error_msgs={errors}, "
            f"rejected_missing_keys={rejected_missing}, "
            f"rejected_unexpected_keys={rejected_unexpected}")
    unused = sorted((set(missing) | set(unexpected)) & allowed)
    return {
        "model_class": _class_identity(model),
        "add_pooling_layer": add_pooling_layer,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "error_msgs": errors,
        "allowed_unused_pooler_keys": unused,
        "validated": True,
    }


def _load_model(checkpoint_path, task, scrub=None):
    kwargs = {
        "dtype": torch.float32,
        "attn_implementation": "eager",
        "local_files_only": True,
        "trust_remote_code": True,
    }
    add_pooling_layer = None
    standard_type = None
    if task == "embeddings":
        standard_type = _standard_embedding_type(checkpoint_path)
        if standard_type is not None:
            add_pooling_layer = False
            kwargs["add_pooling_layer"] = False
    loader = AutoModel if task == "embeddings" else AutoModelForSequenceClassification
    result = loader.from_pretrained(str(checkpoint_path),
                                    output_loading_info=True, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise UnavailableError("from_pretrained did not return (model, loading_info)")
    model, info = result
    if add_pooling_layer is False:
        expected_name, expected_module = STANDARD_EMBEDDING_TYPES[standard_type]
        if type(model).__name__ != expected_name \
                or type(model).__module__ != expected_module:
            raise UnavailableError(
                f"add_pooling_layer=False requested for {standard_type} but loaded "
                f"{_class_identity(model)}; expected {expected_module}.{expected_name}")
        if getattr(model, "pooler", None) is not None:
            raise UnavailableError(
                "add_pooling_layer=False did not remove pooler for "
                f"{_class_identity(model)}")
    loading = _validate_loading_info(task, model, info, add_pooling_layer,
                                     scrub or {})
    return model, loading


def _scrub_text(text, replacements):
    result = str(text)
    for path, token in sorted(replacements.items(), key=lambda pair: len(pair[0]),
                              reverse=True):
        if path:
            result = result.replace(path, token)
    result = re.sub(r"/(?:root|home|tmp|mnt|data|scratch|workspace)"
                    r"(?:/[^\s'\"():,\[\]{}]+)+", "<path>", result)
    return result


def _status_for(exc):
    if isinstance(exc, UnavailableError):
        return "unavailable"
    if isinstance(exc, SmokeError):
        return "invalid"
    return "error"


def _failure(model, length, status, reason, checkpoint=None, sample_sha256=None,
             local_tokens=None, scrub=None):
    record = {
        "model": model["id"],
        "revision": model["revision"],
        "task": model["task"],
        "length": length,
        "status": status,
        "reason": _scrub_text(reason, scrub or {}),
    }
    if checkpoint is not None:
        record["checkpoint"] = checkpoint
    if sample_sha256 is not None:
        record["sample_sha256"] = sample_sha256
    if local_tokens is not None:
        record["local_tokens"] = local_tokens
    return record


def _run_model(input_dir, model, device, client, output_path, guard, common):
    lengths = list(model["lengths"])
    references = []
    failures = []
    replacements = {str(input_dir): "<input>", str(output_path): "<output>"}
    checkpoint = None
    try:
        receipt, checkpoint_path, dependencies, code_dependencies, receipt_path = \
            _validated_receipt(input_dir, model)
        replacements[str(checkpoint_path)] = "<checkpoint>"
        for dependency in dependencies + code_dependencies:
            raw = dependency.get("path")
            if isinstance(raw, str) and raw:
                replacements[str(Path(raw))] = "<dependency>"
        guard.verify()
        try:
            fresh = common.checkpoint_identity(receipt, {})
        except Exception as exc:
            raise SmokeError(f"checkpoint content identity failed: {exc}") from exc
        guard.verify()
        checkpoint = _validated_checkpoint_identity(
            receipt, model, receipt_path, fresh, common.canonical_sha256)
        tokenizer_path = _tokenizer_path(checkpoint_path, dependencies)
        replacements[str(tokenizer_path)] = "<tokenizer>"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return references, [
            _failure(model, length, _status_for(exc), reason, checkpoint,
                     scrub=replacements) for length in lengths
        ]

    pooling = None
    if model["task"] == "embeddings":
        try:
            pooling = _dense_pooling(model, checkpoint_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return references, [
                _failure(model, length, _status_for(exc), reason, checkpoint,
                         scrub=replacements) for length in lengths
            ]

    try:
        tokenizer = client._load_tokenizer(str(tokenizer_path))
        safe = client._safe_words(tokenizer)
        specials = client._token_len(tokenizer, "")
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return references, [
            _failure(model, length, "unavailable", reason, checkpoint,
                     scrub=replacements) for length in lengths
        ]

    probes = {}
    for length in lengths:
        try:
            probes[length] = _prepare_probe(
                client, tokenizer, safe, specials, model["task"], length)
        except Exception as exc:
            failures.append(_failure(
                model, length, _status_for(exc), f"{type(exc).__name__}: {exc}",
                checkpoint, scrub=replacements))

    if not probes:
        return references, failures

    model_instance = None
    try:
        model_instance, loading = _load_model(
            checkpoint_path, model["task"], replacements)
        model_instance.eval()
        model_instance.to(device)
        _assert_model_float32(model_instance)
        for length in lengths:
            if length not in probes:
                continue
            _item, sample_sha256, local_tokens, encoded = probes[length]
            inputs = {name: value.to(device) for name, value in encoded.items()
                      if torch.is_tensor(value)}
            try:
                with torch.inference_mode():
                    outputs = model_instance(**inputs)
                if model["task"] == "embeddings":
                    hidden = getattr(outputs, "last_hidden_state", None)
                    if hidden is None:
                        raise SmokeError("model did not return last_hidden_state")
                    _require_float32_tensor(hidden, "last_hidden_state")
                    vector = _normalize_l2(
                        _pool_dense(hidden, inputs.get("attention_mask"), pooling))
                    values = [float(value) for value in
                              vector.detach().to(dtype=torch.float32,
                                                 device="cpu").reshape(-1).tolist()]
                    if not values or not all(math.isfinite(value) for value in values):
                        raise SmokeError("dense vector is empty or non-finite")
                    references.append({
                        "model": model["id"],
                        "revision": model["revision"],
                        "task": model["task"],
                        "length": length,
                        "sample_sha256": sample_sha256,
                        "local_tokens": local_tokens,
                        "pooling": pooling,
                        "pooling_source": "1_Pooling/config.json",
                        "normalize": True,
                        "normalization": "l2",
                        "values": values,
                        "shape": [len(values)],
                        "all_finite": True,
                        "dtype": "float32",
                        "device": device,
                        "checkpoint": checkpoint,
                        "loading": loading,
                    })
                else:
                    logits = getattr(outputs, "logits", None)
                    rerank_fields = _rerank_reference(logits)
                    references.append({
                        "model": model["id"],
                        "revision": model["revision"],
                        "task": model["task"],
                        "length": length,
                        "sample_sha256": sample_sha256,
                        "local_tokens": local_tokens,
                        "pooling": "none",
                        "normalize": False,
                        "normalization": "none",
                        **rerank_fields,
                        "all_finite": True,
                        "dtype": "float32",
                        "device": device,
                        "checkpoint": checkpoint,
                        "loading": loading,
                    })
            except Exception as exc:
                failures.append(_failure(
                    model, length, _status_for(exc),
                    f"{type(exc).__name__}: {exc}", checkpoint, sample_sha256,
                    local_tokens, replacements))
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        status = _status_for(exc) if isinstance(exc, SmokeError) else "unavailable"
        remaining = [length for length in lengths if length in probes]
        for length in remaining:
            _item, sample_sha256, local_tokens, _encoded = probes[length]
            failures.append(_failure(
                model, length, status, reason, checkpoint,
                sample_sha256, local_tokens, replacements))
    finally:
        if model_instance is not None:
            del model_instance
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return references, failures


def _base_document(manifest_path, selected, args, client_path, helper_hashes,
                   loader_policy, python_environment):
    client_sha256 = _sha256_file(client_path)
    source_sha256 = _sha256_file(Path(__file__))
    manifest_sha256 = _sha256_file(manifest_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "independent_hf_fp32_numerical_smoke_reference",
        "method": "Independent HuggingFace AutoModel/AutoModelForSequenceClassification "
                  "float32 forward pass with eager attention and TF32 disabled on "
                  "CUDA; dense vectors are L2-normalized and rerank logits are raw "
                  "single scalars. HF loading_info is validated; standard BERT/"
                  "Roberta/XLM dense encoders load without an unused pooler, and only "
                  "exact unused pooler weights may be absent/unexpected for those "
                  "proven classes. Numerical smoke reference only; not task-accuracy "
                  "or ranking-quality evaluation.",
        "source_sha256": source_sha256,
        "client_sha256": client_sha256,
        "manifest_sha256": manifest_sha256,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": args.device,
        "threads": args.threads,
        "dtype": "float32",
        "source_hashes": {
            "reference_source": source_sha256,
            "http_client_helper": client_sha256,
            "manifest": manifest_sha256,
            "checkpoint_helper": helper_hashes["checkpoint_helper"],
            "integrity_helper": helper_hashes["integrity_helper"],
        },
        "python_environment_sha256": python_environment["digest_sha256"],
        "python_environment": python_environment,
        "loader_policy": loader_policy,
        "input_layout": {"manifest": "manifest.json", "checkpoints": "checkpoints"},
        "model_counts": {
            "selected": len(selected),
            "embeddings": sum(model["task"] == "embeddings" for model in selected),
            "rerank": sum(model["task"] == "rerank" for model in selected),
        },
        "counts": {"references": 0, "failures": 0, "models_attempted": 0},
        "references": [],
        "failures": [],
    }


def _run(args):
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SmokeError("--input must be a local directory")
    output_path = Path(args.output)
    if not output_path.is_absolute():
        raise SmokeError("--output must be an absolute path")
    if args.threads < 1:
        raise SmokeError("--threads must be >= 1")
    if args.device not in ALLOWED_DEVICES:
        raise SmokeError(f"--device must be one of {ALLOWED_DEVICES}")
    _validate_loader_environment()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SmokeError("--device cuda:0 requested but CUDA is unavailable")
    manifest, manifest_path = _load_manifest(input_dir)
    selected = _selected_models(manifest["models"], args.only)
    client = _load_client_module()
    guard, common, integrity = _load_helpers()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    python_environment = _environment_receipt(guard, integrity)
    loader_policy = _loader_policy(common)
    guard.verify()
    document = _base_document(manifest_path, selected, args,
                              HERE / "benchmark-http.py", guard.hashes(),
                              loader_policy, python_environment)
    _write_json_atomic(output_path, document)
    for model in selected:
        references, failures = _run_model(
            input_dir, model, args.device, client, output_path, guard, common)
        document["references"].extend(references)
        document["failures"].extend(failures)
        document["counts"] = {
            "references": len(document["references"]),
            "failures": len(document["failures"]),
            "models_attempted": len({item["model"] for item in
                                     document["references"] + document["failures"]}),
        }
        _write_json_atomic(output_path, document)
    _write_json_atomic(output_path, document)
    return 0


def _expect_error(function, label, error=SmokeError):
    try:
        function()
    except error:
        return
    except SmokeError as exc:
        raise AssertionError(f"expected {error.__name__}: {label}; got "
                             f"{type(exc).__name__}") from exc
    raise AssertionError(f"expected {error.__name__}: {label}")


def _write_fixture(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _pipeline_fixture(root, modules):
    root.mkdir(parents=True, exist_ok=True)
    _write_fixture(root / "modules.json", modules)
    _write_fixture(root / "1_Pooling" / "config.json", {
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_lasttoken": False,
    })
    (root / "2_Normalize").mkdir(exist_ok=True)


def _tiny_tokenizer(client):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    words = sorted(set(client.WORDS) | {"the"})
    vocab = {token: index for index, token in enumerate(specials + words)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]")


def _self_test(args):
    input_dir = Path(args.input).resolve() if args.input else None
    tests = []

    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    mean = _pool_dense(hidden, mask, "mean")
    assert torch.allclose(mean, torch.tensor([[2.0, 2.0]]))
    tests.append("masked mean excludes padding")

    left_hidden = torch.tensor([[[-1.0, -1.0], [5.0, 6.0], [7.0, 8.0]]])
    left_mask = torch.tensor([[0, 1, 1]])
    last = _pool_dense(left_hidden, left_mask, "last")
    assert torch.allclose(last, torch.tensor([[7.0, 8.0]]))
    right_last = _pool_dense(hidden, mask, "last")
    assert torch.allclose(right_last, torch.tensor([[3.0, 3.0]]))
    cls = _pool_dense(hidden, mask, "cls")
    assert torch.allclose(cls, torch.tensor([[1.0, 1.0]]))
    tests.append("last attended handles left/right padding and cls uses index 0")

    unit = _normalize_l2(torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(unit, torch.tensor([[0.6, 0.8]]))
    _expect_error(lambda: _normalize_l2(torch.empty(0)), "empty vector")
    _expect_error(lambda: _normalize_l2(torch.tensor([float("nan")])), "nan vector")
    _expect_error(lambda: _normalize_l2(torch.tensor([0.0])), "zero norm")
    tests.append("L2 normalization validates empty/non-finite/zero vectors")

    rerank_fields = _rerank_reference(torch.tensor([[-3.5]]))
    assert rerank_fields == {"values": [-3.5], "shape": [1],
                             "logits_shape": [1, 1]}
    zero_fields = _rerank_reference(torch.tensor([[0.0]]))
    assert zero_fields["values"] == [0.0] and zero_fields["shape"] == [1]
    _expect_error(lambda: _rerank_reference(torch.tensor([[1.0, 2.0]])),
                  "two logits")
    _expect_error(lambda: _rerank_reference(torch.tensor([float("inf")])),
                  "non-finite logit")
    _expect_error(lambda: _rerank_reference(
        torch.tensor([[1.0]], dtype=torch.bfloat16)), "bf16 logits")
    tests.append("rerank keeps raw finite zero/negative scalar with shape [1]")

    _assert_model_float32(torch.nn.Linear(2, 2))
    _require_float32_tensor(torch.ones(1), "float32 output")
    _expect_error(lambda: _assert_model_float32(
        torch.nn.Linear(2, 2).to(torch.bfloat16)), "mixed model precision")
    _expect_error(lambda: _require_float32_tensor(
        torch.ones(1, dtype=torch.bfloat16), "logits"), "bf16 output")
    tests.append("mixed-precision parameters and outputs are rejected")

    from transformers import BertModel

    standard_model = object.__new__(BertModel)
    custom_class = type("BertModel", (), {})
    custom_class.__module__ = "custom.module"
    custom_model = custom_class()
    for field in ("missing_keys", "unexpected_keys"):
        provenance = _validate_loading_info(
            "embeddings", standard_model, {field: sorted(POOLER_KEYS)}, False, {})
        assert provenance["allowed_unused_pooler_keys"] == sorted(POOLER_KEYS)
        assert provenance["validated"] is True
    _expect_error(lambda: _validate_loading_info(
        "embeddings", standard_model,
        {"missing_keys": ["encoder.layer.0.attention.self.query.weight"]},
        False, {}), "required encoder missing", UnavailableError)
    _expect_error(lambda: _validate_loading_info(
        "embeddings", standard_model, {"missing_keys": ["classifier.weight"]},
        False, {}), "required classifier missing", UnavailableError)
    _expect_error(lambda: _validate_loading_info(
        "embeddings", standard_model,
        {"missing_keys": ["pooler.projection.weight"]},
        False, {}), "unknown pooler prefix", UnavailableError)
    _expect_error(lambda: _validate_loading_info(
        "embeddings", custom_model, {"missing_keys": sorted(POOLER_KEYS)},
        False, {}), "custom class pooler", UnavailableError)
    _expect_error(lambda: _validate_loading_info(
        "embeddings", standard_model, {"mismatched_keys": ["pooler.dense.weight"]},
        False, {}), "mismatched key", UnavailableError)
    try:
        _validate_loading_info(
            "embeddings", standard_model,
            {"error_msgs": ["bad /home/private/x"]}, False, {})
    except UnavailableError as exc:
        assert "/home" not in str(exc)
    else:
        raise AssertionError("expected loading error")
    rerank_clean = _validate_loading_info("rerank", standard_model, {}, None, {})
    assert rerank_clean["validated"] is True
    assert rerank_clean["allowed_unused_pooler_keys"] == []
    _expect_error(lambda: _validate_loading_info(
        "rerank", standard_model, {"missing_keys": ["classifier.weight"]},
        None, {}), "rerank missing classifier", UnavailableError)
    tests.append("loading guard rejects required/mismatch/error/custom keys and "
                 "accepts only exact unused standard-encoder pooler keys; rerank "
                 "requires a clean report")

    model = {"id": "org/model", "revision": "a" * 40, "task": "embeddings",
             "lengths": [128]}
    for bad in ("main", "a" * 39, "a" * 41):
        _expect_error(lambda bad=bad: _validate_model(
            {**model, "revision": bad}), f"bad revision {bad}")
    tests.append("manifest revision requires a full 40-character hex string")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        standard = [
            {"type": "sentence_transformers.models.Transformer", "path": ""},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
        ]
        _pipeline_fixture(root / "good2", standard)
        assert _dense_pooling(model, root / "good2") == "cls"
        standard3 = standard + [
            {"type": "sentence_transformers.models.Normalize",
             "path": "2_Normalize"},
        ]
        _pipeline_fixture(root / "good3", standard3)
        assert _dense_pooling(model, root / "good3") == "cls"
        _pipeline_fixture(root / "dense", standard + [
            {"type": "sentence_transformers.models.Dense", "path": "2_Dense"}])
        _pipeline_fixture(root / "duplicate", [standard[0], standard[0]])
        _pipeline_fixture(root / "order", [standard[1], standard[0]])
        _expect_error(lambda: _dense_pooling(model, root / "dense"),
                      "dense module", UnavailableError)
        _expect_error(lambda: _dense_pooling(model, root / "duplicate"),
                      "duplicate transformer", UnavailableError)
        _expect_error(lambda: _dense_pooling(model, root / "order"),
                      "out of order pipeline", UnavailableError)
    tests.append("2/3-stage module pipeline accepts standard order and rejects "
                 "Dense/duplicate/out-of-order")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = root / "first"
        (first / "sub").mkdir(parents=True)
        (first / "config.json").write_text('{"x": 1}', encoding="utf-8")
        (first / "sub" / "model.py").write_text("x = 1\n", encoding="utf-8")
        (first / "weights.safetensors").write_bytes(b"excluded")
        (first / "__pycache__").mkdir()
        (first / "__pycache__" / "ignored.py").write_text("x = 1\n",
                                                          encoding="utf-8")
        (first / ".git").mkdir()
        (first / ".git" / "config").write_text("ignored", encoding="utf-8")
        digest = _metadata_sha256(first)
        relocated = root / "relocated"
        shutil.copytree(first, relocated)
        assert _metadata_sha256(relocated) == digest
        _write_fixture(root / "receipt.json", {"id": "org/model"})
        assert _metadata_sha256(first) == digest
    tests.append("metadata tree hash is relocatable and receipt-independent")

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        checkpoint = fixture / "checkpoint"
        checkpoint.mkdir()
        dependency = fixture / "dependency"
        dependency.mkdir()
        (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
        (dependency / "tokenizer.json").write_text("{}", encoding="utf-8")
        slug = model["id"].replace("/", "--")
        receipt_path = fixture / "checkpoints" / f"{slug}.json"
        good = {
            "id": model["id"],
            "revision": model["revision"],
            "status": "ready",
            "path": str(checkpoint),
            "dependencies": [{"id": "org/dependency",
                              "revision": "b" * 40,
                              "path": str(dependency)}],
        }
        _write_fixture(receipt_path, good)
        receipt, path, dependencies, code_dependencies, _ = _validated_receipt(
            fixture, model)
        assert receipt["status"] == "ready"
        assert _tokenizer_path(path, dependencies) == dependency
        assert code_dependencies == []
        _write_fixture(receipt_path, {**good, "dependencies": [
            good["dependencies"][0],
            {"id": "org/second", "revision": "d" * 40,
             "path": str(fixture / "missing-second")}]})
        receipt, path, dependencies, code_dependencies, _ = _validated_receipt(
            fixture, model)
        assert _tokenizer_path(path, dependencies) == dependency
        _write_fixture(receipt_path, {**good, "dependencies": [
            {"id": "org/dependency", "revision": "b" * 40,
             "path": str(fixture / "missing")}]})
        _expect_error(lambda: _validated_receipt(fixture, model),
                      "missing dependency", UnavailableError)
        _write_fixture(receipt_path, {**good, "id": "org/other"})
        _expect_error(lambda: _validated_receipt(fixture, model),
                      "wrong receipt id")
        _write_fixture(receipt_path, {**good, "revision": "b" * 40})
        _expect_error(lambda: _validated_receipt(fixture, model),
                      "wrong receipt revision")
        code_only = {
            "id": model["id"],
            "revision": model["revision"],
            "status": "ready",
            "path": str(checkpoint),
            "code_dependencies": [{"id": "org/code",
                                   "revision": "c" * 40,
                                   "path": str(dependency)}],
        }
        _write_fixture(receipt_path, code_only)
        receipt, path, dependencies, code_dependencies, _ = _validated_receipt(
            fixture, model)
        assert dependencies == []
        assert _tokenizer_path(path, dependencies) == path
        assert code_dependencies[0]["id"] == "org/code"
    tests.append("dependency0 wins tokenizer selection; code_dependencies are "
                 "provenance only; bad receipt id/revision/dependency fail")

    guard, common, integrity = _load_helpers()
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        checkpoint = fixture / "checkpoint"
        dependency = fixture / "dependency"
        code_dependency = fixture / "code-dependency"
        for directory in (checkpoint, dependency, code_dependency):
            directory.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"weights-v1")
        (checkpoint / "config.json").write_text('{"model_type": "bert"}',
                                                encoding="utf-8")
        (dependency / "pytorch_model.bin").write_bytes(b"dependency-weights-v1")
        (dependency / "config.json").write_text('{"dep": 1}', encoding="utf-8")
        (code_dependency / "custom.py").write_text("VALUE = 1\n", encoding="utf-8")
        (code_dependency / "custom.bin").write_bytes(b"code-weights-v1")
        slug = model["id"].replace("/", "--")
        receipt_path = fixture / "checkpoints" / f"{slug}.json"
        _write_fixture(receipt_path, {
            "id": model["id"],
            "revision": model["revision"],
            "status": "ready",
            "path": "checkpoint",
            "dependencies": [{"id": "org/dependency",
                              "revision": "b" * 40, "path": "dependency",
                              "metadata_sha256": "0" * 64}],
            "code_dependencies": [{"id": "org/code",
                                   "revision": "c" * 40,
                                   "path": "code-dependency"}],
        })
        receipt, path, _dependencies, _code_dependencies, _ = _validated_receipt(
            fixture, model)
        assert path == checkpoint.resolve()
        assert receipt["path"] == str(checkpoint.resolve())
        assert receipt["dependencies"][0]["metadata_sha256"] == "0" * 64
        assert "metadata_sha256" not in receipt["code_dependencies"][0]
        guard.verify()
        fresh = common.checkpoint_identity(receipt, {})
        checkpoint_identity = _validated_checkpoint_identity(
            receipt, model, receipt_path, fresh, common.canonical_sha256)
        serialized = json.dumps(checkpoint_identity)
        assert str(fixture) not in serialized and '"path"' not in serialized
        assert set(checkpoint_identity) == {
            "id", "revision", "status", "receipt_sha256", "metadata_sha256",
            "weights_sha256", "weight_files_sha256", "dependencies",
            "code_dependencies"}
        names = checkpoint_identity["weight_files_sha256"]
        assert any(name.startswith("dependencies/") for name in names)
        assert any(name.startswith("code_dependencies/") for name in names)
        first_weights = checkpoint_identity["weights_sha256"]
        first_dependency = checkpoint_identity["dependencies"][0]
        first_code = checkpoint_identity["code_dependencies"][0]
        (checkpoint / "model.safetensors").write_bytes(b"weights-v2")
        changed = _validated_checkpoint_identity(
            receipt, model, receipt_path,
            common.checkpoint_identity(receipt, {}), common.canonical_sha256)
        assert changed["weights_sha256"] != first_weights
        after_main_weights = changed["weights_sha256"]
        dependency_weight = next(name for name in names
                                 if name.startswith("dependencies/"))
        (dependency / "pytorch_model.bin").write_bytes(
            b"dependency-weights-v2")
        changed = _validated_checkpoint_identity(
            receipt, model, receipt_path,
            common.checkpoint_identity(receipt, {}), common.canonical_sha256)
        assert changed["weights_sha256"] != after_main_weights
        assert changed["weight_files_sha256"][dependency_weight] \
            != names[dependency_weight]
        (dependency / "config.json").write_text('{"dep": 2}', encoding="utf-8")
        changed = _validated_checkpoint_identity(
            receipt, model, receipt_path,
            common.checkpoint_identity(receipt, {}), common.canonical_sha256)
        assert changed["dependencies"][0]["metadata_sha256"] \
            != first_dependency["metadata_sha256"]
        (code_dependency / "custom.py").write_text("VALUE = 2\n", encoding="utf-8")
        changed = _validated_checkpoint_identity(
            receipt, model, receipt_path,
            common.checkpoint_identity(receipt, {}), common.canonical_sha256)
        assert changed["code_dependencies"][0]["metadata_sha256"] \
            != first_code["metadata_sha256"]
        malformed = json.loads(json.dumps(fresh))
        malformed["checkpoint"]["weight_files_sha256"] = {}
        _expect_error(lambda: _validated_checkpoint_identity(
            receipt, model, receipt_path, malformed, common.canonical_sha256),
            "empty weight map")
        malformed = json.loads(json.dumps(fresh))
        name = next(iter(malformed["checkpoint"]["weight_files_sha256"]))
        malformed["checkpoint"]["weight_files_sha256"][name] = "bad-hash"
        _expect_error(lambda: _validated_checkpoint_identity(
            receipt, model, receipt_path, malformed, common.canonical_sha256),
            "invalid weight file hash")
        malformed = json.loads(json.dumps(fresh))
        malformed["checkpoint"]["dependencies"][0]["metadata_sha256"] = "f" * 64
        _expect_error(lambda: _validated_checkpoint_identity(
            receipt, model, receipt_path, malformed, common.canonical_sha256),
            "dependency identity mismatch")
        client_probe = mock.Mock()
        malformed = json.loads(json.dumps(fresh))
        malformed["checkpoint"]["weight_files_sha256"] = {"model.safetensors":
                                                          "bad-hash"}
        with mock.patch.object(common, "checkpoint_identity",
                               return_value=malformed):
            references, failures = _run_model(
                fixture, model, "cpu", client_probe, fixture / "out.json",
                guard, common)
        assert references == []
        assert len(failures) == len(model["lengths"])
        assert all(item["status"] == "invalid" for item in failures)
        assert client_probe._load_tokenizer.call_count == 0
    tests.append("checkpoint content identity binds weight/dependency/code bytes "
                 "after path normalization, omits host paths, and rejects "
                 "malformed identity before tokenizer/client use")

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        checkpoint = fixture / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"main-weights-v1")
        (checkpoint / "config.json").write_text('{"model_type": "bert"}',
                                                encoding="utf-8")
        slug = model["id"].replace("/", "--")
        receipt_path = fixture / "checkpoints" / f"{slug}.json"
        _write_fixture(receipt_path, {
            "id": model["id"],
            "revision": model["revision"],
            "status": "ready",
            "path": "checkpoint",
        })
        receipt, _path, dependencies, code_dependencies, _ = _validated_receipt(
            fixture, model)
        assert receipt.get("dependencies") is None
        assert receipt.get("code_dependencies") is None
        assert dependencies == [] and code_dependencies == []
        fresh = common.checkpoint_identity(receipt, {})
        identity = _validated_checkpoint_identity(
            receipt, model, receipt_path, fresh, common.canonical_sha256)
        assert identity["dependencies"] == []
        assert identity["code_dependencies"] == []
        client_probe = mock.Mock()
        malformed = json.loads(json.dumps(fresh))
        malformed["checkpoint"]["weight_files_sha256"] = {}
        with mock.patch.object(common, "checkpoint_identity",
                               return_value=malformed), \
                mock.patch.object(torch.cuda, "is_available",
                                  side_effect=AssertionError("GPU probe")):
            references, failures = _run_model(
                fixture, model, "cpu", client_probe, fixture / "out.json",
                guard, common)
        assert references == []
        assert len(failures) == len(model["lengths"])
        assert all(item["status"] == "invalid" for item in failures)
        assert client_probe._load_tokenizer.call_count == 0
    tests.append("no-dependency main model receipt accepts absent declarations "
                 "and rejects malformed identity before tokenizer/GPU work")

    environment = {
        "schema_version": 1,
        "kind": "python_environment_content_fingerprint",
        "interpreter": {"safe_path": True},
        "integrity_source_sha256": guard.hashes()["integrity_helper"],
        "digest_sha256": "a" * 64,
    }
    probe = mock.Mock(return_value=environment)
    bound = _environment_receipt(
        guard, SimpleNamespace(fingerprint_runtime=probe))
    assert probe.call_count == 1 and bound is environment
    policy = _loader_policy(common)
    assert policy == EXPECTED_LOADER_POLICY
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest_path = root / "manifest.json"
        client_path = root / "client.py"
        manifest_path.write_text("{}", encoding="utf-8")
        client_path.write_text("", encoding="utf-8")
        document = _base_document(
            manifest_path, [model], SimpleNamespace(device="cpu", threads=1),
            client_path, guard.hashes(), policy, bound)
        assert document["source_sha256"]
        assert document["source_hashes"]["checkpoint_helper"] \
            == guard.hashes()["checkpoint_helper"]
        assert document["source_hashes"]["integrity_helper"] \
            == guard.hashes()["integrity_helper"]
        assert document["python_environment_sha256"] == "a" * 64
        assert document["python_environment"] is bound
        assert document["loader_policy"] == EXPECTED_LOADER_POLICY
        bad = dict(environment, private_roots=[])
        _expect_error(lambda: _environment_receipt(
            guard, SimpleNamespace(fingerprint_runtime=lambda: bad)),
            "private roots in environment")
        bad = dict(environment, integrity_source_sha256="b" * 64)
        _expect_error(lambda: _environment_receipt(
            guard, SimpleNamespace(fingerprint_runtime=lambda: bad)),
            "environment helper source mismatch")
        bad = dict(environment, interpreter={"safe_path": False})
        _expect_error(lambda: _environment_receipt(
            guard, SimpleNamespace(fingerprint_runtime=lambda: bad)),
            "environment safe path")
        source = root / "unstable-helper.py"
        source.write_text("v1", encoding="utf-8")
        source_guard = _SourceGuard({"helper": source})
        source.write_text("v2", encoding="utf-8")
        _expect_error(source_guard.verify, "concurrent helper edit")
    _expect_error(lambda: _validate_loader_environment(
        {}, SimpleNamespace(safe_path=False)), "unsafe python path")
    for name in LOADER_ENVIRONMENT_VARIABLES:
        _expect_error(lambda name=name: _validate_loader_environment(
            {name: ""}, SimpleNamespace(safe_path=True)), f"set {name}")
    assert _validate_loader_environment(
        {}, SimpleNamespace(safe_path=True)) is True
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        args = _parser().parse_args([
            "--input", str(root), "--output", str(root / "out.json"),
            "--device", "cuda:0"])
        module = sys.modules[__name__]
        with mock.patch.object(sys, "flags",
                               SimpleNamespace(safe_path=False)), \
                mock.patch.object(torch.cuda, "is_available",
                                  side_effect=AssertionError("cuda probe")), \
                mock.patch.object(module, "_load_client_module",
                                  side_effect=AssertionError("client load")), \
                mock.patch.object(module, "_load_helpers",
                                  side_effect=AssertionError("helper load")):
            _expect_error(lambda: _run(args), "unsafe run")
    tests.append("header binds helper bytes, one mocked environment probe, and "
                 "sglang loader policy; unsafe loader environment fails before "
                 "CUDA/client/model work and helper edits fail closed")

    failure = _failure(model, 128, "error",
                       "OSError: /home/private/model/config.json")
    assert "/home" not in json.dumps(failure)
    assert "<path>" in failure["reason"]
    scrubbed = _failure(model, 128, "error",
                        f"OSError: {model['id']}/config.json",
                        scrub={model["id"]: "<checkpoint>"})
    assert "<checkpoint>" in scrubbed["reason"]
    tests.append("failure reasons scrub host paths")

    dense = {"text": "a b c", "local_tokens": 3}
    assert _semantic_sha256("embeddings", dense) == hashlib.sha256(
        json.dumps({"text": "a b c"}, sort_keys=True).encode()).hexdigest()
    pair = {"query": "q", "document": "d"}
    assert _semantic_sha256("rerank", pair) == hashlib.sha256(
        json.dumps({"query": "q", "document": "d"},
                   sort_keys=True).encode()).hexdigest()
    tests.append("semantic hash uses canonical original payload")

    client = _load_client_module()
    tokenizer = _tiny_tokenizer(client)
    safe = client._safe_words(tokenizer)
    specials = client._token_len(tokenizer, "")
    for task in ALLOWED_TASKS:
        item, sample_sha256, local_tokens, encoded = _prepare_probe(
            client, tokenizer, safe, specials, task, 128)
        assert sample_sha256 == item["sha256"]
        assert int(encoded["input_ids"].shape[1]) == local_tokens
    tests.append("standalone WordLevel tokenizer raw_item(0) semantic hash and "
                 "token count match client for dense/rerank")

    if input_dir is not None:
        manifest, _manifest_path = _load_manifest(input_dir)
        cached = next((item for item in manifest["models"]
                       if item["id"] == "BAAI/bge-small-en-v1.5"), None)
        if cached is None:
            raise AssertionError("self-test requires cached BAAI/bge-small-en-v1.5")
        receipt, checkpoint_path, dependencies, _code_dependencies, _ = \
            _validated_receipt(input_dir, cached)
        del receipt
        tokenizer = client._load_tokenizer(
            str(_tokenizer_path(checkpoint_path, dependencies)))
        safe = client._safe_words(tokenizer)
        specials = client._token_len(tokenizer, "")
        for task in ALLOWED_TASKS:
            item, sample_sha256, local_tokens, encoded = _prepare_probe(
                client, tokenizer, safe, specials, task, 128)
            assert sample_sha256 == item["sha256"]
            assert int(encoded["input_ids"].shape[1]) == local_tokens
        tests.append("cached BAAI Workload raw_item(0) semantic hash and token "
                     "count match client")

    print(json.dumps({"self_test": "ok", "tests": tests}, indent=2))
    return 0


def _parser():
    parser = argparse.ArgumentParser(
        description="Independent HuggingFace FP32 numerical smoke reference.")
    parser.add_argument("--input",
                        help="campaign directory with manifest.json and checkpoints/; "
                             "required for generation, optional for --self-test")
    parser.add_argument("--output", help="absolute output JSON path")
    parser.add_argument("--device", default="cpu", choices=ALLOWED_DEVICES)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--only", help="optional comma-separated model ids")
    parser.add_argument("--self-test", action="store_true",
                        help="run offline CPU self-tests; no model download or GPU")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            return _self_test(args)
        if not args.input:
            parser.error("--input is required for generation")
        if not args.output:
            parser.error("--output is required")
        return _run(args)
    except SmokeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
