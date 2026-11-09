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
    checkpoint_path = _local_path(raw_path, input_dir)
    if not checkpoint_path.is_dir():
        raise UnavailableError("checkpoint path is not a local directory")
    dependencies = _dependency_entries(receipt, "dependencies")
    if dependencies:
        raw = dependencies[0].get("path")
        if isinstance(raw, str) and raw and not _local_path(raw, input_dir).exists():
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


def _checkpoint_identity(receipt, receipt_path, checkpoint_path,
                         code_dependencies):
    return {
        "id": receipt.get("id"),
        "revision": receipt.get("revision"),
        "status": receipt.get("status"),
        "receipt_sha256": _sha256_file(receipt_path),
        "metadata_sha256": _metadata_sha256(checkpoint_path),
        "code_dependencies": [
            {"id": item["id"], "revision": item["revision"]}
            for item in code_dependencies
            if item.get("id") and item.get("revision")
        ],
    }


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


def _run_model(input_dir, model, device, client, output_path):
    lengths = list(model["lengths"])
    references = []
    failures = []
    replacements = {str(input_dir): "<input>", str(output_path): "<output>"}
    checkpoint = None
    try:
        receipt, checkpoint_path, dependencies, code_dependencies, receipt_path = \
            _validated_receipt(input_dir, model)
        checkpoint = _checkpoint_identity(receipt, receipt_path, checkpoint_path,
                                          code_dependencies)
        replacements[str(checkpoint_path)] = "<checkpoint>"
        for dependency in dependencies + code_dependencies:
            raw = dependency.get("path")
            if isinstance(raw, str) and raw:
                replacements[str(_local_path(raw, input_dir))] = "<dependency>"
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


def _base_document(manifest_path, selected, args, client_path):
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
        },
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SmokeError("--device cuda:0 requested but CUDA is unavailable")
    manifest, manifest_path = _load_manifest(input_dir)
    selected = _selected_models(manifest["models"], args.only)
    client = _load_client_module()
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    document = _base_document(manifest_path, selected, args,
                              HERE / "benchmark-http.py")
    _write_json_atomic(output_path, document)
    for model in selected:
        references, failures = _run_model(
            input_dir, model, args.device, client, output_path)
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
