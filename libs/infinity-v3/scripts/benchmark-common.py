#!/usr/bin/env python3
"""Prepare and run a reproducible dense-embedding / reranker HTTP campaign.

``prepare`` pins every Hub snapshot to the immutable revision in the manifest,
materialises code dependencies, and writes one receipt per checkpoint.  ``run``
executes Infinity, vLLM and SGLang serially on each GPU lane and records portable
fingerprints, full client measurements and owned server/client logs.

Only processes started by this script are stopped.  Existing results are
reused solely when their run fingerprint matches exactly; terminal failures are
retried only with ``--retry-failures``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "benchmark-common-models.json"
DEFAULT_SERVER_CPUS = ("0-7", "8-15", "16-23", "24-31", "64-71", "72-79")
DEFAULT_CLIENT_CPUS = ("32-39", "40-47", "48-55", "56-63", "80-87", "88-95")
RUNTIMES = ("infinity", "vllm", "sglang")
TASKS = frozenset({"embeddings", "rerank"})
TERMINAL_STATUSES = frozenset(
    {"measured", "client_failed", "load_failed", "checkpoint_unavailable"}
)
SENTENCE_CONFIG_FILES = (
    "sentence_bert_config.json", "sentence_roberta_config.json",
    "sentence_distilbert_config.json", "sentence_camembert_config.json",
    "sentence_albert_config.json", "sentence_xlm-roberta_config.json",
    "sentence_xlnet_config.json",
)
CODE_PATTERNS = ("*.json", "*.txt", "*.model", "*.tiktoken", "*.py", "*.jinja",
                 "*.yaml", "*.yml", "*.cfg", "*.spm")
METADATA_SUFFIXES = frozenset(
    {".json", ".py", ".txt", ".model", ".tiktoken", ".jinja", ".yaml", ".yml",
     ".cfg", ".spm", ".md"}
)
PATH_KEYS = frozenset({"path", "cached_path", "cache_path", "local_dir", "local_path"})
REVISION_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class CampaignError(ValueError):
    """A manifest, campaign or lane placement is invalid."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def _revision(value) -> str:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        raise CampaignError(f"revision must be a full immutable 40/64-hex SHA, got {value!r}")
    return value.lower()


def _dependency_list(value, label: str):
    if value is None:
        return []
    if not isinstance(value, list):
        raise CampaignError(f"{label} must be a list")
    result = []
    for position, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise CampaignError(f"{label}[{position}] must be an object")
        if not isinstance(entry.get("id"), str) or not entry["id"].strip():
            raise CampaignError(f"{label}[{position}].id must be a non-empty string")
        result.append({"id": entry["id"], "revision": _revision(entry.get("revision"))})
    return result


def validate_manifest(document) -> list[dict]:
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise CampaignError("manifest must be an object with a models list")
    if not document["models"]:
        raise CampaignError("manifest.models must not be empty")
    models, seen = [], set()
    for position, raw in enumerate(document["models"]):
        if not isinstance(raw, dict):
            raise CampaignError(f"manifest.models[{position}] must be an object")
        model_id = raw.get("id")
        if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
            raise CampaignError(f"manifest.models[{position}].id is invalid")
        if model_id in seen:
            raise CampaignError(f"duplicate manifest model id {model_id!r}")
        seen.add(model_id)
        if raw.get("task") not in TASKS:
            raise CampaignError(f"{model_id}.task must be one of {sorted(TASKS)}")
        _revision(raw.get("revision"))
        lengths = raw.get("lengths")
        if not isinstance(lengths, list) or not lengths:
            raise CampaignError(f"{model_id}.lengths must be a non-empty list")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in lengths):
            raise CampaignError(f"{model_id}.lengths must contain positive integers")
        if len(set(lengths)) != len(lengths):
            raise CampaignError(f"{model_id}.lengths must be unique")
        _dependency_list(raw.get("dependencies"), f"{model_id}.dependencies")
        _dependency_list(raw.get("code_dependencies"), f"{model_id}.code_dependencies")
        models.append(raw)
    return models


def portable(value):
    """Drop absolute paths so fingerprints survive relocation."""
    if isinstance(value, dict):
        return {key: portable(item) for key, item in sorted(value.items())
                if key not in PATH_KEYS}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        return Path(value).name
    return value


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise CampaignError(f"{path} is not valid JSON: {exc}") from exc


def load_models(args) -> tuple[dict, list[dict]]:
    source = Path(args.manifest)
    snapshot = args.output / "manifest.json"
    if not source.is_file() and snapshot.is_file():
        source = snapshot
    document = _load_json(source)
    models = validate_manifest(document)
    if snapshot.is_file():
        if _load_json(snapshot) != document:
            raise CampaignError("input manifest differs from output/manifest.json snapshot")
    else:
        save_json(snapshot, document)
    selected = models
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        selected = [model for model in models if model["id"] in wanted]
        missing = wanted - {model["id"] for model in selected}
        if missing:
            raise CampaignError(f"unknown model ID(s) in --only: {sorted(missing)}")
    return document, selected


def metadata_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_dir():
        return ""
    files = {}

    def visit(directory: Path, prefix: str, ancestors: frozenset) -> None:
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
                files[relative] = sha256_file(item)

    visit(path, "", frozenset())
    return sha256_bytes(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())


def weight_bytes(path: Path) -> int:
    total = 0
    if Path(path).is_dir():
        for file in Path(path).rglob("*"):
            if file.is_file() and file.suffix in {".safetensors", ".bin"}:
                total += file.stat().st_size
    return total


def _config_scope(path: Path) -> dict:
    config = _load_json(Path(path) / "config.json")
    if not isinstance(config, dict):
        raise CampaignError(f"{path}/config.json must be an object")
    scope = config.get("text_config", config)
    return scope if isinstance(scope, dict) else config


def base_context(path: Path, requested: int) -> int:
    scope = _config_scope(path)
    limit = scope.get("max_position_embeddings", scope.get("n_positions"))
    if limit is None:
        return requested
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise CampaignError(f"Invalid max_position_embeddings in {path}/config.json")
    return min(requested, limit)


def vllm_context_limit(path: Path, requested: int) -> int:
    for filename in SENTENCE_CONFIG_FILES:
        source = Path(path) / filename
        if not source.is_file():
            continue
        encoder = json.loads(source.read_text(encoding="utf-8"))
        if not encoder:
            continue
        limit = encoder.get("max_seq_length")
        if limit is None:
            return requested
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise CampaignError(f"Invalid saved max_seq_length in {source}")
        return min(requested, limit)
    return requested


def _head_dim(path: Path) -> int:
    scope = _config_scope(path)
    hidden = scope.get("hidden_size", scope.get("d_model", 0))
    heads = scope.get("num_attention_heads", 1)
    if isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
        return hidden // heads
    return 0


def _external_auto_map_entries(config: dict):
    scopes = [config]
    if isinstance(config.get("text_config"), dict):
        scopes.append(config["text_config"])
    for scope in scopes:
        auto_map = scope.get("auto_map")
        if not isinstance(auto_map, dict):
            continue
        for value in auto_map.values():
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, str) and entry.strip():
                    yield entry.strip()


def validate_auto_map(path: Path, code_dependencies: list[dict]) -> None:
    config = _load_json(Path(path) / "config.json")
    for entry in _external_auto_map_entries(config):
        if "--" in entry:
            raise CampaignError(
                f"external auto_map reference {entry!r} cannot be pinned by the runtime; "
                "materialise and rewrite it locally before launching")
        module_name = entry.split(".", 1)[0]
        if not ((Path(path) / f"{module_name}.py").is_file()
                or (Path(path) / module_name).is_dir()):
            raise CampaignError(f"auto_map module {entry!r} is missing from checkpoint")


def _verify_pinned(api, repository: str, revision: str) -> None:
    info = api.model_info(repository, revision=revision)
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or resolved.lower() != revision.lower():
        raise CampaignError(
            f"{repository}: resolved revision {resolved!r} does not equal pinned {revision!r}")


def _download_snapshot(api, snapshot_download, repository: str, revision: str,
                       patterns, workers: int) -> str:
    _verify_pinned(api, repository, revision)
    return snapshot_download(repository, revision=revision,
                             allow_patterns=list(patterns), max_workers=workers)


def _download_model(model: dict, args):
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    revision = model["revision"]
    _verify_pinned(api, model["id"], revision)
    names = api.list_repo_files(model["id"], revision=revision)
    has_safetensors = any(name.endswith(".safetensors") for name in names)
    has_bin = any(name.endswith(".bin") for name in names)
    if not has_safetensors and not has_bin:
        raise CampaignError(f"{model['id']}: snapshot has no .safetensors or .bin weights")
    patterns = list(CODE_PATTERNS)
    patterns.append("*.safetensors" if has_safetensors else "*.bin")
    path = _download_snapshot(api, snapshot_download, model["id"], revision,
                              patterns, args.download_workers)
    code_dependencies = []
    for dependency in _dependency_list(model.get("code_dependencies"), "code_dependencies"):
        target = _download_snapshot(api, snapshot_download, dependency["id"],
                                    dependency["revision"], CODE_PATTERNS,
                                    args.download_workers)
        code_dependencies.append({**dependency, "path": target})
    dependencies = []
    for dependency in _dependency_list(model.get("dependencies"), "dependencies"):
        target = _download_snapshot(api, snapshot_download, dependency["id"],
                                    dependency["revision"], CODE_PATTERNS,
                                    args.download_workers)
        dependencies.append({**dependency, "path": target})
    vela = Path(path) / "vela3_config.json"
    if vela.is_file():
        base = json.loads(vela.read_text(encoding="utf-8")).get("base")
        if isinstance(base, dict) and (base.get("repo_id") or base.get("revision")):
            if not base.get("repo_id") or not base.get("revision"):
                raise CampaignError(f"{vela}: vela base lacks a pinned repo_id/revision")
            pinned = _revision(base["revision"])
            if pinned not in {item["revision"] for item in dependencies}:
                target = _download_snapshot(api, snapshot_download, base["repo_id"],
                                            pinned, CODE_PATTERNS, args.download_workers)
                dependencies.append({"id": base["repo_id"], "revision": pinned,
                                     "path": target})
    validate_auto_map(path, code_dependencies)
    return path, weight_bytes(Path(path)), dependencies, code_dependencies


def _link_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(item.resolve(), target)


TOKENIZER_PROBES = (
    "", "benchmark", "The quick brown fox jumps over the lazy dog.",
    "naive cafe \u6771\u4eac", "def f(x): return x + 1",
)


def _verified_fast_tokenizer(source: Path, loaded: Path, model: dict):
    from transformers import AutoTokenizer

    reference = AutoTokenizer.from_pretrained(
        str(source), use_fast=False, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(loaded), use_fast=True, local_files_only=True, trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False) or not hasattr(tokenizer, "backend_tokenizer"):
        raise CampaignError(f"{model['id']}: no fast tokenizer available for tokenizer.json")
    if (reference.padding_side, reference.truncation_side) != (
            tokenizer.padding_side, tokenizer.truncation_side):
        raise CampaignError(f"{model['id']}: fast tokenizer changed padding/truncation sides")
    probe_ids = []
    for text in TOKENIZER_PROBES:
        for add_special_tokens in (True, False):
            expected = reference.encode(text, add_special_tokens=add_special_tokens)
            actual = tokenizer.encode(text, add_special_tokens=add_special_tokens)
            if expected != actual:
                raise CampaignError(
                    f"{model['id']}: generated fast tokenizer changed token IDs for {text!r}")
            probe_ids.append({"text": text, "special": add_special_tokens, "ids": actual})
    digest = sha256_bytes(json.dumps(probe_ids, sort_keys=True,
                                     separators=(",", ":")).encode())
    return tokenizer, digest, len(probe_ids)


def _tokenizer_adapter(tokenizer, prepared: Path, probes: int, digest: str) -> dict:
    return {"kind": "transformers_fast_tokenizer_json", "generated": True,
            "hub_cache_modified": False, "generated_file": "tokenizer.json",
            "generated_sha256": sha256_file(prepared / "tokenizer.json"),
            "tokenizer_class": type(tokenizer).__name__,
            "id_probes": probes, "id_probe_sha256": digest,
            "padding": "unchanged", "truncation": "unchanged"}


def ensure_tokenizer(source: Path, model: dict, output: Path):
    source = Path(source)
    if (source / "tokenizer.json").is_file():
        return source, {"kind": "native_tokenizer_json", "generated": False,
                        "hub_cache_modified": False}
    prepared = output / "checkpoints" / (slug(model["id"]) + ".prepared")
    if (prepared / "tokenizer.json").is_file():
        tokenizer, digest, probes = _verified_fast_tokenizer(source, prepared, model)
        return prepared, _tokenizer_adapter(tokenizer, prepared, probes, digest)
    from transformers import AutoTokenizer

    _link_tree(source, prepared)
    tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True,
                                              local_files_only=True,
                                              trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False) or not hasattr(tokenizer, "backend_tokenizer"):
        raise CampaignError(f"{model['id']}: no fast tokenizer available for tokenizer.json")
    temporary = prepared / ".tokenizer.json.tmp"
    tokenizer.backend_tokenizer.save(str(temporary))
    os.replace(temporary, prepared / "tokenizer.json")
    tokenizer, digest, probes = _verified_fast_tokenizer(source, prepared, model)
    return prepared, _tokenizer_adapter(tokenizer, prepared, probes, digest)


def _declared_missing(receipt: dict, model: dict, key: str) -> bool:
    declared = {(item["id"], item["revision"])
                for item in _dependency_list(model.get(key), key)}
    present = {(item.get("id"), item.get("revision"))
               for item in receipt.get(key) or [] if isinstance(item, dict)}
    return not declared.issubset(present)


def _refresh_receipt(receipt: dict, model: dict, args):
    path, adapter = ensure_tokenizer(Path(receipt["path"]), model, args.output)
    receipt = dict(receipt)
    receipt.update(path=str(path), metadata_sha256=metadata_sha256(path),
                   weight_bytes=receipt.get("weight_bytes") or weight_bytes(path),
                   tokenizer_adapter=adapter)
    return receipt


def prepare_one(args, model: dict) -> dict:
    target = args.output / "checkpoints" / (slug(model["id"]) + ".json")
    old = None
    if target.is_file():
        try:
            old = _load_json(target)
        except CampaignError:
            old = None
    if old and not args.retry_failures:
        same_revision = str(old.get("revision", "")).lower() == model["revision"]
        if old.get("status") == "ready" and same_revision and old.get("path") \
                and Path(old["path"]).is_dir() \
                and not _declared_missing(old, model, "dependencies") \
                and not _declared_missing(old, model, "code_dependencies"):
            try:
                receipt = _refresh_receipt(old, model, args)
                save_json(target, receipt)
                return receipt
            except Exception as exc:
                old = {"id": model["id"], "status": "unavailable",
                       "revision": model["revision"],
                       "error": f"{type(exc).__name__}: {exc}"}
        elif old.get("status") != "ready" and same_revision:
            return old
    item = {"id": model["id"], "status": "unavailable", "revision": model["revision"]}
    try:
        path, size, dependencies, code_dependencies = _download_model(model, args)
        receipt = {"id": model["id"], "status": "ready", "path": str(path),
                   "revision": model["revision"], "weight_bytes": size,
                   "dependencies": dependencies,
                   "code_dependencies": code_dependencies}
        receipt = _refresh_receipt(receipt, model, args)
        item = receipt
    except Exception as exc:
        item["error"] = f"{type(exc).__name__}: {exc}"
    save_json(target, item)
    print(json.dumps({"checkpoint": item["id"], "status": item["status"]}), flush=True)
    return item


def prepare(args, models: list[dict]) -> None:
    if args.download_workers < 1:
        raise CampaignError("--download-workers must be >= 1")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        receipts = list(pool.map(lambda model: prepare_one(args, model), models))
    save_json(args.output / "checkpoints.json", receipts)


def _first(method: dict, names):
    for name in names:
        if name in method:
            return method[name]
    return None


def _same_number(left, right, label: str) -> None:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        raise CampaignError(f"campaign.method.{label} must be numeric")
    if float(left) != float(right):
        raise CampaignError(f"campaign.method.{label} is {left!r}, not {right!r}")


def validate_campaign_method(campaign, args) -> None:
    if not isinstance(campaign, dict):
        raise CampaignError("campaign.json must be an object")
    if campaign.get("source_clean") is not True:
        raise CampaignError("campaign.source_clean must be exactly true")
    commit = campaign.get("source_commit")
    if not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit):
        raise CampaignError("campaign.source_commit must be a full 40-hex commit")
    method = campaign.get("method")
    if not isinstance(method, dict):
        raise CampaignError("campaign.method must be an object")
    checks = [
        (("seconds", "window_seconds"), args.seconds, "seconds"),
        (("warmup", "warmup_seconds"), args.warmup, "warmup"),
        (("repeats",), args.repeats, "repeats"),
        (("long_input_concurrency",), args.long_input_concurrency,
         "long_input_concurrency"),
        (("max_batch_tokens", "batch_size"), args.max_batch_tokens, "max_batch_tokens"),
        (("max_sequence_length",), args.max_sequence_length, "max_sequence_length"),
    ]
    for names, expected, label in checks:
        value = _first(method, names)
        if value is not None:
            _same_number(value, expected, names[0])
    concurrency = method.get("concurrency")
    if isinstance(concurrency, dict):
        for key, value in concurrency.items():
            length = None if key == "*" else int(key)
            expected = args.long_input_concurrency if length is not None and length >= 2048 \
                else args.concurrency
            _same_number(value, expected, f"concurrency[{key!r}]")
    elif concurrency is not None:
        _same_number(concurrency, args.concurrency, "concurrency")
    client_concurrency = method.get("client_concurrency")
    if client_concurrency is not None:
        _same_number(client_concurrency, args.concurrency, "client_concurrency")
    runtimes = method.get("runtimes")
    if runtimes is not None and list(runtimes) != list(args.runtimes):
        raise CampaignError(f"campaign.method.runtimes is {runtimes!r}, not {args.runtimes!r}")
    dtype = method.get("dtype")
    if dtype is not None and str(dtype).lower() not in {"bfloat16", "bf16"}:
        raise CampaignError(f"campaign.method.dtype is {dtype!r}, not bfloat16")


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for root, directories, names in os.walk(path):
        directories[:] = sorted(name for name in directories if name != "__pycache__")
        for name in sorted(names):
            file = Path(root) / name
            if not file.is_file() or file.suffix in {".pyc", ".pyo"}:
                continue
            relative = file.relative_to(path).as_posix()
            digest.update(relative.encode() + b"\0")
            digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def runtime_tree_sha(campaign, runtime: str, cache: dict) -> str:
    if runtime in cache:
        return cache[runtime]
    if campaign is None:
        value = "campaign-unavailable"
    else:
        installations = campaign.get("runtime_installations")
        if not isinstance(installations, dict):
            value = "campaign-no-runtime-installations"
        elif runtime not in installations:
            raise CampaignError(f"campaign.runtime_installations lacks {runtime!r}")
        else:
            entry = installations[runtime]
            if isinstance(entry, str):
                value = entry
            elif isinstance(entry, dict):
                value = None
                for key in ("runtime_package_tree_sha256", "package_tree_sha256",
                            "tree_sha256", "sha256", "digest"):
                    if isinstance(entry.get(key), str):
                        value = entry[key]
                        break
                if value is None and entry.get("path"):
                    value = _tree_sha256(Path(entry["path"]))
                if value is None:
                    raise CampaignError(
                        f"campaign.runtime_installations[{runtime!r}] lacks a tree SHA")
            else:
                raise CampaignError(f"campaign.runtime_installations[{runtime!r}] is invalid")
    cache[runtime] = value
    return value


def parse_lanes(value: str, label: str) -> list[str]:
    lanes = [part.strip() for part in value.split(",") if part.strip()]
    if not lanes or len(set(lanes)) != len(lanes):
        raise CampaignError(f"{label} must contain distinct CPU lane ranges")
    for lane in lanes:
        if not re.fullmatch(r"\d+(?:-\d+)?", lane):
            raise CampaignError(f"invalid CPU lane {lane!r}")
    return lanes


def parse_gpus(value: str) -> list[int]:
    try:
        gpus = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise CampaignError("--gpus must be comma-separated integers") from exc
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise CampaignError("--gpus must contain distinct non-negative integers")
    return gpus


def lane_placement(index: int, gpu: int, gpus: list[int], server_lanes: list[str],
                   client_lanes: list[str]) -> tuple[str, str]:
    if len(server_lanes) != len(client_lanes):
        raise CampaignError("server and client CPU lane counts must match")
    if len(gpus) == len(server_lanes):
        position = index
    elif 0 <= gpu < len(server_lanes):
        position = gpu
    else:
        raise CampaignError(f"no CPU lane is defined for GPU {gpu}")
    return server_lanes[position], client_lanes[position]


def _port(args, runtime: str, gpu: int) -> int:
    base = {"infinity": args.infinity_port_base, "vllm": args.vllm_port_base,
            "sglang": args.sglang_port_base}[runtime]
    return base + gpu


def require_port_available(port: int, label: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise CampaignError(
                f"{label} port {port} is not available on 127.0.0.1: {exc}") from exc


def require_runtime_ports(args, runtime: str, gpu: int) -> None:
    port = _port(args, runtime, gpu)
    require_port_available(port, f"{runtime} server")
    if runtime == "infinity":
        require_port_available(port + args.prometheus_port_offset,
                               "infinity prometheus")


def _runtime_context(args, path: Path, runtime: str) -> tuple[int, int]:
    base = base_context(path, args.max_sequence_length)
    if runtime == "vllm":
        return vllm_context_limit(path, base), base
    if runtime == "sglang":
        return base + 1, base
    return base, base


def _settings(args, model: dict, runtime: str, path: Path | None) -> dict:
    context = None
    base = None
    if path is not None:
        context, base = _runtime_context(args, path, runtime)
    overrides = ({"INFINITY_TORCH_CUDA_GRAPHS": "1",
                  "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"} if runtime == "infinity"
                 else {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
                 if runtime == "sglang" else {})
    return {"dtype": args.dtype, "seconds": args.seconds, "repeats": args.repeats,
            "warmup": args.warmup, "concurrency": args.concurrency,
            "long_input_concurrency": args.long_input_concurrency,
            "max_batch_tokens": args.max_batch_tokens,
            "max_sequence_length": args.max_sequence_length,
            "lengths": list(model["lengths"]), "context": context,
            "input_context": base, "runtime_environment": overrides}


def _server_command(args, model: dict, path: Path, runtime: str, gpu: int) -> list[str]:
    context, _ = _runtime_context(args, path, runtime)
    port = _port(args, runtime, gpu)
    if runtime == "infinity":
        return [args.infinity_bin, "--model-id", str(path),
                "--served-model-name", "benchmark", "--torch-device", "cuda",
                "--dtype", args.dtype, "--tokenization-workers", "8",
                "--max-batch-tokens", str(args.max_batch_tokens),
                "--radix-mlp-threshold", "0", "--auto-truncate=false",
                "--max-sequence-length", str(context),
                "--hostname", "127.0.0.1", "--port", str(port),
                "--prometheus-port", str(port + args.prometheus_port_offset)]
    if runtime == "vllm":
        command = [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", str(path), "--served-model-name", "benchmark",
                   "--runner", "pooling", "--dtype", args.dtype,
                   "--trust-remote-code", "--max-model-len", str(context),
                   "--max-num-batched-tokens", str(args.max_batch_tokens),
                   "--max-num-seqs", "32", "--no-enable-prefix-caching",
                   "--mm-processor-cache-gb", "0", "--disable-log-requests",
                   "--host", "127.0.0.1", "--port", str(port)]
        if model["task"] == "embeddings":
            command.extend(["--pooler-config", json.dumps({"use_activation": True},
                                                          separators=(",", ":"))])
        return command
    command = [args.sglang_python, "-m", "sglang.launch_server",
               "--model-path", str(path), "--served-model-name", "benchmark",
               "--dtype", args.dtype, "--trust-remote-code",
               "--context-length", str(context),
               "--max-prefill-tokens", str(args.max_batch_tokens),
               "--max-running-requests", "32",
               "--max-total-tokens", str(args.max_batch_tokens),
               "--disable-radix-cache", "--mem-fraction-static", "0.75",
               "--host", "127.0.0.1", "--port", str(port), "--is-embedding"]
    if _head_dim(path) < 64:
        command.extend(["--attention-backend", "triton"])
    return command


def _environment(args, runtime: str, gpu: int) -> tuple[dict, dict]:
    env = os.environ.copy()
    overrides = {
        "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    if runtime == "infinity":
        overrides.update({"INFINITY_TORCH_CUDA_GRAPHS": "1",
                          "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"})
    elif runtime == "sglang":
        overrides["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    env.update(overrides)
    ninja_python = getattr(args, "sglang_python", None) or args.client_python
    python_bin = str(Path(ninja_python).absolute().parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    existing = env.get("LD_LIBRARY_PATH", "")
    injected = []
    if runtime == "infinity":
        injected = [str(Path(args.infinity_bin).resolve().parent)]
        if args.torch_lib:
            injected.append(str(Path(args.torch_lib).resolve()))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            injected + ([existing] if existing else []))
    recorded = dict(overrides)
    recorded["PATH_prepend"] = python_bin
    recorded["LD_LIBRARY_PATH_injected"] = injected
    recorded["LD_LIBRARY_PATH_inherited"] = existing
    return env, recorded


def _binary_path(args, runtime: str) -> str:
    return {"infinity": args.infinity_bin, "vllm": args.vllm_python,
            "sglang": args.sglang_python}[runtime]


def runtime_files(args, runtime: str, campaign, cache: dict) -> dict:
    libraries = {}
    binary = Path(_binary_path(args, runtime)).resolve()
    for library in sorted(Path(args.infinity_bin).resolve().parent.glob("*.so")):
        name = library.name
        digest = sha256_file(library)
        if name in libraries and libraries[name] != digest:
            raise CampaignError(f"duplicate native library basename {name!r}")
        libraries[name] = digest
    return {
        "runner_sha256": sha256_file(Path(__file__)),
        "client_sha256": sha256_file(HERE / "benchmark-http.py"),
        "binary_sha256": sha256_file(binary),
        "native_library_sha256": libraries,
        "runtime_package_tree_sha256": runtime_tree_sha(campaign, runtime, cache),
    }


def _checkpoint_identity(receipt: dict, metadata: str) -> dict:
    code_dependencies = []
    for item in receipt.get("code_dependencies") or []:
        if isinstance(item, dict) and item.get("id") and item.get("revision"):
            code_dependencies.append({"id": item["id"], "revision": item["revision"]})
    return {"id": receipt.get("id"), "revision": receipt.get("revision"),
            "metadata_sha256": metadata, "code_dependencies": code_dependencies}


def fingerprint_inputs(model: dict, checkpoint: dict, runtime: str, gpu: int,
                       cpus: str, client_cpus: str, settings: dict, files: dict) -> dict:
    return {
        "model": portable(model),
        "checkpoint": checkpoint,
        "runtime": runtime,
        "lane": {"gpu": gpu, "cpus": cpus, "client_cpus": client_cpus},
        "settings": settings,
        "files": files,
    }


def fingerprint(inputs: dict) -> str:
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_bytes(payload.encode())


def resume_decision(target: Path, current: str, retry_failures: bool) -> str:
    if not Path(target).is_file():
        return "run"
    try:
        old = _load_json(Path(target))
    except CampaignError:
        return "archive"
    if old.get("run_fingerprint") != current:
        return "archive"
    status = old.get("status")
    if status == "measured" and isinstance(old.get("measurement"), dict):
        return "skip"
    if status in TERMINAL_STATUSES and not retry_failures:
        return "skip"
    return "archive"


def _archive_previous(output: Path, kind: str, path: Path) -> None:
    if not path.exists():
        return
    destination = output / "previous" / kind
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{path.stem}--{time.time_ns()}{path.suffix}"
    path.replace(target)


def _clear_client_result(path: Path) -> None:
    path.unlink(missing_ok=True)


def _receipt_for(model: dict, path: Path) -> dict:
    if path.is_file():
        try:
            receipt = _load_json(path)
            if isinstance(receipt, dict):
                return receipt
        except CampaignError:
            pass
    return {"id": model["id"], "status": "unavailable",
            "revision": model["revision"], "error": "checkpoint preparation missing"}


def receipt_identity_error(model: dict, receipt: dict):
    if receipt.get("status") != "ready":
        return receipt.get("error") or "checkpoint is not ready"
    if receipt.get("id") != model["id"]:
        return f"receipt id {receipt.get('id')!r} does not match {model['id']!r}"
    if str(receipt.get("revision", "")).lower() != model["revision"]:
        return (f"receipt revision {receipt.get('revision')!r} does not match "
                f"{model['revision']!r}")
    return None


def stop(process: subprocess.Popen) -> None:
    group = process.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _tokenizer_path(receipt: dict, checkpoint_path: Path) -> str:
    dependencies = receipt.get("dependencies") or []
    if dependencies and isinstance(dependencies[0], dict) and dependencies[0].get("path"):
        return dependencies[0]["path"]
    return str(checkpoint_path)


def run_one(args, model: dict, runtime: str, gpu: int, cpus: str, client_cpus: str,
            campaign, tree_cache: dict) -> dict:
    stem = slug(model["id"]) + "--" + runtime
    target = args.output / "runs" / (stem + ".json")
    client_result = args.output / "clients" / (stem + ".json")
    log = args.output / "logs" / (stem + ".log")
    receipt = _receipt_for(model, args.output / "checkpoints" / (slug(model["id"]) + ".json"))
    path = Path(receipt["path"]) if receipt.get("path") else None
    metadata = metadata_sha256(path) if path and path.is_dir() else ""
    checkpoint = dict(receipt)
    checkpoint["metadata_sha256"] = metadata
    preflight_error = None
    try:
        settings = _settings(args, model, runtime,
                             path if path and path.is_dir() else None)
    except Exception as exc:
        preflight_error = f"{type(exc).__name__}: {exc}"
        settings = _settings(args, model, runtime, None)
    files = runtime_files(args, runtime, campaign, tree_cache)
    inputs = fingerprint_inputs(model, _checkpoint_identity(receipt, metadata), runtime,
                                gpu, cpus, client_cpus, settings, files)
    current = fingerprint(inputs)
    decision = resume_decision(target, current, args.retry_failures)
    if decision == "skip":
        return _load_json(target)
    if decision == "archive":
        for kind, owned in (("runs", target), ("clients", client_result), ("logs", log)):
            _archive_previous(args.output, kind, owned)
    _clear_client_result(client_result)
    item = {"model": model["id"], "task": model["task"], "runtime": runtime,
            "status": "load_failed", "gpu": gpu, "cpus": cpus,
            "client_cpus": client_cpus, "dtype": args.dtype,
            "checkpoint": checkpoint, "command": None, "client_command": None,
            "run_fingerprint": current, "fingerprint_inputs": inputs,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    identity_error = receipt_identity_error(model, receipt)
    if identity_error or not path or not path.is_dir():
        item.update(status="checkpoint_unavailable",
                    error=identity_error or "checkpoint path is unavailable")
        save_json(target, item)
        return item
    if preflight_error:
        item.update(status="load_failed", error=preflight_error,
                    total_elapsed_s=0.0)
        save_json(target, item)
        return item
    startup = time.perf_counter()
    process = None
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        command = ["taskset", "-c", cpus,
                   *_server_command(args, model, path, runtime, gpu)]
        environment, recorded_env = _environment(args, runtime, gpu)
        item["command"] = command
        item["environment_overrides"] = recorded_env
        port = _port(args, runtime, gpu)
        require_runtime_ports(args, runtime, gpu)
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command, env=environment, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True)
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {process.returncode}")
                if _health(port):
                    break
                if time.perf_counter() - startup > args.startup_timeout:
                    raise TimeoutError("Server readiness timeout")
                time.sleep(0.25)
            item["startup_s"] = time.perf_counter() - startup
            item["tokenizer"] = _tokenizer_path(receipt, path)
            client = ["taskset", "-c", client_cpus, args.client_python,
                      str(HERE / "benchmark-http.py"),
                      "--url", f"http://127.0.0.1:{port}", "--runtime", runtime,
                      "--model", "benchmark", "--tokenizer", item["tokenizer"],
                      "--task", model["task"],
                      "--lengths", ",".join(str(length) for length in model["lengths"]),
                      "--concurrency", str(args.concurrency),
                      "--seconds", str(args.seconds), "--repeats", str(args.repeats),
                      "--warmup", str(args.warmup),
                      "--output", str(client_result.resolve())]
            item["client_command"] = client
            measured = subprocess.run(client, env=environment, stdout=output,
                                      stderr=subprocess.STDOUT, timeout=args.client_timeout)
            if client_result.exists():
                item["measurement"] = _load_json(client_result)
            if measured.returncode:
                raise RuntimeError(f"Benchmark client exited with code {measured.returncode}")
            if not isinstance(item.get("measurement"), dict) \
                    or item["measurement"].get("ok") is not True:
                raise RuntimeError("Benchmark client did not report ok=true")
            item["status"] = "measured"
    except Exception as exc:
        item["status"] = "client_failed" if "startup_s" in item else "load_failed"
        item["error"] = f"{type(exc).__name__}: {exc}"
        item["log_tail"] = log.read_text(errors="replace")[-6000:] if log.exists() else ""
    finally:
        if process is not None:
            stop(process)
        item["total_elapsed_s"] = time.perf_counter() - startup
        save_json(target, item)
        print(json.dumps({"model": model["id"], "runtime": runtime,
                          "status": item["status"]}), flush=True)
    return item


def _placements(args, models: list[dict], gpus: list[int]) -> dict:
    if not args.model_lanes:
        return {model["id"]: gpus[index % len(gpus)] for index, model in enumerate(models)}
    placements = _load_json(Path(args.model_lanes))
    if not isinstance(placements, dict) or set(placements) != {m["id"] for m in models}:
        raise CampaignError("--model-lanes must map exactly the selected model IDs")
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu not in gpus
           for gpu in placements.values()):
        raise CampaignError("--model-lanes contains a GPU outside --gpus")
    return placements


def run(args, models: list[dict], campaign) -> None:
    gpus = parse_gpus(args.gpus)
    server_lanes = parse_lanes(args.server_cpus, "--server-cpus")
    client_lanes = parse_lanes(args.client_cpus, "--client-cpus")
    if len(server_lanes) != len(client_lanes):
        raise CampaignError("--server-cpus and --client-cpus must have equal counts")
    placements = _placements(args, models, gpus)
    tree_cache = {}
    if any(2048 in model["lengths"] for model in models) \
            and args.long_input_concurrency != 8:
        raise CampaignError("the client fixes 2048-token concurrency to 8")
    def lane(index: int, gpu: int) -> None:
        cpus, client_cpus = lane_placement(index, gpu, gpus, server_lanes, client_lanes)
        lane_models = [model for model in models if placements[model["id"]] == gpu]
        for model in lane_models:
            for runtime in args.runtimes:
                run_one(args, model, runtime, gpu, cpus, client_cpus, campaign, tree_cache)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, index, gpu) for index, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    rows = [_load_json(path) for path in sorted((args.output / "runs").glob("*.json"))]
    save_json(args.output / "results.json", rows)


def _expect(error, function, *args) -> None:
    try:
        function(*args)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def self_test() -> int:
    parsed = parse_args(["prepare"])
    assert parsed.infinity_bin is None and parsed.vllm_python is None
    assert parsed.sglang_python is None and parsed.client_python == sys.executable
    model = {"id": "org/model", "revision": "a" * 40, "task": "embeddings",
             "lengths": [128, 512, 2048], "parameters": 1,
             "code_dependencies": [{"id": "org/code", "revision": "b" * 40}]}
    validate_manifest({"models": [model]})
    for bad in ("main", "abc", "a" * 39, "a" * 41):
        _expect(CampaignError, validate_manifest,
                {"models": [{**model, "revision": bad}]})
    _expect(CampaignError, validate_manifest,
            {"models": [model, dict(model)]})
    _expect(CampaignError, validate_manifest,
            {"models": [{**model, "task": "sparse"}]})
    _expect(CampaignError, validate_manifest,
            {"models": [{**model, "lengths": [128, 128]}]})
    _expect(CampaignError, _verify_pinned, _FakeApi("c" * 40), "org/model", "a" * 40)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "config.json").write_text(json.dumps(
            {"hidden_size": 768, "num_attention_heads": 12,
             "max_position_embeddings": 512}))
        assert base_context(root, 2048) == 512
        assert _head_dim(root) == 64
        (root / "sentence_bert_config.json").write_text("{}")
        (root / "sentence_roberta_config.json").write_text(
            json.dumps({"max_seq_length": 300}))
        assert vllm_context_limit(root, 2048) == 300
        (root / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": 256}))
        assert vllm_context_limit(root, 2048) == 256
        args = argparse.Namespace(max_sequence_length=2048)
        assert _runtime_context(args, root, "sglang") == (513, 512)
        assert _runtime_context(args, root, "infinity") == (512, 512)
        assert _runtime_context(args, root, "vllm") == (256, 512)
        pooler_args = argparse.Namespace(
            vllm_python="vllm", dtype="bfloat16", max_batch_tokens=16384,
            max_sequence_length=2048, infinity_port_base=18200,
            vllm_port_base=18300, sglang_port_base=18400)
        embedding_command = _server_command(
            pooler_args, {**model, "task": "embeddings"}, root, "vllm", 0)
        rerank_command = _server_command(
            pooler_args, {**model, "task": "rerank"}, root, "vllm", 0)
        assert '{"use_activation":true}' in embedding_command
        assert not any("normalize" in part for part in embedding_command)
        assert "--pooler-config" not in rerank_command
        ready = {"id": model["id"], "status": "ready", "revision": model["revision"]}
        assert receipt_identity_error(model, ready) is None
        assert "id" in receipt_identity_error(model, {**ready, "id": "org/other"})
        assert "revision" in receipt_identity_error(
            model, {**ready, "revision": "c" * 40})
        inf_bin = root / "infinity" / "bin"
        inf_bin.mkdir(parents=True)
        torch_lib = root / "torch"
        torch_lib.mkdir()
        sglang_bin = root / "sglang" / "bin"
        sglang_bin.mkdir(parents=True)
        client_bin = root / "client" / "bin"
        client_bin.mkdir(parents=True)
        env_args = argparse.Namespace(
            infinity_bin=str(inf_bin / "infinity"), torch_lib=str(torch_lib),
            sglang_python=str(sglang_bin / "python"),
            client_python=str(client_bin / "python"))
        previous_ld = os.environ.get("LD_LIBRARY_PATH")
        os.environ["LD_LIBRARY_PATH"] = "sentinel"
        try:
            inf_env, inf_record = _environment(env_args, "infinity", 0)
            vllm_env, vllm_record = _environment(env_args, "vllm", 0)
            sg_env, _ = _environment(env_args, "sglang", 0)
        finally:
            if previous_ld is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = previous_ld
        assert vllm_env["LD_LIBRARY_PATH"] == "sentinel"
        assert sg_env["LD_LIBRARY_PATH"] == "sentinel"
        assert str(inf_bin.resolve()) in inf_env["LD_LIBRARY_PATH"]
        assert str(torch_lib.resolve()) in inf_env["LD_LIBRARY_PATH"]
        assert str(inf_bin.resolve()) not in vllm_env["LD_LIBRARY_PATH"]
        assert str(torch_lib.resolve()) not in vllm_env["LD_LIBRARY_PATH"]
        assert inf_record["LD_LIBRARY_PATH_injected"] == [
            str(inf_bin.resolve()), str(torch_lib.resolve())]
        assert vllm_record["LD_LIBRARY_PATH_injected"] == []
        assert vllm_env["PATH"].startswith(str(sglang_bin.resolve()) + os.pathsep)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            occupied = listener.getsockname()[1]
            _expect(CampaignError, require_port_available, occupied, "test server")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free_socket:
            free_socket.bind(("127.0.0.1", 0))
            free_port = free_socket.getsockname()[1]
        require_port_available(free_port, "test server")
        snapshot = root / "snapshot_meta"
        snapshot.mkdir()
        external = root / "external_pool"
        external.mkdir()
        (external / "config.json").write_text('{"pooling":1}')
        (snapshot / "config.json").write_text("{}")
        (snapshot / "1_Pooling").symlink_to(external, target_is_directory=True)
        first_meta = metadata_sha256(snapshot)
        (external / "config.json").write_text('{"pooling":2}')
        second_meta = metadata_sha256(snapshot)
        assert first_meta != second_meta
        clone = root / "snapshot_clone"
        shutil.copytree(snapshot, clone, symlinks=True)
        assert metadata_sha256(clone) == second_meta
        external_map = root / "external_auto_map"
        external_map.mkdir()
        (external_map / "config.json").write_text(json.dumps(
            {"auto_map": {"AutoModel": "org/code--modeling.Foo"}}))
        _expect(CampaignError, validate_auto_map, external_map,
                [{"id": "org/code", "revision": "b" * 40, "path": str(external)}])
        server = list(DEFAULT_SERVER_CPUS)
        client = list(DEFAULT_CLIENT_CPUS)
        assert lane_placement(0, 0, list(range(6)), server, client) == ("0-7", "32-39")
        assert lane_placement(4, 4, list(range(6)), server, client) == ("64-71", "80-87")
        assert lane_placement(0, 2, [2], server, client) == ("16-23", "48-55")
        campaign = {"runtime_installations": {"infinity": "9" * 64,
                                              "vllm": "8" * 64,
                                              "sglang": "7" * 64}}
        cache = {}
        fingerprint_values = []
        for name in ("left", "right"):
            base = root / name
            (base / "runtime").mkdir(parents=True)
            (base / "bin").write_bytes(b"binary")
            (base / "runtime" / "libx.so").write_bytes(b"library")
            fake = argparse.Namespace(infinity_bin=str(base / "bin"),
                                      vllm_python=str(base / "bin"),
                                      sglang_python=str(base / "bin"))
            files = runtime_files(fake, "infinity", campaign, cache)
            inputs = fingerprint_inputs(model, _checkpoint_identity(
                {"id": model["id"], "revision": model["revision"]}, "f" * 64),
                "infinity", 0, "0-7", "32-39",
                {"seconds": 10, "repeats": 3}, files)
            assert "/tmp/" not in json.dumps(inputs)
            fingerprint_values.append(fingerprint(inputs))
        assert fingerprint_values[0] == fingerprint_values[1]
        stale = root / "client.json"
        stale.write_text("{}")
        _clear_client_result(stale)
        assert not stale.exists()
        raw = root / "raw.json"
        save_json(raw, {"run_fingerprint": "x", "status": "measured",
                        "measurement": {"ok": True}})
        assert resume_decision(raw, "x", False) == "skip"
        assert resume_decision(raw, "y", False) == "archive"
        save_json(raw, {"run_fingerprint": "x", "status": "load_failed"})
        assert resume_decision(raw, "x", False) == "skip"
        assert resume_decision(raw, "x", True) == "archive"
        pidfile = root / "child.pid"
        leader = subprocess.Popen(
            ["sh", "-c", 'sleep 300 & echo $! > "$1"; exit 0', "sh", str(pidfile)],
            start_new_session=True)
        leader.wait(timeout=10)
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            time.sleep(0.02)
        child = int(pidfile.read_text().strip())
        stop(leader)
        for _ in range(200):
            if not _process_running(child):
                break
            time.sleep(0.01)
        assert not _process_running(child)
    print("self-test passed")
    return 0


class _FakeApi:
    def __init__(self, resolved: str):
        self.resolved = resolved

    def model_info(self, _repository, revision=None):
        return type("Info", (), {"sha": self.resolved})()


def _process_running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return False
    return state != "Z"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("prepare", "run"))
    parser.add_argument("--self-test", action="store_true",
                        help="run mocked offline contract tests and exit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--only", help="comma-separated exact checkpoint IDs")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--server-cpus", default=",".join(DEFAULT_SERVER_CPUS))
    parser.add_argument("--client-cpus", default=",".join(DEFAULT_CLIENT_CPUS))
    parser.add_argument("--model-lanes", type=Path,
                        help="optional exact model-ID to GPU JSON map")
    parser.add_argument("--runtimes", default=",".join(RUNTIMES))
    parser.add_argument("--infinity-bin",
                        help="Infinity executable; required for run and fingerprinted")
    parser.add_argument("--vllm-python", help="vLLM executable/wrapper; required for vLLM runs")
    parser.add_argument("--sglang-python",
                        help="SGLang Python executable; required for SGLang runs")
    parser.add_argument("--client-python", default=sys.executable)
    parser.add_argument("--torch-lib",
                        help="Infinity-only torch library directory; ignored for vLLM/SGLang")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--long-input-concurrency", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--client-timeout", type=float, default=900.0)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--infinity-port-base", type=int, default=18200)
    parser.add_argument("--vllm-port-base", type=int, default=18300)
    parser.add_argument("--sglang-port-base", type=int, default=18400)
    parser.add_argument("--prometheus-port-offset", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.mode:
        raise SystemExit("mode is required (prepare or run); use --self-test for tests")
    if args.output is None or not args.output.is_absolute():
        raise SystemExit("--output must be an absolute path")
    args.output = args.output.resolve()
    if args.dtype.lower() not in {"bfloat16", "bf16"}:
        raise SystemExit("--dtype must be bfloat16")
    if args.seconds <= 0 or args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--seconds must be > 0; --repeats >= 1; --warmup >= 0")
    if args.concurrency < 1 or args.long_input_concurrency < 1:
        raise SystemExit("--concurrency and --long-input-concurrency must be >= 1")
    if args.max_batch_tokens < 1 or args.max_sequence_length < 1:
        raise SystemExit("--max-batch-tokens and --max-sequence-length must be >= 1")
    args.runtimes = [part.strip() for part in args.runtimes.split(",") if part.strip()]
    if not args.runtimes or any(runtime not in RUNTIMES for runtime in args.runtimes):
        raise SystemExit(f"--runtimes must be a subset of {RUNTIMES}")
    if len(set(args.runtimes)) != len(args.runtimes):
        raise SystemExit("--runtimes must be unique")
    if args.torch_lib and not Path(args.torch_lib).is_dir():
        raise SystemExit(f"--torch-lib is not a directory: {args.torch_lib}")
    for directory in ("checkpoints", "runs", "clients", "logs"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    try:
        _, models = load_models(args)
    except CampaignError as exc:
        raise SystemExit(str(exc))
    if args.mode == "prepare":
        prepare(args, models)
        return 0
    required = [("--infinity-bin", args.infinity_bin)]
    if "vllm" in args.runtimes:
        required.append(("--vllm-python", args.vllm_python))
    if "sglang" in args.runtimes:
        required.append(("--sglang-python", args.sglang_python))
    required.append(("--client-python", args.client_python))
    for label, path in required:
        if not path or not Path(path).exists():
            raise SystemExit(f"{label} is required and must exist: {path}")
    campaign_path = args.output / "campaign.json"
    if not campaign_path.is_file():
        raise SystemExit("campaign.json is required in run mode")
    campaign = _load_json(campaign_path)
    try:
        validate_campaign_method(campaign, args)
        run(args, models, campaign)
    except CampaignError as exc:
        raise SystemExit(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
