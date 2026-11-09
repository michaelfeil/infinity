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
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shlex
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
SCRUBBED_LOADER_VARIABLES = (
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME")
PYTHON_SAFE_PATH = "1"
INTEGRITY_HELPER = HERE / "benchmark-integrity.py"
CAMPAIGN_INSPECTOR = HERE / "benchmark-campaign.py"
WEIGHT_FAMILIES = (
    ("safetensors", (".safetensors",), "*.safetensors.index.json"),
    ("bin", (".bin",), "*.bin.index.json"),
    ("torch", (".pt", ".pth"), None),
)


class CampaignError(ValueError):
    """A manifest, campaign or lane placement is invalid."""


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256_bytes(text.encode())


def sanitized_environment(base=None) -> dict:
    """Return an environment with ambient loader/Python overrides removed."""
    env = dict(os.environ if base is None else base)
    for name in SCRUBBED_LOADER_VARIABLES:
        env.pop(name, None)
    env["PYTHONSAFEPATH"] = PYTHON_SAFE_PATH
    return env


def runtime_loader_environment(runtime: str, infinity_bin=None, torch_lib=None,
                               base=None) -> tuple[dict, list[str]]:
    env = sanitized_environment(base)
    injected = []
    if runtime == "infinity":
        if not infinity_bin:
            raise CampaignError("Infinity binary is required for its loader environment")
        injected.append(str(Path(infinity_bin).resolve().parent))
        if torch_lib:
            injected.append(str(Path(torch_lib).resolve()))
        env["LD_LIBRARY_PATH"] = os.pathsep.join(injected)
    return env, injected


def loader_policy(runtime: str) -> dict:
    return {"version": 1,
            "ld_library_path": ("infinity_selected_only" if runtime == "infinity"
                                else "empty"),
            "ld_preload": "unset", "ld_audit": "unset",
            "pythonpath": "unset", "pythonhome": "unset",
            "python_safe_path": True}


def library_tree_sha256(path: Path) -> str:
    """Hash a library directory by portable relative filename and file content."""
    path = Path(path)
    if not path.is_dir():
        raise CampaignError(f"library directory is not a directory: {path}")
    files = {}

    def visit(directory: Path, prefix: str, ancestors: frozenset) -> None:
        try:
            resolved = directory.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CampaignError(f"cannot resolve library directory {directory}: {exc}") from exc
        if resolved in ancestors:
            return
        ancestors = ancestors | {resolved}
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise CampaignError(f"cannot read library directory {directory}: {exc}") from exc
        for item in entries:
            relative = prefix + item.name
            try:
                if item.is_dir():
                    visit(item, relative + "/", ancestors)
                elif item.is_file():
                    files[relative] = sha256_file(item)
                elif item.is_symlink():
                    raise CampaignError(f"library symlink {item} does not resolve to a file")
            except OSError as exc:
                raise CampaignError(f"cannot read library entry {item}: {exc}") from exc

    visit(path, "", frozenset())
    if not files:
        raise CampaignError(f"library directory is empty: {path}")
    return canonical_sha256(files)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path = path.resolve()
        parent = parent.resolve()
    except (OSError, RuntimeError):
        return False
    return path == parent or parent in path.parents


def _elf_runpath_directories(torch_lib: Path) -> list[Path]:
    result = []
    for candidate in sorted(torch_lib.rglob("*.so*")):
        if not candidate.is_file():
            continue
        try:
            completed = subprocess.run(
                ["readelf", "-d", str(candidate)], capture_output=True, text=True,
                check=False, timeout=30, env=sanitized_environment())
        except OSError:
            continue
        if completed.returncode:
            continue
        for line in completed.stdout.splitlines():
            match = re.search(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", line)
            if match is None:
                continue
            for entry in match.group(1).split(":"):
                entry = entry.strip()
                if not entry or ("$" in entry and "$ORIGIN" not in entry
                                 and "${ORIGIN}" not in entry):
                    continue
                expanded = entry.replace("${ORIGIN}", str(candidate.parent))
                expanded = expanded.replace("$ORIGIN", str(candidate.parent))
                if "$" in expanded or not os.path.isabs(expanded):
                    continue
                directory = Path(expanded)
                if directory.is_dir():
                    result.append(directory)
    return result


def torch_sdk_bundle(torch_lib: Path) -> dict:
    """Return the canonical selected Torch SDK bundle digest and portable roots."""
    try:
        root = Path(torch_lib).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CampaignError(
            f"Torch library directory cannot be resolved: {torch_lib}: {exc}") from exc
    if not root.is_dir():
        raise CampaignError(f"Torch library directory is not a directory: {root}")
    site = (root.parent.parent if root.name == "lib" and root.parent.name == "torch"
            else None)
    nvidia = site / "nvidia" if site is not None else None
    roots = {"torch_lib": library_tree_sha256(root)}
    if nvidia is not None and nvidia.is_dir():
        roots["nvidia"] = library_tree_sha256(nvidia)
    sdk_base = site if site is not None else root.parent
    known = [root.resolve()]
    if nvidia is not None and nvidia.is_dir():
        known.append(nvidia.resolve())
    rpath_roles = {}
    for directory in _elf_runpath_directories(root):
        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            continue
        if not _is_within(resolved, sdk_base) or any(
                resolved == item or item in resolved.parents
                or resolved in item.parents for item in known):
            continue
        relative = os.path.relpath(resolved, sdk_base.resolve()).replace(os.sep, "/")
        rpath_roles[f"rpath/{relative}"] = resolved
    for role, directory in sorted(rpath_roles.items()):
        roots[role] = library_tree_sha256(directory)
    payload = {"schema_version": 1, "roots": roots}
    return {"schema_version": 1, "roots": roots,
            "digest_sha256": canonical_sha256(payload)}


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


def _safe_weight_relative(name, label: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise CampaignError(f"{label}: weight reference is not a non-empty string")
    if os.path.isabs(name) or name.startswith(("/", "\\")):
        raise CampaignError(f"{label}: absolute weight reference {name!r} is not allowed")
    parts = [part for part in re.split(r"[\\/]+", name) if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise CampaignError(f"{label}: escaping weight reference {name!r} is not allowed")
    return Path(*parts)


def _walk_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise CampaignError(f"directory is unavailable: {root}")
    result = []

    def visit(directory: Path, ancestors: frozenset) -> None:
        try:
            resolved = directory.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CampaignError(f"cannot resolve directory {directory}: {exc}") from exc
        if resolved in ancestors:
            return
        ancestors = ancestors | {resolved}
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise CampaignError(f"cannot read directory {directory}: {exc}") from exc
        for item in entries:
            try:
                if item.is_dir():
                    visit(item, ancestors)
                elif item.is_file():
                    result.append(item)
            except OSError as exc:
                raise CampaignError(f"cannot inspect {item}: {exc}") from exc

    visit(root, frozenset())
    return result


def _weight_suffixes() -> frozenset:
    return frozenset(suffix for _, suffixes, _ in WEIGHT_FAMILIES for suffix in suffixes)


def _index_family(path: Path):
    for family, suffixes, pattern in WEIGHT_FAMILIES:
        if pattern is not None and path.name.endswith(pattern[1:]):
            return family, suffixes
    return None


def _weight_and_index_files(root: Path) -> tuple[list[Path], list[Path]]:
    suffixes = _weight_suffixes()
    weights, indexes = [], []
    for path in _walk_files(root):
        if path.name.lower() == "tokenizer.bin":
            continue
        if path.suffix.lower() in suffixes:
            weights.append(path)
        if _index_family(path) is not None:
            indexes.append(path)
    return weights, indexes


def _validated_index_weights(index: Path, root: Path, label: str,
                             suffixes) -> dict:
    document = _load_json(index)
    weight_map = document.get("weight_map") if isinstance(document, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise CampaignError(f"{label}: {index.name} has no valid weight_map")
    selected = {}
    for tensor, name in weight_map.items():
        windows = PureWindowsPath(name) if isinstance(name, str) else None
        if windows is not None and (windows.is_absolute() or windows.drive):
            raise CampaignError(
                f"{label}: absolute weight reference {name!r} is not allowed")
        relative = _safe_weight_relative(name, f"{label}: {index.name} {tensor}")
        if relative.suffix.lower() not in suffixes:
            raise CampaignError(
                f"{label}: weight reference {name!r} has no active weight suffix")
        candidate = index.parent / relative
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise CampaignError(
                f"{label}: weight reference {name!r} escapes checkpoint") from exc
        if not candidate.is_file():
            raise CampaignError(
                f"{label}: active weight {name!r} referenced by {index.name} is missing")
        relative_name = candidate.relative_to(root).as_posix()
        selected[relative_name] = candidate
    return selected


def _weight_files_for_root(root: Path, label: str) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise CampaignError(f"{label}: checkpoint directory is unavailable")
    weights, indexes = _weight_and_index_files(root)
    selected = {path.relative_to(root).as_posix(): path for path in weights}
    indexes_by_directory = {}
    for index in indexes:
        family = _index_family(index)
        if family is not None:
            indexes_by_directory.setdefault(index.parent, []).append(index)
    for directory, directory_indexes in sorted(indexes_by_directory.items()):
        direct_weights = [path for path in weights if path.parent == directory]
        preferred = None
        for family, suffixes, _ in WEIGHT_FAMILIES:
            has_file = any(path.suffix.lower() in suffixes for path in direct_weights)
            has_index = any(_index_family(index)[0] == family
                            for index in directory_indexes)
            if has_file or has_index:
                preferred = (family, suffixes)
                break
        if preferred is None:
            continue
        family, suffixes = preferred
        for index in directory_indexes:
            index_family = _index_family(index)
            if index_family is not None and index_family[0] == family:
                selected.update(_validated_index_weights(
                    index, root, label, suffixes))
    if not selected:
        raise CampaignError(f"{label}: no usable .safetensors/.bin/.pt/.pth weights")
    return {name: selected[name] for name in sorted(selected)}


def _has_weight_candidates(root: Path) -> bool:
    weights, indexes = _weight_and_index_files(Path(root))
    return bool(weights or indexes)


def weight_bytes(path: Path) -> int:
    return sum(path.stat().st_size
               for path in _weight_files_for_root(Path(path), "checkpoint").values())


def _dependency_paths(items) -> tuple:
    result = []
    for item in items or []:
        if isinstance(item, dict) and item.get("id") and item.get("revision") \
                and item.get("path"):
            result.append((str(item["id"]), str(item["revision"]), str(item["path"])))
    return tuple(result)


def checkpoint_weights(path: Path, dependencies=(), code_dependencies=(),
                       cache: dict | None = None) -> dict:
    cache = cache if isinstance(cache, dict) else {}
    root = Path(path)
    dependency_paths = _dependency_paths(dependencies)
    code_paths = _dependency_paths(code_dependencies)
    key = ("checkpoint_weights", _probe_path(root), dependency_paths, code_paths)
    if key in cache:
        return cache[key]
    root_files = _weight_files_for_root(root, "checkpoint")
    selected = dict(root_files)
    root_bytes = sum(item.stat().st_size for item in root_files.values())
    seen = {}
    for namespace, items in (("dependencies", dependency_paths),
                             ("code_dependencies", code_paths)):
        for identity, revision, raw_path in items:
            dependency_root = Path(raw_path)
            if not dependency_root.is_dir():
                raise CampaignError(
                    f"dependency {identity}@{revision}: path is unavailable")
            real_root = os.path.realpath(dependency_root)
            previous = seen.get((identity, revision))
            if previous is not None and previous != real_root:
                raise CampaignError(
                    f"dependency {identity}@{revision} has multiple roots")
            seen[(identity, revision)] = real_root
            if not _has_weight_candidates(dependency_root):
                continue
            files = _weight_files_for_root(
                dependency_root, f"dependency {identity}@{revision}")
            prefix = f"{namespace}/{slug(identity)}@{revision}"
            for relative, file in files.items():
                selected[f"{prefix}/{relative}"] = file
    weight_map = {name: sha256_file(file)
                  for name, file in sorted(selected.items())}
    result = {"weight_files_sha256": weight_map,
              "weights_sha256": canonical_sha256(weight_map),
              "weight_bytes": root_bytes}
    cache[key] = result
    return result


def _metadata_sha256_cached(path, cache: dict) -> str:
    key = ("metadata_sha256", _probe_path(path) if path else "")
    if key not in cache:
        cache[key] = metadata_sha256(Path(path)) if path else ""
    return cache[key]


def _refresh_dependency_descriptors(items, cache: dict) -> list[dict]:
    result = []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("id") or not item.get("revision"):
            continue
        descriptor = dict(item)
        path = descriptor.get("path")
        descriptor["metadata_sha256"] = (
            _metadata_sha256_cached(path, cache) if path
            else descriptor.get("metadata_sha256") or "")
        result.append(descriptor)
    return result


def _dependency_identity(items) -> list[dict]:
    result = []
    for item in items or []:
        if isinstance(item, dict) and item.get("id") and item.get("revision"):
            result.append({"id": item["id"], "revision": item["revision"],
                           "metadata_sha256": item.get("metadata_sha256") or ""})
    return result


def _strict_dependency_descriptors(items, label: str, cache: dict) -> list[dict]:
    if items is None:
        return []
    if not isinstance(items, list):
        raise CampaignError(f"{label} must be a list")
    result = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            raise CampaignError(f"{label}[{position}] must be an object")
        if not isinstance(item.get("id"), str) or not item["id"] \
                or not isinstance(item.get("revision"), str) or not item["revision"]:
            raise CampaignError(f"{label}[{position}] lacks id/revision")
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise CampaignError(f"{label}[{position}] lacks a dependency path")
        if not Path(path).is_dir():
            raise CampaignError(f"{label}[{position}] path is unavailable")
        metadata = _metadata_sha256_cached(path, cache)
        if not re.fullmatch(r"[0-9a-f]{64}", metadata):
            raise CampaignError(f"{label}[{position}] metadata is unavailable")
        descriptor = dict(item)
        descriptor["metadata_sha256"] = metadata
        result.append(descriptor)
    return result


def checkpoint_identity(receipt: dict, cache: dict | None = None) -> dict:
    """Return a fresh raw checkpoint receipt and its portable identity."""
    if not isinstance(receipt, dict):
        raise CampaignError("checkpoint receipt must be an object")
    path_value = receipt.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise CampaignError("checkpoint receipt lacks a path")
    path = Path(path_value)
    if not path.is_dir():
        raise CampaignError(f"checkpoint path is unavailable: {path}")
    cache = cache if isinstance(cache, dict) else {}
    dependencies = _strict_dependency_descriptors(
        receipt.get("dependencies"), "dependencies", cache)
    code_dependencies = _strict_dependency_descriptors(
        receipt.get("code_dependencies"), "code_dependencies", cache)
    key = ("checkpoint_identity", _probe_path(path),
           receipt.get("id"), receipt.get("revision"),
           _dependency_paths(dependencies), _dependency_paths(code_dependencies))
    if key in cache:
        return cache[key]
    metadata = _metadata_sha256_cached(path, cache)
    weights = checkpoint_weights(path, dependencies, code_dependencies, cache)
    checkpoint = dict(receipt)
    checkpoint.update(path=str(path), metadata_sha256=metadata,
                      weight_bytes=weights["weight_bytes"],
                      weight_files_sha256=weights["weight_files_sha256"],
                      weights_sha256=weights["weights_sha256"],
                      dependencies=dependencies,
                      code_dependencies=code_dependencies)
    result = {"checkpoint": checkpoint,
              "identity": _checkpoint_identity(
                  receipt, metadata, weights, dependencies, code_dependencies)}
    cache[key] = result
    return result


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
    has_bin = any(name.endswith(".bin") and Path(name).name.lower() != "tokenizer.bin"
                  for name in names)
    has_torch = any(name.endswith((".pt", ".pth")) for name in names)
    if not has_safetensors and not has_bin and not has_torch:
        raise CampaignError(
            f"{model['id']}: snapshot has no .safetensors/.bin/.pt/.pth weights")
    patterns = list(CODE_PATTERNS)
    if has_safetensors:
        patterns.append("*.safetensors")
    elif has_bin:
        patterns.append("*.bin")
    else:
        patterns.extend(["*.pt", "*.pth"])
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


def _refresh_receipt(receipt: dict, model: dict, args, cache=None):
    path, adapter = ensure_tokenizer(Path(receipt["path"]), model, args.output)
    cache = _invocation_cache(args, cache)
    prepared = dict(receipt)
    prepared["path"] = str(path)
    refreshed = checkpoint_identity(prepared, cache)["checkpoint"]
    refreshed["tokenizer_adapter"] = adapter
    return refreshed


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
    args._benchmark_cache = {}
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


def runtime_tree_sha(campaign, runtime: str, cache: dict) -> str:
    files = cache.get(("runtime_files", runtime))
    if not isinstance(files, dict):
        raise CampaignError(
            f"actual {runtime!r} runtime files have not been probed in this invocation")
    return files["runtime_package_tree_sha256"]


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


_IDENTITY_PROGRAM = """
import hashlib, json, os, sys
executable = os.path.abspath(sys.executable)
prefix = os.path.abspath(sys.prefix)
with open(sys.executable, "rb") as handle:
    interpreter = hashlib.file_digest(handle, "sha256").hexdigest()
identity = {"executable": executable, "prefix": prefix,
            "interpreter_sha256": interpreter}
identity["sha256"] = hashlib.sha256(
    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(identity, sort_keys=True))
"""


def _runtime_python(args, runtime: str):
    return {"infinity": args.client_python, "vllm": args.vllm_python,
            "sglang": args.sglang_python}[runtime]


def _invocation_cache(args, cache=None) -> dict:
    if cache is not None:
        return cache
    existing = getattr(args, "_benchmark_cache", None)
    if not isinstance(existing, dict):
        existing = {}
        setattr(args, "_benchmark_cache", existing)
    return existing


def _probe_path(path) -> str:
    return os.path.abspath(str(path))


def _identity_key(python) -> tuple:
    return ("python_identity", _probe_path(python))


def _package_key(python, module: str) -> tuple:
    return ("package_probe", _probe_path(python), module)


def _probe_environment(args, runtime: str) -> dict:
    env, _ = runtime_loader_environment(
        runtime, getattr(args, "infinity_bin", None),
        getattr(args, "torch_lib", None))
    return env


def _run_probe(command, label: str, env=None) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                check=False, timeout=600, env=env)
    except OSError as exc:
        raise CampaignError(f"{label}: {exc}") from exc
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise CampaignError(f"{label} returned invalid JSON: {detail or exc}") from exc
    if result.returncode or not isinstance(payload, dict):
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise CampaignError(f"{label} failed: {detail or result.stderr.strip()[:500]}")
    if "error" in payload:
        raise CampaignError(f"{label} failed: {payload['error']}")
    return payload


def _normalise_identity(identity) -> dict:
    if not isinstance(identity, dict):
        raise CampaignError("runtime Python identity probe returned no identity")
    base = {key: identity.get(key)
            for key in ("executable", "prefix", "interpreter_sha256")}
    if not all(isinstance(value, str) and value for value in base.values()):
        raise CampaignError("runtime Python identity probe returned invalid paths/hash")
    if not os.path.isabs(base["executable"]) or not os.path.isabs(base["prefix"]):
        raise CampaignError("runtime Python identity probe returned relative paths")
    if not re.fullmatch(r"[0-9a-f]{64}", base["interpreter_sha256"]):
        raise CampaignError("runtime Python identity probe returned invalid interpreter SHA")
    base["sha256"] = canonical_sha256(base)
    if identity.get("sha256") not in (None, base["sha256"]):
        raise CampaignError("runtime Python identity probe returned inconsistent SHA")
    return base


def runtime_python_identity_info(args, runtime: str, cache=None) -> dict:
    selected = _runtime_python(args, runtime)
    cache = _invocation_cache(args, cache)
    key = _identity_key(selected)
    if key not in cache:
        payload = _run_probe([str(selected), "-c", _IDENTITY_PROGRAM],
                             f"runtime Python identity probe ({selected})",
                             env=_probe_environment(args, runtime))
        cache[key] = _normalise_identity(payload)
    return cache[key]


def runtime_python_identity_sha256(args, runtime: str, cache=None) -> str:
    return runtime_python_identity_info(args, runtime, cache)["sha256"]


def runtime_package_probe(args, runtime: str, cache=None) -> dict:
    if runtime not in {"vllm", "sglang"}:
        raise CampaignError(f"no package probe is defined for {runtime!r}")
    selected = _runtime_python(args, runtime)
    cache = _invocation_cache(args, cache)
    key = _package_key(selected, runtime)
    if key not in cache:
        payload = _run_probe(
            [str(selected), str(CAMPAIGN_INSPECTOR),
             "--inspect-package", runtime],
            f"runtime package probe ({selected})",
            env=_probe_environment(args, runtime))
        cache[key] = payload
        if payload.get("python_identity") is not None:
            cache[_identity_key(selected)] = _normalise_identity(
                payload["python_identity"])
    return cache[key]


def seed_python_identity(cache: dict, python, identity) -> None:
    cache[_identity_key(python)] = _normalise_identity(identity)


def seed_package_probe(cache: dict, python, module: str, payload: dict) -> None:
    cache[_package_key(python, module)] = payload
    if payload.get("python_identity") is not None:
        cache[_identity_key(python)] = _normalise_identity(payload["python_identity"])


def python_environment_probe(args, runtime: str, cache=None) -> dict:
    selected = _runtime_python(args, runtime)
    cache = _invocation_cache(args, cache)
    key = ("python_environment", _probe_path(selected))
    if key not in cache:
        helper = Path(INTEGRITY_HELPER)
        if not helper.is_file():
            raise CampaignError(
                f"Python environment helper is unavailable: {helper}")
        payload = _run_probe(
            [str(selected), str(helper)],
            f"python environment probe ({selected})",
            env=_probe_environment(args, runtime))
        digest = payload.get("digest_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CampaignError(
                f"python environment probe ({selected}) returned no valid digest")
        cache[key] = payload
    return cache[key]


def python_environment_sha256(args, runtime: str, cache=None) -> str:
    return python_environment_probe(args, runtime, cache)["digest_sha256"]


def _source_file_sha256(path, cache: dict, role: str) -> str:
    key = ("source_file_sha256", role, _probe_path(path))
    if key not in cache:
        cache[key] = sha256_file(Path(path))
    return cache[key]


def _library_tree_sha256(path, cache: dict) -> str:
    key = ("library_tree", _probe_path(path))
    if key not in cache:
        cache[key] = library_tree_sha256(Path(path))
    return cache[key]


def _torch_sdk_bundle(path, cache: dict) -> dict:
    key = ("torch_sdk_bundle", _probe_path(path))
    if key not in cache:
        cache[key] = torch_sdk_bundle(Path(path))
    return cache[key]


def _settings(args, model: dict, runtime: str, path: Path | None, cache=None) -> dict:
    context = None
    base = None
    if path is not None:
        context, base = _runtime_context(args, path, runtime)
    overrides = ({"INFINITY_TORCH_CUDA_GRAPHS": "1",
                  "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"} if runtime == "infinity"
                 else {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
                 if runtime == "sglang" else {})
    overrides["python_identity_sha256"] = runtime_python_identity_sha256(
        args, runtime, cache)
    overrides["loader_policy"] = loader_policy(runtime)
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


def _environment(args, runtime: str, gpu: int, cache=None) -> tuple[dict, dict]:
    env, injected = runtime_loader_environment(
        runtime, getattr(args, "infinity_bin", None), getattr(args, "torch_lib", None))
    overrides = {
        "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONSAFEPATH": PYTHON_SAFE_PATH,
    }
    if runtime == "infinity":
        overrides.update({"INFINITY_TORCH_CUDA_GRAPHS": "1",
                          "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"})
    elif runtime == "sglang":
        overrides["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    identity = runtime_python_identity_info(args, runtime, cache)
    python_bin = str(Path(identity["executable"]).absolute().parent)
    env.update(overrides)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
    recorded = dict(overrides)
    recorded["loader_policy"] = loader_policy(runtime)
    recorded["python_identity_sha256"] = identity["sha256"]
    recorded["python_executable"] = identity["executable"]
    recorded["python_prefix"] = identity["prefix"]
    recorded["PATH_prepend"] = python_bin
    recorded["LD_LIBRARY_PATH_injected"] = injected
    recorded["LD_LIBRARY_PATH_inherited"] = ""
    recorded["sanitized_variables"] = list(SCRUBBED_LOADER_VARIABLES)
    return env, recorded


def _binary_path(args, runtime: str) -> str:
    return {"infinity": args.infinity_bin, "vllm": args.vllm_python,
            "sglang": args.sglang_python}[runtime]


def _native_libraries(args, cache: dict) -> dict:
    key = ("native_libraries",)
    if key not in cache:
        libraries = {}
        for library in sorted(Path(args.infinity_bin).resolve().parent.glob("*.so")):
            name = library.name
            digest = sha256_file(library)
            if name in libraries and libraries[name] != digest:
                raise CampaignError(f"duplicate native library basename {name!r}")
            libraries[name] = digest
        cache[key] = libraries
    return cache[key]


def _campaign_installation(campaign, runtime: str):
    if campaign is None:
        return None
    installations = campaign.get("runtime_installations")
    if not isinstance(installations, dict) or runtime not in installations:
        raise CampaignError(f"campaign.runtime_installations lacks {runtime!r}")
    entry = installations[runtime]
    if isinstance(entry, str):
        return {"package_tree_sha256": entry}
    if not isinstance(entry, dict):
        raise CampaignError(f"campaign.runtime_installations[{runtime!r}] is invalid")
    return entry


def _verify_runtime_installation(campaign, runtime: str, package_tree: str, probe):
    entry = _campaign_installation(campaign, runtime)
    if entry is None:
        return
    expected_tree = entry.get("package_tree_sha256")
    if expected_tree is not None and expected_tree != package_tree:
        raise CampaignError(
            f"current {runtime!r} runtime package tree differs from campaign")
    if runtime not in {"vllm", "sglang"}:
        return
    if not isinstance(probe, dict):
        raise CampaignError(f"no actual package probe is available for {runtime!r}")
    for field in ("version", "torch_version", "transformers_version"):
        if entry.get(field) is not None and entry[field] != probe.get(field):
            raise CampaignError(f"current {runtime!r} {field} differs from campaign")
    expected_record = entry.get("record_sha256")
    if expected_record is not None and expected_record != probe.get("record_sha256"):
        raise CampaignError(f"current {runtime!r} wheel RECORD differs from campaign")
    for field in ("record_mismatches", "unrecorded_files", "missing_record_files"):
        expected = sorted(entry.get(field) or [])
        actual = sorted(probe.get(field) or [])
        if expected != actual:
            raise CampaignError(
                f"current {runtime!r} {field} differ from campaign: "
                f"{actual!r} != {expected!r}")


def _verify_infinity_torch(campaign, torch_tree, roots=None):
    if campaign is None:
        return
    build = campaign.get("build")
    if not isinstance(build, dict):
        raise CampaignError("campaign.build is missing")
    if build.get("torch_library_tree_sha256") != torch_tree:
        raise CampaignError("current Infinity Torch SDK bundle differs from campaign")
    expected_roots = build.get("torch_library_roots")
    if roots is not None and expected_roots is not None and expected_roots != roots:
        raise CampaignError("current Infinity Torch SDK roots differ from campaign")


def _verify_python_environment(campaign, runtime: str, digest: str):
    entry = _campaign_installation(campaign, runtime)
    if entry is None:
        return
    receipt = entry.get("python_environment")
    if receipt is None:
        return
    if not isinstance(receipt, dict) \
            or receipt.get("digest_sha256") != digest:
        raise CampaignError(
            f"current {runtime!r} Python environment differs from campaign receipt")


def runtime_files(args, runtime: str, campaign, cache=None) -> dict:
    cache = _invocation_cache(args, cache)
    key = ("runtime_files", runtime)
    if key in cache:
        return cache[key]
    binary = Path(_binary_path(args, runtime)).resolve()
    libraries = _native_libraries(args, cache)
    binary_sha = sha256_file(binary)
    probe = None
    torch_tree = None
    torch_roots = None
    if runtime == "infinity":
        package_tree = canonical_sha256(
            {"binary_sha256": binary_sha, "native_libraries": libraries})
        if getattr(args, "torch_lib", None):
            bundle = _torch_sdk_bundle(args.torch_lib, cache)
            torch_tree = bundle["digest_sha256"]
            torch_roots = bundle["roots"]
        _verify_infinity_torch(campaign, torch_tree, torch_roots)
    else:
        probe = runtime_package_probe(args, runtime, cache)
        package_tree = probe.get("package_tree_sha256")
        if not isinstance(package_tree, str) \
                or not re.fullmatch(r"[0-9a-f]{64}", package_tree):
            raise CampaignError(
                f"runtime package probe for {runtime!r} returned no package tree")
    python_environment = python_environment_sha256(args, runtime, cache)
    _verify_runtime_installation(campaign, runtime, package_tree, probe)
    _verify_python_environment(campaign, runtime, python_environment)
    files = {
        "runner_sha256": sha256_file(Path(__file__)),
        "client_sha256": sha256_file(HERE / "benchmark-http.py"),
        "binary_sha256": binary_sha,
        "native_library_sha256": libraries,
        "runtime_package_tree_sha256": package_tree,
        "torch_library_tree_sha256": torch_tree,
        "python_environment_sha256": python_environment,
        "integrity_helper_sha256": _source_file_sha256(
            INTEGRITY_HELPER, cache, "integrity_helper"),
        "campaign_inspector_sha256": _source_file_sha256(
            CAMPAIGN_INSPECTOR, cache, "campaign_inspector"),
    }
    cache[key] = files
    return files


def _verify_runtime_anchor(args, campaign, runtime: str, files: dict, cache) -> None:
    if not isinstance(campaign, dict):
        raise CampaignError("campaign.json must be an object")
    expected = campaign.get("fingerprint_files")
    if not isinstance(expected, dict) or runtime not in expected:
        raise CampaignError(f"campaign.fingerprint_files lacks {runtime!r}")
    if expected[runtime] != files:
        raise CampaignError(
            f"current {runtime!r} runtime files differ from campaign.fingerprint_files")
    method = campaign.get("method")
    identities = method.get("runtime_python_identity_sha256") \
        if isinstance(method, dict) else None
    if not isinstance(identities, dict) or runtime not in identities \
            or not isinstance(identities[runtime], str) \
            or not re.fullmatch(r"[0-9a-f]{64}", identities[runtime]):
        raise CampaignError(
            f"campaign.method.runtime_python_identity_sha256 lacks {runtime!r}")
    actual = runtime_python_identity_sha256(args, runtime, cache)
    if actual != identities[runtime]:
        raise CampaignError(
            f"current {runtime!r} runtime Python identity differs from campaign")


def _checkpoint_identity(receipt: dict, metadata: str, weights=None,
                         dependencies=None, code_dependencies=None) -> dict:
    if weights is None:
        weights_sha = receipt.get("weights_sha256")
    else:
        weights_sha = weights.get("weights_sha256")
    if dependencies is None:
        dependencies = receipt.get("dependencies")
    if code_dependencies is None:
        code_dependencies = receipt.get("code_dependencies")
    return {"id": receipt.get("id"), "revision": receipt.get("revision"),
            "metadata_sha256": metadata, "weights_sha256": weights_sha,
            "dependencies": _dependency_identity(dependencies),
            "code_dependencies": _dependency_identity(code_dependencies)}


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
    checkpoint_error = None
    try:
        fresh = checkpoint_identity(receipt, tree_cache)
        checkpoint = fresh["checkpoint"]
        checkpoint_identity_value = fresh["identity"]
        path = Path(checkpoint["path"])
    except Exception as exc:
        checkpoint_error = f"{type(exc).__name__}: {exc}"
        metadata = metadata_sha256(path) if path and path.is_dir() else ""
        checkpoint = dict(receipt)
        checkpoint.update(
            metadata_sha256=metadata, weight_files_sha256={},
            weights_sha256=None, weight_bytes=0,
            dependencies=[], code_dependencies=[])
        checkpoint_identity_value = _checkpoint_identity(
            {"id": receipt.get("id"), "revision": receipt.get("revision")},
            metadata, {"weights_sha256": None}, [], [])
    preflight_error = None
    try:
        settings = _settings(args, model, runtime,
                             path if path and path.is_dir() else None, tree_cache)
    except Exception as exc:
        preflight_error = f"{type(exc).__name__}: {exc}"
        settings = _settings(args, model, runtime, None, tree_cache)
    files = runtime_files(args, runtime, campaign, tree_cache)
    _verify_runtime_anchor(args, campaign, runtime, files, tree_cache)
    inputs = fingerprint_inputs(model, checkpoint_identity_value, runtime,
                                gpu, cpus, client_cpus, settings, files)
    current = fingerprint(inputs)
    item = {"model": model["id"], "task": model["task"], "runtime": runtime,
            "status": "load_failed", "gpu": gpu, "cpus": cpus,
            "client_cpus": client_cpus, "dtype": args.dtype,
            "checkpoint": checkpoint, "command": None, "client_command": None,
            "run_fingerprint": current, "fingerprint_inputs": inputs,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    identity_error = receipt_identity_error(model, receipt)
    if identity_error or checkpoint_error or not path or not path.is_dir():
        for kind, owned in (("runs", target), ("clients", client_result),
                            ("logs", log)):
            _archive_previous(args.output, kind, owned)
        item.update(status="checkpoint_unavailable",
                    error=identity_error or checkpoint_error
                    or "checkpoint path is unavailable")
        save_json(target, item)
        return item
    decision = resume_decision(target, current, args.retry_failures)
    if decision == "skip":
        return _load_json(target)
    if decision == "archive":
        for kind, owned in (("runs", target), ("clients", client_result), ("logs", log)):
            _archive_previous(args.output, kind, owned)
    _clear_client_result(client_result)
    item["status"] = "load_failed"
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
        environment, recorded_env = _environment(args, runtime, gpu, tree_cache)
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


def _preflight_checkpoints(args, models: list[dict], cache: dict) -> None:
    for model in models:
        receipt = _receipt_for(
            model, args.output / "checkpoints" / (slug(model["id"]) + ".json"))
        try:
            checkpoint_identity(receipt, cache)
        except Exception:
            pass


def _preflight_runtimes(args, campaign, runtimes, cache: dict) -> None:
    method = campaign.get("method") if isinstance(campaign, dict) else None
    identities = method.get("runtime_python_identity_sha256") \
        if isinstance(method, dict) else None
    if not isinstance(identities, dict) or set(identities) != set(RUNTIMES) \
            or any(not isinstance(value, str)
                   or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in identities.values()):
        raise CampaignError(
            "campaign.method.runtime_python_identity_sha256 must map exactly "
            f"{list(RUNTIMES)} to 64-hex identities")
    anchors = campaign.get("fingerprint_files")
    if not isinstance(anchors, dict):
        raise CampaignError("campaign.fingerprint_files must be an object")
    for runtime in runtimes:
        if runtime not in anchors:
            raise CampaignError(f"campaign.fingerprint_files lacks {runtime!r}")
        files = runtime_files(args, runtime, campaign, cache)
        _verify_runtime_anchor(args, campaign, runtime, files, cache)


def run(args, models: list[dict], campaign) -> None:
    gpus = parse_gpus(args.gpus)
    server_lanes = parse_lanes(args.server_cpus, "--server-cpus")
    client_lanes = parse_lanes(args.client_cpus, "--client-cpus")
    if len(server_lanes) != len(client_lanes):
        raise CampaignError("--server-cpus and --client-cpus must have equal counts")
    placements = _placements(args, models, gpus)
    tree_cache = {}
    args._benchmark_cache = tree_cache
    if any(2048 in model["lengths"] for model in models) \
            and args.long_input_concurrency != 8:
        raise CampaignError("the client fixes 2048-token concurrency to 8")
    _preflight_checkpoints(args, models, tree_cache)
    _preflight_runtimes(args, campaign, args.runtimes, tree_cache)
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
        def make_venv(name):
            base = root / name
            bin_dir = base / "bin"
            bin_dir.mkdir(parents=True)
            python = bin_dir / "python"
            python.symlink_to(sys.executable)
            (base / "pyvenv.cfg").write_text(
                f"home = {Path(sys.executable).resolve().parent}\n"
                "include-system-site-packages = false\n", encoding="utf-8")
            return python

        def make_wrapper(name, python, pythonpath=None):
            path = root / "wrappers" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            environment = ""
            if pythonpath is not None:
                environment = f"PYTHONPATH={shlex.quote(str(pythonpath))} "
            path.write_text(
                f"#!/bin/sh\n{environment}exec {shlex.quote(str(python))} \"$@\"\n",
                encoding="utf-8")
            path.chmod(0o755)
            return path

        inf_bin = root / "infinity" / "bin"
        inf_bin.mkdir(parents=True)
        infinity = inf_bin / "infinity"
        infinity.write_bytes(b"infinity-binary")
        bridge_names = ["libinfinity_backend.so", "libinfinity_fa4.so",
                        "libinfinity_cuda.so", "libtorch_bridge.so", "libc10_bridge.so"]
        for name in bridge_names:
            (inf_bin / name).write_bytes(("bridge-" + name).encode())
        sdk_site = root / "sdk-site"
        torch_lib = sdk_site / "torch" / "lib"
        (torch_lib / "nested").mkdir(parents=True)
        (torch_lib / "libtorch.so").write_bytes(b"torch-library")
        (torch_lib / "nested" / "libc10.so").write_bytes(b"c10-library")
        nvidia_lib = sdk_site / "nvidia" / "cudnn" / "lib"
        nvidia_lib.mkdir(parents=True)
        (nvidia_lib / "libcudnn.so").write_bytes(b"cudnn-library")
        integrity_marker = root / "integrity-marker"
        integrity_marker.write_bytes(b"python-environment-v1")
        fake_integrity = root / "fake-integrity.py"
        fake_integrity.write_text(
            "import hashlib, json, os\n"
            f"marker = {str(integrity_marker)!r}\n"
            "leaked = next((name for name in "
            "('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'PYTHONPATH', "
            "'PYTHONHOME') if os.environ.get(name) == 'sentinel-' + name), None)\n"
            "data = (b'leaked:' + leaked.encode()) if leaked else "
            "open(marker, 'rb').read()\n"
            "print(json.dumps({'schema_version': 1, 'scope': 'fixture', "
            "'digest_sha256': hashlib.sha256(data).hexdigest(), "
            "'file_count': 1, 'byte_count': len(data), 'distributions': [], "
            "'content_roots': [], 'interpreter': {}}, sort_keys=True))\n",
            encoding="utf-8")
        original_integrity_helper = INTEGRITY_HELPER
        globals()["INTEGRITY_HELPER"] = fake_integrity
        client_python = make_venv("client-env")
        vllm_python = make_venv("vllm-env")
        sg_python = make_venv("sg-env")
        vllm_wrapper = make_wrapper("vllm-python", vllm_python)
        env_args = argparse.Namespace(
            infinity_bin=str(infinity), torch_lib=str(torch_lib),
            vllm_python=str(vllm_wrapper), sglang_python=str(sg_python),
            client_python=str(client_python))
        sentinel_values = {name: f"sentinel-{name}"
                           for name in SCRUBBED_LOADER_VARIABLES}
        previous_environment = {
            name: os.environ.get(name) for name in sentinel_values}
        os.environ.update(sentinel_values)
        try:
            cache = {}
            inf_env, inf_record = _environment(env_args, "infinity", 0, cache)
            vllm_env, vllm_record = _environment(env_args, "vllm", 0, cache)
            sg_env, sg_record = _environment(env_args, "sglang", 0, cache)
            settings_args = argparse.Namespace(
                **{**vars(env_args), "dtype": "bfloat16", "seconds": 10.0,
                   "repeats": 3, "warmup": 3.0, "concurrency": 32,
                   "long_input_concurrency": 8, "max_batch_tokens": 16384,
                   "max_sequence_length": 2048})
            vllm_settings = _settings(settings_args, model, "vllm", None, cache)
            sentinel_probe_digest = python_environment_sha256(
                settings_args, "vllm", cache)
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        assert "LD_LIBRARY_PATH" not in vllm_env
        assert "LD_LIBRARY_PATH" not in sg_env
        assert "sentinel" not in json.dumps(inf_env)
        assert "sentinel" not in json.dumps(vllm_env)
        assert inf_env["LD_LIBRARY_PATH"] == os.pathsep.join(
            [str(inf_bin.resolve()), str(torch_lib.resolve())])
        assert inf_env["PYTHONSAFEPATH"] == "1"
        assert vllm_env["PYTHONSAFEPATH"] == "1"
        for name in SCRUBBED_LOADER_VARIABLES:
            if name != "LD_LIBRARY_PATH":
                assert name not in inf_env and name not in vllm_env
        assert inf_record["LD_LIBRARY_PATH_injected"] == [
            str(inf_bin.resolve()), str(torch_lib.resolve())]
        assert vllm_record["LD_LIBRARY_PATH_injected"] == []
        assert inf_record["LD_LIBRARY_PATH_inherited"] == ""
        assert vllm_record["LD_LIBRARY_PATH_inherited"] == ""
        assert inf_record["PYTHONSAFEPATH"] == "1"
        assert inf_record["loader_policy"] == loader_policy("infinity")
        assert vllm_record["loader_policy"] == loader_policy("vllm")
        assert "sentinel" not in json.dumps(inf_record)
        assert "sentinel" not in json.dumps(vllm_record)
        assert sentinel_probe_digest == sha256_bytes(integrity_marker.read_bytes())
        client_bin = str(Path(str(client_python)).absolute().parent)
        vllm_bin = str(Path(str(vllm_python)).absolute().parent)
        sg_bin = str(Path(str(sg_python)).absolute().parent)
        assert inf_env["PATH"].startswith(client_bin + os.pathsep)
        assert vllm_env["PATH"].startswith(vllm_bin + os.pathsep)
        assert sg_env["PATH"].startswith(sg_bin + os.pathsep)
        assert not vllm_env["PATH"].startswith(sg_bin + os.pathsep)
        assert inf_record["PATH_prepend"] == client_bin
        assert vllm_record["PATH_prepend"] == vllm_bin
        assert sg_record["PATH_prepend"] == sg_bin
        assert inf_record["python_executable"] == str(client_python)
        assert vllm_record["python_executable"] == str(vllm_python)
        assert inf_record["python_prefix"] == str(client_python.parent.parent)
        assert vllm_record["python_prefix"] == str(vllm_python.parent.parent)
        assert "python_identity_sha256" not in inf_env
        for record, runtime in ((inf_record, "infinity"),
                                (vllm_record, "vllm"), (sg_record, "sglang")):
            identity = record["python_identity_sha256"]
            assert re.fullmatch(r"[0-9a-f]{64}", identity)
            assert identity == runtime_python_identity_sha256(env_args, runtime, cache)
        assert vllm_settings["runtime_environment"]["python_identity_sha256"] \
            == vllm_record["python_identity_sha256"]
        assert vllm_settings["runtime_environment"]["loader_policy"] \
            == vllm_record["loader_policy"]
        switched_wrapper = make_wrapper(
            "vllm-switched", make_venv("vllm-env-switched"))
        switched_args = argparse.Namespace(
            **{**vars(env_args), "vllm_python": str(switched_wrapper)})
        assert runtime_python_identity_sha256(switched_args, "vllm", {}) \
            != runtime_python_identity_sha256(env_args, "vllm", {})

        files = runtime_files(env_args, "infinity", None, {})
        assert set(files) == {
            "runner_sha256", "client_sha256", "binary_sha256",
            "native_library_sha256", "runtime_package_tree_sha256",
            "torch_library_tree_sha256", "python_environment_sha256",
            "integrity_helper_sha256", "campaign_inspector_sha256"}
        assert set(files["native_library_sha256"]) == set(bridge_names)
        bundle = torch_sdk_bundle(torch_lib)
        assert set(bundle["roots"]) == {"torch_lib", "nvidia"}
        assert files["torch_library_tree_sha256"] == bundle["digest_sha256"]
        alias_lib = root / "sdk-libs-alias"
        alias_lib.symlink_to(torch_lib, target_is_directory=True)
        assert torch_sdk_bundle(alias_lib) == bundle
        assert files["python_environment_sha256"] == sha256_bytes(
            integrity_marker.read_bytes())
        fake_integrity_b = root / "fake-integrity-b.py"
        fake_integrity_b.write_text(
            fake_integrity.read_text(encoding="utf-8") + "# source-only-change\n",
            encoding="utf-8")
        original_helper_for_source = INTEGRITY_HELPER
        globals()["INTEGRITY_HELPER"] = fake_integrity
        helper_a_files = runtime_files(env_args, "infinity", None, {})
        globals()["INTEGRITY_HELPER"] = fake_integrity_b
        helper_b_files = runtime_files(env_args, "infinity", None, {})
        globals()["INTEGRITY_HELPER"] = original_helper_for_source
        assert helper_a_files["python_environment_sha256"] \
            == helper_b_files["python_environment_sha256"]
        assert helper_a_files["integrity_helper_sha256"] \
            != helper_b_files["integrity_helper_sha256"]
        helper_anchor = {
            "fingerprint_files": {"infinity": helper_a_files},
            "method": {"runtime_python_identity_sha256": {
                "infinity": inf_record["python_identity_sha256"]}}}
        _expect(CampaignError, _verify_runtime_anchor, env_args, helper_anchor,
                "infinity", helper_b_files, {})
        inspector_a = root / "fake-inspector-a.py"
        inspector_b = root / "fake-inspector-b.py"
        inspector_a.write_text("# inspector-a\n", encoding="utf-8")
        inspector_b.write_text("# inspector-b\n", encoding="utf-8")
        original_inspector = CAMPAIGN_INSPECTOR
        globals()["CAMPAIGN_INSPECTOR"] = inspector_a
        inspector_a_files = runtime_files(env_args, "infinity", None, {})
        globals()["CAMPAIGN_INSPECTOR"] = inspector_b
        inspector_b_files = runtime_files(env_args, "infinity", None, {})
        globals()["CAMPAIGN_INSPECTOR"] = original_inspector
        assert inspector_a_files["campaign_inspector_sha256"] \
            != inspector_b_files["campaign_inspector_sha256"]
        assert inspector_a_files["python_environment_sha256"] \
            == inspector_b_files["python_environment_sha256"]
        inspector_anchor = {
            "fingerprint_files": {"infinity": inspector_a_files},
            "method": {"runtime_python_identity_sha256": {
                "infinity": inf_record["python_identity_sha256"]}}}
        _expect(CampaignError, _verify_runtime_anchor, env_args, inspector_anchor,
                "infinity", inspector_b_files, {})
        fingerprint_values = [fingerprint(fingerprint_inputs(
            model, _checkpoint_identity(
                {"id": model["id"], "revision": model["revision"]}, "f" * 64),
            "infinity", 0, "0-7", "32-39", {"seconds": 10, "repeats": 3}, files))]
        assert str(root) not in json.dumps(fingerprint_inputs(
            model, _checkpoint_identity(
                {"id": model["id"], "revision": model["revision"]}, "f" * 64),
            "infinity", 0, "0-7", "32-39", {"seconds": 10, "repeats": 3}, files))
        moved_site = root / "sdk-relocated"
        shutil.copytree(sdk_site, moved_site, symlinks=True)
        moved_args = argparse.Namespace(
            **{**vars(env_args), "torch_lib": str(moved_site / "torch" / "lib")})
        assert runtime_files(moved_args, "infinity", None, {})[
            "torch_library_tree_sha256"] == files["torch_library_tree_sha256"]
        standalone = root / "standalone-torch" / "lib"
        standalone.mkdir(parents=True)
        (standalone / "libtorch.so").write_bytes(b"standalone")
        standalone_bundle = torch_sdk_bundle(standalone)
        assert set(standalone_bundle["roots"]) == {"torch_lib"}
        no_sdk = argparse.Namespace(**{**vars(env_args), "torch_lib": None})
        assert runtime_files(no_sdk, "infinity", None, {})[
            "torch_library_tree_sha256"] is None
        nvidia_bytes = (nvidia_lib / "libcudnn.so").read_bytes()
        (nvidia_lib / "libcudnn.so").write_bytes(
            b"X" + nvidia_bytes[1:])
        nvidia_changed = runtime_files(env_args, "infinity", None, {})
        assert nvidia_changed["torch_library_tree_sha256"] != files[
            "torch_library_tree_sha256"]
        assert torch_sdk_bundle(alias_lib)["digest_sha256"] != bundle["digest_sha256"]
        (nvidia_lib / "libcudnn.so").write_bytes(nvidia_bytes)
        (torch_lib / "libtorch.so").write_bytes(b"torch-library-changed")
        changed = runtime_files(env_args, "infinity", None, {})
        assert changed["torch_library_tree_sha256"] != files["torch_library_tree_sha256"]
        changed_fingerprint = fingerprint(fingerprint_inputs(
            model, _checkpoint_identity(
                {"id": model["id"], "revision": model["revision"]}, "f" * 64),
            "infinity", 0, "0-7", "32-39", {"seconds": 10, "repeats": 3}, changed))
        assert changed_fingerprint != fingerprint_values[0]
        resume_raw = root / "torch-resume.json"
        save_json(resume_raw, {"run_fingerprint": fingerprint_values[0],
                               "status": "measured", "measurement": {"ok": True}})
        assert resume_decision(resume_raw, changed_fingerprint, False) == "archive"
        (torch_lib / "nested" / "new.so").write_bytes(b"new-library")
        assert runtime_files(env_args, "infinity", None, {})[
            "torch_library_tree_sha256"] != changed["torch_library_tree_sha256"]

        checkpoint = root / "checkpoint-weights"
        checkpoint.mkdir()
        shard_a = checkpoint / "model-00001-of-00002.safetensors"
        shard_b = checkpoint / "model-00002-of-00002.safetensors"
        shard_a.write_bytes(b"shard-a")
        shard_b.write_bytes(b"shard-b")
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": shard_a.name, "b": shard_b.name}}),
            encoding="utf-8")
        weight_info = checkpoint_weights(checkpoint, cache={})
        assert set(weight_info["weight_files_sha256"]) == {shard_a.name, shard_b.name}
        assert weight_info["weights_sha256"] == canonical_sha256(
            weight_info["weight_files_sha256"])
        missing = root / "weights-missing"
        shutil.copytree(checkpoint, missing)
        (missing / shard_b.name).unlink()
        _expect(CampaignError, checkpoint_weights, missing, (), (), {})
        escaping = root / "weights-escaping"
        shutil.copytree(checkpoint, escaping)
        (escaping / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "../escape.safetensors"}}), encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, escaping, (), (), {})
        single = root / "weights-single"
        single.mkdir()
        (single / "model.safetensors").write_bytes(b"single")
        assert set(checkpoint_weights(single, cache={})["weight_files_sha256"]) \
            == {"model.safetensors"}
        nested = root / "weights-nested"
        (nested / "1_Dense").mkdir(parents=True)
        (nested / "model.safetensors").write_bytes(b"nested-main")
        dense_weight = nested / "1_Dense" / "pytorch_model.bin"
        dense_weight.write_bytes(b"dense-v1")
        nested_info = checkpoint_weights(nested, cache={})
        assert set(nested_info["weight_files_sha256"]) == {
            "model.safetensors", "1_Dense/pytorch_model.bin"}
        nested_identity = _checkpoint_identity(
            {"id": model["id"], "revision": model["revision"]}, "f" * 64,
            nested_info, [], [])
        dense_weight.write_bytes(b"dense-v2")
        nested_changed = checkpoint_weights(nested, cache={})
        assert nested_changed["weights_sha256"] != nested_info["weights_sha256"]
        assert _checkpoint_identity(
            {"id": model["id"], "revision": model["revision"]}, "f" * 64,
            nested_changed, [], [])["weights_sha256"] \
            != nested_identity["weights_sha256"]
        preferred_failure = root / "weights-preferred-safe-failure"
        preferred_failure.mkdir()
        (preferred_failure / "pytorch_model.bin").write_bytes(b"bin-valid")
        (preferred_failure / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, preferred_failure, (), (), {})
        unused_bin_index = root / "weights-unused-bin-index"
        unused_bin_index.mkdir()
        (unused_bin_index / "model.safetensors").write_bytes(b"safe-main")
        (unused_bin_index / "pytorch_model.bin.index.json").write_text(json.dumps(
            {"weight_map": {"a": "pytorch_model-00001.bin",
                            "b": "pytorch_model-00002.bin"}}), encoding="utf-8")
        unused_info = checkpoint_weights(unused_bin_index, cache={})
        assert set(unused_info["weight_files_sha256"]) == {"model.safetensors"}
        bin_preferred_failure = root / "weights-preferred-bin-failure"
        bin_preferred_failure.mkdir()
        (bin_preferred_failure / "pytorch_model-00001.bin").write_bytes(b"bin-one")
        (bin_preferred_failure / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {
                "a": "pytorch_model-00001.bin",
                "b": "pytorch_model-00002.bin"}}), encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, bin_preferred_failure, (), (), {})
        nested_preferred = root / "weights-nested-preferred"
        (nested_preferred / "1_Dense").mkdir(parents=True)
        (nested_preferred / "model.safetensors").write_bytes(b"nested-root")
        (nested_preferred / "1_Dense" / "pytorch_model.bin").write_bytes(b"bin")
        (nested_preferred / "1_Dense" / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "a": "model-00001-of-00002.safetensors"}}), encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, nested_preferred, (), (), {})
        windows_escape = root / "weights-windows"
        windows_escape.mkdir()
        (windows_escape / "model.safetensors").write_bytes(b"safe")
        (windows_escape / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "C:\\escape.safetensors"}}), encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, windows_escape, (), (), {})
        unc_escape = root / "weights-unc"
        unc_escape.mkdir()
        (unc_escape / "model.safetensors").write_bytes(b"safe")
        (unc_escape / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "\\\\server\\share\\model.safetensors"}}),
            encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, unc_escape, (), (), {})
        tokenizer_only = root / "weights-tokenizer"
        tokenizer_only.mkdir()
        (tokenizer_only / "tokenizer.bin").write_bytes(b"tokenizer")
        _expect(CampaignError, checkpoint_weights, tokenizer_only, (), (), {})
        empty_weights = root / "weights-empty"
        empty_weights.mkdir()
        _expect(CampaignError, checkpoint_weights, empty_weights, (), (), {})
        mutable = root / "weights-mutable"
        shutil.copytree(checkpoint, mutable)
        before = checkpoint_weights(mutable, cache={})
        before_identity = _checkpoint_identity(
            {"id": model["id"], "revision": model["revision"]}, "f" * 64,
            before, [], [])
        before_fingerprint = fingerprint(fingerprint_inputs(
            model, before_identity, "infinity", 0, "0-7", "32-39",
            {"seconds": 10, "repeats": 3}, files))
        original_shard = (mutable / shard_a.name).read_bytes()
        (mutable / shard_a.name).write_bytes(
            b"X" + original_shard[1:])
        after = checkpoint_weights(mutable, cache={})
        assert len((mutable / shard_a.name).read_bytes()) == len(original_shard)
        assert after["weights_sha256"] != before["weights_sha256"]
        after_fingerprint = fingerprint(fingerprint_inputs(
            model, _checkpoint_identity(
                {"id": model["id"], "revision": model["revision"]}, "f" * 64,
                after, [], []), "infinity", 0, "0-7", "32-39",
            {"seconds": 10, "repeats": 3}, files))
        assert after_fingerprint != before_fingerprint
        save_json(root / "weights-resume.json", {
            "run_fingerprint": before_fingerprint, "status": "measured",
            "measurement": {"ok": True}})
        assert resume_decision(root / "weights-resume.json",
                               after_fingerprint, False) == "archive"
        shared_weights = {}
        cached_before = checkpoint_weights(mutable, cache=shared_weights)
        (mutable / shard_a.name).write_bytes(b"Y" + original_shard[1:])
        assert checkpoint_weights(mutable, cache=shared_weights) == cached_before
        freshened = checkpoint_weights(mutable, cache={})
        assert freshened["weights_sha256"] != cached_before["weights_sha256"]
        dependency = root / "weight-dependency"
        dependency.mkdir()
        (dependency / "model.safetensors").write_bytes(b"dependency-v1")
        (dependency / "config.json").write_text('{"version":1}', encoding="utf-8")
        dependency_items = [{"id": "org/dependency", "revision": "b" * 40,
                             "path": str(dependency), "metadata_sha256": "stale"}]
        dependency_weights = checkpoint_weights(
            mutable, dependency_items, dependency_items, cache={})
        assert any(name.startswith("dependencies/org--dependency@")
                   for name in dependency_weights["weight_files_sha256"])
        assert any(name.startswith("code_dependencies/org--dependency@")
                   for name in dependency_weights["weight_files_sha256"])
        malformed_dependency = root / "malformed-dependency"
        malformed_dependency.mkdir()
        (malformed_dependency / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {"a": "missing.bin"}}), encoding="utf-8")
        _expect(CampaignError, checkpoint_weights, mutable,
                [{"id": "org/malformed", "revision": "c" * 40,
                  "path": str(malformed_dependency)}], (), {})
        duplicate_dependency = root / "duplicate-dependency"
        duplicate_dependency.mkdir()
        (duplicate_dependency / "model.safetensors").write_bytes(b"duplicate")
        _expect(CampaignError, checkpoint_weights, mutable,
                dependency_items,
                [{"id": "org/dependency", "revision": "b" * 40,
                  "path": str(duplicate_dependency)}], {})
        dependency_descriptors = _refresh_dependency_descriptors(
            dependency_items, {})
        dependency_identity = _checkpoint_identity(
            {"id": model["id"], "revision": model["revision"]}, "f" * 64,
            dependency_weights, dependency_descriptors, dependency_descriptors)
        (dependency / "model.safetensors").write_bytes(b"dependency-v2")
        (dependency / "config.json").write_text('{"version":2}', encoding="utf-8")
        changed_descriptors = _refresh_dependency_descriptors(
            dependency_items, {})
        changed_weights = checkpoint_weights(
            mutable, changed_descriptors, changed_descriptors, cache={})
        changed_identity = _checkpoint_identity(
            {"id": model["id"], "revision": model["revision"]}, "f" * 64,
            changed_weights, changed_descriptors, changed_descriptors)
        assert dependency_identity["weights_sha256"] \
            != changed_identity["weights_sha256"]
        assert dependency_identity["dependencies"] \
            != changed_identity["dependencies"]
        metadata_only = root / "metadata-only-dependency"
        metadata_only.mkdir()
        (metadata_only / "config.json").write_text('{"metadata":1}', encoding="utf-8")
        receipt = {"id": model["id"], "revision": model["revision"],
                   "status": "ready", "path": str(single),
                   "dependencies": [{"id": "org/meta", "revision": "d" * 40,
                                     "path": str(metadata_only),
                                     "metadata_sha256": "stale"}],
                   "code_dependencies": [],
                   "weight_files_sha256": {"stale.bin": "f" * 64},
                   "weights_sha256": "f" * 64}
        fresh = checkpoint_identity(receipt, {})
        assert set(fresh) == {"checkpoint", "identity"}
        assert fresh["checkpoint"]["weights_sha256"] \
            != receipt["weights_sha256"]
        assert "weight_files_sha256" in fresh["checkpoint"]
        assert fresh["identity"]["weights_sha256"] \
            == fresh["checkpoint"]["weights_sha256"]
        assert fresh["identity"]["dependencies"][0]["metadata_sha256"] \
            == metadata_sha256(metadata_only)
        assert str(root) not in json.dumps(fresh["identity"])
        shared_identity = {}
        first_api = checkpoint_identity(receipt, shared_identity)
        original_single = (single / "model.safetensors").read_bytes()
        (single / "model.safetensors").write_bytes(b"X" + original_single[1:])
        assert checkpoint_identity(receipt, shared_identity) == first_api
        assert checkpoint_identity(receipt, {})["identity"]["weights_sha256"] \
            != first_api["identity"]["weights_sha256"]
        (single / "model.safetensors").write_bytes(original_single)
        missing_dependency = root / "missing-dependency"
        empty_dependency = root / "empty-dependency"
        empty_dependency.mkdir()
        broken = {**receipt,
                  "dependencies": [{"id": "org/meta", "revision": "d" * 40,
                                    "path": str(missing_dependency),
                                    "metadata_sha256": "a" * 64}]}
        _expect(CampaignError, checkpoint_identity, broken, {})
        no_weights = {**receipt, "path": str(empty_weights),
                      "dependencies": [], "code_dependencies": []}
        _expect(CampaignError, checkpoint_identity, no_weights, {})

        site = root / "vllm-site"
        package = site / "vllm"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (package / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        dist_info = site / "vllm-0.0.1.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: vllm\nVersion: 0.0.1\n\n", encoding="utf-8")
        dependency_metadata = []
        for dependency in ("torch", "transformers"):
            directory = site / f"{dependency}-0.0.1.dist-info"
            directory.mkdir()
            metadata = directory / "METADATA"
            metadata.write_text(f"Metadata-Version: 2.1\nName: {dependency}\nVersion: 0.0.1\n\n")
            dependency_metadata.append(metadata)
        def record_line(relative):
            data = (site / relative).read_bytes()
            encoded = base64.urlsafe_b64encode(
                hashlib.sha256(data).digest()).decode().rstrip("=")
            return f"{relative},sha256={encoded},{len(data)}\n"
        (dist_info / "RECORD").write_text(
            "".join(record_line(item) for item in ("vllm/__init__.py",
                                                    "vllm/model.py")),
            encoding="utf-8")
        package_wrapper = make_wrapper(
            "vllm-package-python", make_venv("vllm-package-env"), site)
        package_args = argparse.Namespace(
            **{**vars(env_args), "vllm_python": str(package_wrapper)})
        probe = runtime_package_probe(package_args, "vllm", {})
        package_campaign = {"runtime_installations": {"vllm": {
            "version": probe["version"],
            "torch_version": probe["torch_version"],
            "transformers_version": probe["transformers_version"],
            "package_tree_sha256": probe["package_tree_sha256"],
            "record_sha256": probe["record_sha256"],
            "record_mismatches": [], "unrecorded_files": [],
            "missing_record_files": []}}}
        package_files = runtime_files(package_args, "vllm", package_campaign, {})
        assert package_files["torch_library_tree_sha256"] is None
        metadata = dist_info / "METADATA"
        original_metadata = metadata.read_text()
        metadata.write_text(original_metadata.replace("Version: 0.0.1", "Version: 0.0.2"))
        _expect(CampaignError, runtime_files, package_args, "vllm", package_campaign, {})
        metadata.write_text(original_metadata)
        for metadata in dependency_metadata:
            original_metadata = metadata.read_text()
            metadata.write_text(original_metadata.replace("Version: 0.0.1", "Version: 0.0.2"))
            _expect(CampaignError, runtime_files, package_args, "vllm", package_campaign, {})
            metadata.write_text(original_metadata)
        (package / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
        _expect(CampaignError, runtime_files, package_args, "vllm",
                package_campaign, {})
        (package / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        assert runtime_files(package_args, "vllm", package_campaign, {}) \
            == package_files
        (package / "__init__.py").unlink()
        _expect(CampaignError, runtime_files, package_args, "vllm",
                package_campaign, {})
        (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        assert runtime_files(package_args, "vllm", package_campaign, {}) \
            == package_files
        original_probe = globals()["_run_probe"]
        calls = []
        def counting_probe(command, label, **keywords):
            calls.append(label)
            return original_probe(command, label, **keywords)
        globals()["_run_probe"] = counting_probe
        try:
            shared = {}
            runtime_package_probe(package_args, "vllm", shared)
            runtime_package_probe(package_args, "vllm", shared)
            assert len(calls) == 1
            runtime_package_probe(package_args, "vllm", {})
            assert len(calls) == 2
        finally:
            globals()["_run_probe"] = original_probe

        inf_files = runtime_files(env_args, "infinity", None, {})
        vllm_files = runtime_files(package_args, "vllm", package_campaign, {})
        run_campaign = {
            "fingerprint_files": {"infinity": inf_files, "vllm": vllm_files},
            "method": {"runtime_python_identity_sha256": {
                "infinity": runtime_python_identity_sha256(env_args, "infinity", {}),
                "vllm": runtime_python_identity_sha256(package_args, "vllm", {}),
                "sglang": runtime_python_identity_sha256(env_args, "sglang", {})}},
            "build": {"torch_library_tree_sha256":
                      inf_files["torch_library_tree_sha256"],
                      "torch_library_roots": torch_sdk_bundle(torch_lib)["roots"]},
            "runtime_installations": {
                "infinity": {"package_tree_sha256": canonical_sha256({
                    "binary_sha256": inf_files["binary_sha256"],
                    "native_libraries": inf_files["native_library_sha256"]}),
                    "python_environment": {
                        "digest_sha256": inf_files["python_environment_sha256"]}},
                "vllm": {**package_campaign["runtime_installations"]["vllm"],
                         "python_environment": {
                             "digest_sha256": vllm_files[
                                 "python_environment_sha256"]}}}}
        run_args = argparse.Namespace(
            **{**vars(env_args), "vllm_python": str(package_wrapper),
               "runtimes": ["infinity", "vllm"],
               "output": root / "run-preflight", "gpus": "0",
               "server_cpus": "0-7", "client_cpus": "8-15",
               "model_lanes": None, "long_input_concurrency": 8,
               "retry_failures": False})
        run_calls, launch_commands = [], []
        original_run_one = globals()["run_one"]
        original_resume = globals()["resume_decision"]
        original_popen = subprocess.Popen
        def fake_run_one(*arguments, **keywords):
            run_calls.append("run_one")
            return {}
        def fake_resume(*arguments, **keywords):
            run_calls.append("resume_decision")
            return "run"
        def recording_popen(command, *arguments, **keywords):
            launch_commands.append(command)
            return original_popen(command, *arguments, **keywords)
        globals()["run_one"] = fake_run_one
        globals()["resume_decision"] = fake_resume
        subprocess.Popen = recording_popen
        try:
            (package / "model.py").write_text("MODEL = 2\n", encoding="utf-8")
            _expect(CampaignError, run, run_args, [model], run_campaign)
        finally:
            subprocess.Popen = original_popen
            globals()["run_one"] = original_run_one
            globals()["resume_decision"] = original_resume
            (package / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
        assert run_calls == []
        assert all(Path(str(command[0])).name != "taskset"
                   for command in launch_commands if command)

        original_marker = integrity_marker.read_bytes()
        run_calls.clear()
        launch_commands.clear()
        globals()["run_one"] = fake_run_one
        globals()["resume_decision"] = fake_resume
        subprocess.Popen = recording_popen
        integrity_marker.write_bytes(b"python-environment-v2")
        try:
            _expect(CampaignError, run, run_args, [model], run_campaign)
            _expect(CampaignError, _preflight_runtimes, run_args, run_campaign,
                    ["infinity", "vllm"], {})
        finally:
            subprocess.Popen = original_popen
            globals()["run_one"] = original_run_one
            globals()["resume_decision"] = original_resume
            integrity_marker.write_bytes(original_marker)
        assert run_calls == []
        assert all(Path(str(command[0])).name != "taskset"
                   for command in launch_commands if command)

        run_one_output = root / "run-one-output"
        run_one_receipt = run_one_output / "checkpoints" / "org--model.json"
        bad_checkpoint = root / "bad-checkpoint"
        bad_checkpoint.mkdir()
        (bad_checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"present")
        (bad_checkpoint / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors"}}),
            encoding="utf-8")
        save_json(run_one_receipt, {
            "id": model["id"], "revision": model["revision"],
            "status": "ready", "path": str(bad_checkpoint)})
        old_target = run_one_output / "runs" / "org--model--infinity.json"
        save_json(old_target, {
            "run_fingerprint": "old", "status": "measured",
            "measurement": {"ok": True},
            "checkpoint": {"weight_files_sha256": {"old.safetensors": "e" * 64},
                           "weights_sha256": "e" * 64}})
        run_one_args = argparse.Namespace(
            **{**vars(run_args), "output": run_one_output,
               "dtype": "bfloat16", "seconds": 10.0, "repeats": 3,
               "warmup": 3.0, "concurrency": 32, "max_batch_tokens": 16384,
               "max_sequence_length": 2048, "infinity_port_base": 18200,
               "vllm_port_base": 18300, "sglang_port_base": 18400,
               "prometheus_port_offset": 1000, "startup_timeout": 1.0,
               "client_timeout": 1.0})
        failed_resume_calls = []
        original_resume_decision = globals()["resume_decision"]
        def failing_resume(*arguments, **keywords):
            failed_resume_calls.append("resume")
            return "skip"
        launch_commands.clear()
        globals()["resume_decision"] = failing_resume
        subprocess.Popen = recording_popen
        try:
            failed_raw = run_one(run_one_args, model, "infinity", 0,
                                 "0-7", "32-39", run_campaign, {})
        finally:
            subprocess.Popen = original_popen
            globals()["resume_decision"] = original_resume_decision
        assert failed_raw["status"] == "checkpoint_unavailable"
        assert "UnboundLocalError" not in failed_raw["error"]
        assert "is missing" in failed_raw["error"]
        assert failed_raw["checkpoint"]["weight_files_sha256"] == {}
        assert failed_raw["checkpoint"]["weights_sha256"] is None
        assert failed_raw["fingerprint_inputs"]["checkpoint"]["weights_sha256"] is None
        assert failed_resume_calls == []
        assert all(Path(str(command[0])).name != "taskset"
                   for command in launch_commands if command)
        archived = list((run_one_output / "previous" / "runs").glob("*.json"))
        assert len(archived) == 1
        assert _load_json(archived[0])["run_fingerprint"] == "old"
        assert _load_json(old_target)["status"] == "checkpoint_unavailable"

        valid_checkpoint = root / "valid-checkpoint"
        valid_checkpoint.mkdir()
        (valid_checkpoint / "config.json").write_text(
            '{"hidden_size":64,"num_attention_heads":1}', encoding="utf-8")
        (valid_checkpoint / "model.safetensors").write_bytes(b"valid-weight")
        valid_output = root / "valid-run-one"
        save_json(valid_output / "checkpoints" / "org--model.json", {
            "id": model["id"], "revision": model["revision"],
            "status": "ready", "path": str(valid_checkpoint)})
        measured_raw = {"run_fingerprint": "measured-old", "status": "measured",
                        "measurement": {"ok": True}}
        save_json(valid_output / "runs" / "org--model--infinity.json",
                  measured_raw)
        valid_args = argparse.Namespace(
            **{**vars(run_one_args), "output": valid_output})
        identity_calls = []
        original_identity = globals()["checkpoint_identity"]
        original_settings = globals()["_settings"]
        original_runtime_files = globals()["runtime_files"]
        original_verify_anchor = globals()["_verify_runtime_anchor"]
        def counting_identity(receipt, cache=None):
            identity_calls.append(receipt)
            return original_identity(receipt, cache)
        def fake_settings(*arguments, **keywords):
            return {"fake": "settings"}
        def fake_runtime_files(*arguments, **keywords):
            return {"fake": "runtime-files"}
        def fake_anchor(*arguments, **keywords):
            return None
        def skip_resume(*arguments, **keywords):
            return "skip"
        globals()["checkpoint_identity"] = counting_identity
        globals()["_settings"] = fake_settings
        globals()["runtime_files"] = fake_runtime_files
        globals()["_verify_runtime_anchor"] = fake_anchor
        globals()["resume_decision"] = skip_resume
        launch_commands.clear()
        subprocess.Popen = recording_popen
        try:
            resumed = run_one(valid_args, model, "infinity", 0,
                              "0-7", "32-39", run_campaign, {})
        finally:
            subprocess.Popen = original_popen
            globals()["checkpoint_identity"] = original_identity
            globals()["_settings"] = original_settings
            globals()["runtime_files"] = original_runtime_files
            globals()["_verify_runtime_anchor"] = original_verify_anchor
            globals()["resume_decision"] = original_resume_decision
        assert len(identity_calls) == 1
        assert resumed == measured_raw
        assert resumed["status"] == "measured"
        assert "UnboundLocalError" not in json.dumps(resumed)
        assert not (valid_output / "previous" / "runs").exists()
        assert all(Path(str(command[0])).name != "taskset"
                   for command in launch_commands if command)

        preflight_output = root / "valid-preflight-failure"
        save_json(preflight_output / "checkpoints" / "org--model.json", {
            "id": model["id"], "revision": model["revision"],
            "status": "ready", "path": str(valid_checkpoint)})
        preflight_args = argparse.Namespace(
            **{**vars(run_one_args), "output": preflight_output})
        settings_calls = []
        def flaky_settings(*arguments, **keywords):
            settings_calls.append(arguments)
            if len(settings_calls) == 1:
                raise RuntimeError("settings preflight boom")
            return {"fake": "settings"}
        globals()["_settings"] = flaky_settings
        globals()["runtime_files"] = fake_runtime_files
        globals()["_verify_runtime_anchor"] = fake_anchor
        launch_commands.clear()
        subprocess.Popen = recording_popen
        try:
            preflight_failed = run_one(preflight_args, model, "infinity", 0,
                                       "0-7", "32-39", run_campaign, {})
        finally:
            subprocess.Popen = original_popen
            globals()["_settings"] = original_settings
            globals()["runtime_files"] = original_runtime_files
            globals()["_verify_runtime_anchor"] = original_verify_anchor
        assert preflight_failed["status"] == "load_failed"
        assert "settings preflight boom" in preflight_failed["error"]
        assert "UnboundLocalError" not in preflight_failed["error"]
        assert all(Path(str(command[0])).name != "taskset"
                   for command in launch_commands if command)

        original_probe = globals()["_run_probe"]
        probe_calls = []
        def counting_preflight_probe(command, label, **keywords):
            probe_calls.append(label)
            return original_probe(command, label, **keywords)
        globals()["_run_probe"] = counting_preflight_probe
        try:
            shared_preflight = {}
            _preflight_runtimes(run_args, run_campaign,
                                ["infinity", "vllm"], shared_preflight)
            first_invocation = len(probe_calls)
            assert first_invocation
            _preflight_runtimes(run_args, run_campaign,
                                ["infinity", "vllm"], shared_preflight)
            assert len(probe_calls) == first_invocation
            _preflight_runtimes(run_args, run_campaign,
                                ["infinity", "vllm"], {})
            assert len(probe_calls) == first_invocation * 2
        finally:
            globals()["_run_probe"] = original_probe

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
        anchor_campaign = {
            "fingerprint_files": {"infinity": files},
            "method": {"runtime_python_identity_sha256": {
                "infinity": inf_record["python_identity_sha256"]}}}
        _verify_runtime_anchor(env_args, anchor_campaign, "infinity", files, cache)
        _expect(CampaignError, _verify_runtime_anchor, env_args, {},
                "infinity", files, cache)
        tampered = {**files, "binary_sha256": "0" * 64}
        _expect(CampaignError, _verify_runtime_anchor, env_args, {
            "fingerprint_files": {"infinity": tampered},
            "method": anchor_campaign["method"]},
            "infinity", files, cache)
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
        globals()["INTEGRITY_HELPER"] = original_integrity_helper
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
