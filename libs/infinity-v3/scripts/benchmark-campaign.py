#!/usr/bin/env python3
"""Create a provenance-rich campaign.json for the embedding benchmark.

Records the clean source checkout, provided Infinity binary, comparator package
integrity, actual GPU/CPU placement and frozen method defaults. It never starts
a server, loads a model or modifies an installed runtime.
"""
from __future__ import annotations
import argparse, ast, base64, datetime, errno, hashlib, importlib.metadata, importlib.util
import json, os, re, shlex, subprocess, sys, tempfile, tomllib, types
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

EXCLUDED = {"docs", "benchmarks", "scripts", "assets", "licenses"}
RESERVED = {"schema_version", "id", "date_local", "timezone", "created_utc", "source_commit",
            "source_clean", "binary_sha256", "build", "runtime_versions", "runtime_installations",
            "hardware", "method", "manifest_sha256", "fingerprint_files"}
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+_-]+)?)\b")
SOURCE_BUILD_NOTE = ("The provided Infinity binary SHA-256 and --version were recorded. The source "
                     "checkout was checked clean for native V3 build inputs, but this tool does not "
                     "assert that the checkout reproduces or produced the provided executable.")
SOURCE_SCOPE = ("Native V3 build inputs only: Cargo.toml/Cargo.lock/.cargo/rust-toolchain.toml, "
                "workspace member code and build.rs/src; docs, scripts/benchmark additions, "
                "assets and licenses are excluded.")
PROVENANCE_NOTE = ("command, providers, libraries and declared_extra come from operator-supplied "
                   "--extra-metadata; they are declarations, not independently verified proof.")
class CampaignError(ValueError): pass

def sha256_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()

def runtime_identity():
    executable = os.path.abspath(sys.executable)
    prefix = os.path.abspath(sys.prefix)
    try:
        interpreter = sha256_file(Path(sys.executable))
    except OSError as exc:
        raise CampaignError(f"cannot hash runtime Python {sys.executable}: {exc}") from exc
    identity = {"executable": executable, "prefix": prefix,
                "interpreter_sha256": interpreter}
    identity["sha256"] = canonical_sha256(identity)
    return identity

def load_common():
    path = Path(__file__).resolve().parent / "benchmark-common.py"
    spec = importlib.util.spec_from_file_location("_benchmark_common_campaign", path)
    if spec is None or spec.loader is None:
        raise CampaignError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise CampaignError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout

def build_inputs(native):
    try:
        document = tomllib.loads((native / "Cargo.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise CampaignError(f"cannot read Cargo.toml: {exc}") from exc
    members = document.get("workspace", {}).get("members", [])
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        raise CampaignError("Cargo.toml workspace.members must be a list of paths")
    names = set(members) | {
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "build.rs", ".cargo", "src",
        "backends", "core", "config", "router"}
    root, inputs = native.resolve(), []
    for name in sorted(names):
        path = (native / name).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise CampaignError(f"build input {name!r} escapes {native}") from exc
        if relative.split("/", 1)[0].lower() not in EXCLUDED:
            inputs.append(relative)
    inputs = [item for item in inputs
              if not any(item != other and item.startswith(other + "/") for other in inputs)]
    if not inputs:
        raise CampaignError(f"no build inputs found under {native}")
    return inputs

def source_snapshot(source_dir):
    source = Path(source_dir).resolve()
    native = next((path for path in (source / "libs" / "infinity-v3", source)
                   if (path / "Cargo.toml").is_file()), None)
    if native is None:
        raise CampaignError(f"{source} is neither a repository root nor a native V3 root")
    commit = git(native, "rev-parse", "HEAD").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise CampaignError(f"git HEAD is not a full 40-hex commit: {commit!r}")
    inputs = build_inputs(native)
    parts = git(native, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                "--", *inputs).split("\0")
    entries, index = [], 0
    while index < len(parts):
        item = parts[index]
        index += 1
        if not item:
            continue
        if len(item) < 4:
            raise CampaignError("unexpected git status output")
        entries.append((item[:2], item[3:]))
        index += "R" in item[:2] or "C" in item[:2]
    if entries:
        shown = ", ".join(f"{status.strip()}:{path}" for status, path in entries[:10])
        raise CampaignError(f"dirty build inputs ({shown}); supply a clean checkout")
    return {"native": native, "commit": commit, "clean": True, "inputs": inputs}

def infinity_binary(infinity_bin, torch_lib):
    binary = Path(infinity_bin).resolve()
    if not binary.is_file():
        raise CampaignError(f"--infinity-bin is not a file: {infinity_bin}")
    libraries = {path.name: sha256_file(path) for path in sorted(binary.parent.glob("*.so"))}
    torch_tree = None
    if torch_lib is not None:
        if not Path(torch_lib).is_dir():
            raise CampaignError(f"--torch-lib is not a directory: {torch_lib}")
        common = load_common()
        try:
            torch_tree = common.library_tree_sha256(torch_lib)
        except common.CampaignError as exc:
            raise CampaignError(str(exc)) from exc
    injected = [str(binary.parent)]
    if torch_lib is not None:
        injected.append(str(Path(torch_lib).resolve()))
    environment = os.environ.copy()
    inherited = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(injected + ([inherited] if inherited else []))
    try:
        result = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                                env=environment, timeout=60, check=False)
    except OSError as exc:
        raise CampaignError(f"cannot run Infinity --version: {exc}") from exc
    output = (result.stdout + result.stderr).strip()
    lines = output.splitlines()
    if result.returncode:
        raise CampaignError(f"Infinity --version failed ({result.returncode}): {output[:500]}")
    if not lines:
        raise CampaignError("Infinity --version produced no output")
    match = VERSION_RE.search(lines[-1])
    digest = sha256_file(binary)
    return {"binary_sha256": digest, "native_library_sha256": libraries,
            "version": match.group(1) if match else lines[-1].strip(), "version_output": output,
            "package_tree_sha256": canonical_sha256(
                {"binary_sha256": digest, "native_libraries": libraries}),
            "torch_library_tree_sha256": torch_tree}

def torch_version_from_lib(torch_lib):
    if torch_lib is None:
        return None
    site = Path(torch_lib).resolve().parent.parent
    try:
        return next((dist.version for dist in importlib.metadata.distributions(path=[str(site)])
                     if (dist.metadata.get("Name") or "").lower() == "torch"), None)
    except OSError:
        return None

def sdk_torch_version(torch_lib):
    if torch_lib is None:
        return None, "not_provided"
    version = torch_version_from_lib(torch_lib)
    if version is not None:
        return version, "python_wheel_metadata"
    root = Path(torch_lib).resolve()
    candidates = (root.parent / "share/cmake/Torch/TorchConfigVersion.cmake",
                  root / "share/cmake/Torch/TorchConfigVersion.cmake")
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r'^\s*set\s*\(\s*PACKAGE_VERSION\s+"?([^"\s)]+)"?\s*\)',
                          text, re.MULTILINE)
        if match is not None:
            return match.group(1), "sdk_cmake"
    return None, "unknown"

def cpu_set(text):
    result = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if match is None:
            raise CampaignError(f"invalid CPU range {token!r}")
        start, end = int(match.group(1)), int(match.group(2) or match.group(1))
        if start > end:
            raise CampaignError(f"invalid descending CPU range {token!r}")
        result.update(range(start, end + 1))
    return result

def physical_core_map():
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    mapping, current = {}, {}
    for line in lines + [""]:
        if not line.strip():
            if {"processor", "physical id", "core id"} <= set(current):
                try:
                    mapping[int(current["processor"])] = (
                        current["physical id"], current["core id"])
                except ValueError:
                    return None
            current = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip()
    return mapping or None

def physical_cores(cpus, mapping):
    if mapping is None:
        return None
    try:
        return len({mapping[cpu] for cpu in cpus})
    except KeyError:
        return None

def cpu_topology():
    fields = {}
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True,
                                env={**os.environ, "LC_ALL": "C"}, check=False)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
    except OSError:
        pass
    def number(name):
        try:
            return int(fields[name])
        except (KeyError, ValueError):
            return None
    model = fields.get("Model name")
    sockets, per_socket = number("Socket(s)"), number("Core(s) per socket")
    logical, physical = number("CPU(s)"), None
    if sockets and per_socket:
        physical = sockets * per_socket
    online = cpu_set(fields["On-line CPU(s) list"]) if "On-line CPU(s) list" in fields else None
    if model is None or logical is None or physical is None:
        try:
            lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        ids = [line.split(":", 1)[1].strip() for line in lines if line.startswith("physical id")]
        cores = [line.split(":", 1)[1].strip() for line in lines if line.startswith("core id")]
        model = model or next((line.split(":", 1)[1].strip() for line in lines
                               if line.startswith("model name")), None)
        logical = logical or sum(line.startswith("processor") for line in lines) or os.cpu_count()
        sockets = sockets or len(set(ids)) or None
        physical = physical or len(set(zip(ids, cores))) or None
    if online is None:
        try:
            online = cpu_set(Path("/sys/devices/system/cpu/online").read_text(encoding="utf-8"))
        except (OSError, CampaignError):
            online = None
    return {"cpu": model, "cpu_sockets": sockets, "physical_cpu_cores": physical,
            "logical_cpus": logical, "online": online}

def cgroup_memory_gib():
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(name).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            continue
        if value < 1 << 62:
            return value // (1 << 30) if value % (1 << 30) == 0 else round(value / (1 << 30), 3)
    return None

def query_gpus():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True,
                                text=True, check=False, timeout=60)
    except OSError as exc:
        raise CampaignError(f"cannot query GPUs with nvidia-smi: {exc}") from exc
    if result.returncode:
        raise CampaignError(f"nvidia-smi failed: {result.stderr.strip() or result.stdout.strip()}")
    devices = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3:
            try:
                index, memory = int(parts[0]), int(parts[2])
            except ValueError:
                continue
            devices[index] = {"index": index, "name": parts[1], "memory_mib": memory}
    if not devices:
        raise CampaignError("nvidia-smi returned no GPU devices")
    return devices

def validate_cpu_lanes(lanes, online=None):
    owners = {}
    for lane in lanes:
        for key in ("server_cpus", "client_cpus"):
            cpus = cpu_set(lane[key])
            if online is not None and not cpus <= online:
                raise CampaignError(f"{key} {lane[key]!r} includes offline CPUs")
            for cpu in cpus:
                if cpu in owners:
                    raise CampaignError(f"CPU {cpu} is shared by {owners[cpu]!r} and {lane[key]!r}")
                owners[cpu] = lane[key]

def hardware(gpus_argument, server_argument, client_argument, common):
    devices = query_gpus()
    try:
        gpus = common.parse_gpus(gpus_argument)
        server = common.parse_lanes(server_argument, "--server-cpus")
        client = common.parse_lanes(client_argument, "--client-cpus")
    except common.CampaignError as exc:
        raise CampaignError(str(exc)) from exc
    if len(server) != len(client):
        raise CampaignError("server and client CPU lane counts must match")
    missing = [gpu for gpu in gpus if gpu not in devices]
    if missing:
        raise CampaignError(f"selected GPUs are not present in nvidia-smi: {missing}")
    lanes = []
    for index, gpu in enumerate(gpus):
        try:
            server_cpus, client_cpus = common.lane_placement(index, gpu, gpus, server, client)
        except common.CampaignError as exc:
            raise CampaignError(str(exc)) from exc
        lanes.append({"gpu": gpu, "server_cpus": server_cpus, "client_cpus": client_cpus})
    topology = cpu_topology()
    validate_cpu_lanes(lanes, topology.pop("online"))
    core_map = physical_core_map()
    lane_cores = []
    for lane in lanes:
        server_cpus = cpu_set(lane["server_cpus"])
        client_cpus = cpu_set(lane["client_cpus"])
        lane_cores.append({
            "gpu": lane["gpu"], "server_cpus": lane["server_cpus"],
            "server_cores": len(server_cpus), "server_logical_cpus": len(server_cpus),
            "server_physical_cores": physical_cores(server_cpus, core_map),
            "client_cpus": lane["client_cpus"],
            "client_cores": len(client_cpus), "client_logical_cpus": len(client_cpus),
            "client_physical_cores": physical_cores(client_cpus, core_map)})
    for key in ("server", "client"):
        logical = {item[f"{key}_logical_cpus"] for item in lane_cores}
        physical = {item[f"{key}_physical_cores"] for item in lane_cores}
        logical_value = logical.pop() if len(logical) == 1 else None
        topology[f"{key}_cores_per_lane"] = logical_value
        topology[f"{key}_logical_cpus_per_lane"] = logical_value
        topology[f"{key}_physical_cores_per_lane"] = (
            physical.pop() if len(physical) == 1 else None)
    topology["cpu_allocation_unit"] = "logical CPU"
    topology["cores_per_lane_unit"] = "logical CPU"
    topology["physical_cores_per_lane_unit"] = "physical core"
    topology["physical_core_map_available"] = core_map is not None
    topology["lane_cores"] = lane_cores
    selected = [devices[lane["gpu"]] for lane in lanes]
    names = {device["name"] for device in selected}
    memories = {device["memory_mib"] for device in selected}
    return {"gpu_name": next(iter(names)) if len(names) == 1 else "mixed",
            "gpu_memory_mib": next(iter(memories)) if len(memories) == 1 else None,
            "gpu_devices": selected, "gpu_lanes": lanes, **topology,
            "cgroup_memory_cap_gib": cgroup_memory_gib(), "shared_host": True}, lanes

def load_extra(text):
    if text is None:
        return {}
    source = text.strip()
    if source.startswith("@"):
        try:
            source = Path(source[1:]).read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignError(f"cannot read --extra-metadata file: {exc}") from exc
    try:
        document = json.loads(source)
    except ValueError as exc:
        raise CampaignError(f"--extra-metadata must be a JSON object: {exc}") from exc
    if not isinstance(document, dict):
        raise CampaignError("--extra-metadata must be a JSON object")
    reserved = sorted(RESERVED & set(document))
    if reserved:
        raise CampaignError(f"--extra-metadata may not override critical keys: {', '.join(reserved)}")
    return document

def declared(extra):
    command = extra.get("command", extra.get("build_command"))
    providers, libraries = extra.get("providers", {}), extra.get("libraries", {})
    if command is not None and not isinstance(command, str):
        raise CampaignError("--extra-metadata command/build_command must be a string")
    if not isinstance(providers, dict) or not isinstance(libraries, dict):
        raise CampaignError("--extra-metadata providers/libraries must be objects")
    remainder = {key: value for key, value in extra.items() if key not in {
        "command", "build_command", "providers", "libraries", "interference_note"}}
    return {"command": command, "providers": providers, "libraries": libraries, "extra": remainder}

def package_files(root, record):
    files, mismatches, unrecorded = [], [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(root.parent).as_posix()
        digest = hashlib.sha256(path.read_bytes()).digest()
        entry = record.get(relative)
        expected = entry.hash.value if entry and entry.hash and entry.hash.mode == "sha256" else None
        actual = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        if expected is None:
            unrecorded.append(relative)
        elif expected != actual:
            mismatches.append(relative)
        files.append({"path": relative, "sha256": digest.hex(),
                      "wheel_record_matches": expected == actual if expected else None})
    return files, mismatches, unrecorded

def metadata_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

def missing_record_files(root, record, module):
    prefix, missing = module + "/", []
    for relative in record:
        parts = Path(relative).parts
        if (relative.startswith(prefix) and not relative.endswith("/")
                and "__pycache__" not in parts and not relative.endswith(".pyc")
                and ".." not in parts and not (root / relative[len(prefix):]).is_file()):
            missing.append(relative)
    return sorted(missing)

def inspect_package(module):
    if MODULE_RE.fullmatch(module) is None:
        raise CampaignError(f"invalid module name {module!r}")
    distribution_name = module.split(".", 1)[0]
    distribution = importlib.metadata.distribution(distribution_name)
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise CampaignError(f"module {module!r} is not importable")
    locations = list(spec.submodule_search_locations or [])
    if locations:
        root = Path(locations[0]).resolve()
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        root = Path(spec.origin).resolve().parent
    else:
        raise CampaignError(f"module {module!r} has no filesystem origin")
    record_text = distribution.read_text("RECORD")
    if record_text is None:
        raise CampaignError(f"distribution {distribution_name!r} has no RECORD")
    record = {str(entry): entry for entry in distribution.files or []}
    files, mismatches, unrecorded = package_files(root, record)
    missing = missing_record_files(root, record, distribution_name)
    commit = None
    version_file = root / "_version.py"
    if version_file.is_file():
        try:
            tree = ast.parse(version_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            tree = None
        for node in tree.body if tree else []:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "__commit_id__"
                    for target in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    value = None
                commit = value if isinstance(value, str) else None
                break
    return {"name": distribution_name, "module": module, "version": distribution.version,
            "commit_id": commit, "record_sha256": hashlib.sha256(record_text.encode()).hexdigest(),
            "package_tree_sha256": canonical_sha256(files), "file_count": len(files),
            "record_mismatches": sorted(mismatches), "unrecorded_files": sorted(unrecorded),
            "missing_record_files": missing, "torch_version": metadata_version("torch"),
            "transformers_version": metadata_version("transformers")}

def private_main(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect-package")
    group.add_argument("--inspect-versions", action="store_true")
    arguments, unknown = parser.parse_known_args(argv)
    if unknown:
        print(json.dumps({"error": f"unexpected arguments: {unknown}"}))
        return 2
    try:
        payload = ({"torch_version": metadata_version("torch"),
                    "transformers_version": metadata_version("transformers")}
                   if arguments.inspect_versions else inspect_package(arguments.inspect_package))
        payload["python_identity"] = runtime_identity()
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0

def run_private(python, *arguments):
    result = subprocess.run([str(python), str(Path(__file__).resolve()), *arguments],
                            capture_output=True, text=True, check=False, timeout=600)
    if result.returncode:
        raise CampaignError(
            f"runtime probe failed ({python}): {(result.stdout or result.stderr).strip()[:500]}")
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise CampaignError(f"runtime probe returned invalid JSON ({python}): {exc}") from exc
    if "error" in payload:
        raise CampaignError(f"runtime probe failed ({python}): {payload['error']}")
    return payload

def probe_python_identity(payload, label):
    identity = payload.get("python_identity")
    if not isinstance(identity, dict):
        raise CampaignError(f"{label} probe returned no Python identity")
    base = {key: identity.get(key)
            for key in ("executable", "prefix", "interpreter_sha256")}
    if not all(isinstance(value, str) and value for value in base.values()) \
            or not os.path.isabs(base["executable"]) or not os.path.isabs(base["prefix"]) \
            or not re.fullmatch(r"[0-9a-f]{64}", base["interpreter_sha256"]):
        raise CampaignError(f"{label} probe returned an invalid Python identity")
    if identity.get("sha256") != canonical_sha256(base):
        raise CampaignError(f"{label} probe returned an inconsistent Python identity")
    return identity

def installation(payload, label, allow_modified):
    problems = ((payload.get("record_mismatches") or []) + (payload.get("missing_record_files") or [])
                + (payload.get("unrecorded_files") or []))
    if problems and not allow_modified:
        raise CampaignError(
            f"{label} package does not match its wheel RECORD ({len(problems)} difference(s)); "
            "use --allow-modified-runtimes to record it as modified")
    return {"version": payload["version"], "commit_id": payload.get("commit_id"),
            "package_tree_sha256": payload["package_tree_sha256"],
            "file_count": payload["file_count"], "record_sha256": payload["record_sha256"],
            "record_mismatches": payload.get("record_mismatches") or [],
            "unrecorded_files": payload.get("unrecorded_files") or [],
            "missing_record_files": payload.get("missing_record_files") or [],
            "torch_version": payload.get("torch_version"),
            "transformers_version": payload.get("transformers_version"),
            "stock": not problems}

def runtime_version_metadata(vllm_probe, sglang_probe, client_versions,
                             infinity_version, sdk_version, sdk_source):
    return {"infinity": infinity_version, "vllm": vllm_probe["version"],
            "sglang": sglang_probe["version"],
            "infinity_torch": sdk_version, "infinity_torch_source": sdk_source,
            "vllm_torch": vllm_probe.get("torch_version"),
            "vllm_transformers": vllm_probe.get("transformers_version"),
            "sglang_torch": sglang_probe.get("torch_version"),
            "sglang_transformers": sglang_probe.get("transformers_version"),
            "client_torch": client_versions.get("torch_version"),
            "client_transformers": client_versions.get("transformers_version")}

def write_campaign(path, document, overwrite):
    path = Path(path)
    payload = json.dumps(document, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise CampaignError(f"{path} already exists; pass --overwrite") from exc
            except OSError as exc:
                if exc.errno in {errno.EPERM, errno.EOPNOTSUPP, errno.EXDEV}:
                    raise CampaignError(
                        f"atomic no-overwrite publication is unsupported for {path}; "
                        "no campaign was written; pass --overwrite to replace explicitly") from exc
                raise CampaignError(f"cannot publish {path} atomically: {exc}") from exc
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

def default_id(gpu_name, date_local):
    slug = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower().replace("nvidia", "")).strip("-")
    return f"{slug or 'gpu'}-{date_local.replace('-', '')}-campaign"

def campaign_parser(common):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="absolute output directory")
    parser.add_argument("--source-dir", type=Path, help="repository root or native V3 root")
    parser.add_argument("--infinity-bin", type=Path)
    parser.add_argument("--vllm-python", type=Path)
    parser.add_argument("--sglang-python", type=Path)
    parser.add_argument("--client-python", type=Path)
    parser.add_argument("--torch-lib", type=Path, help="Infinity-only torch library directory")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--server-cpus", default=",".join(common.DEFAULT_SERVER_CPUS))
    parser.add_argument("--client-cpus", default=",".join(common.DEFAULT_CLIENT_CPUS))
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--id")
    parser.add_argument("--extra-metadata", metavar="JSON",
                        help="declared build command/provider provenance (JSON or @file)")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-modified-runtimes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser

def create_campaign(args, common):
    source = source_snapshot(args.source_dir)
    target = args.output.resolve() / "campaign.json"
    if target.exists() and not args.overwrite:
        raise CampaignError(f"{target} already exists; pass --overwrite")
    if not (args.infinity_bin and args.vllm_python and args.sglang_python):
        raise CampaignError("--infinity-bin, --vllm-python and --sglang-python are required")
    client_python = args.client_python or args.sglang_python or Path(sys.executable)
    for label, value in (("--vllm-python", args.vllm_python),
                         ("--sglang-python", args.sglang_python),
                         ("--client-python", client_python)):
        if not Path(value).exists():
            raise CampaignError(f"{label} does not exist: {value}")
    manifest = (args.manifest or (common.HERE / "benchmark-common-models.json")).resolve()
    if not manifest.is_file():
        raise CampaignError(f"manifest not found: {manifest}")
    extra = load_extra(args.extra_metadata)
    values = declared(extra)
    vllm_probe = run_private(args.vllm_python, "--inspect-package", "vllm")
    sglang_probe = run_private(args.sglang_python, "--inspect-package", "sglang")
    vllm = installation(vllm_probe, "vLLM", args.allow_modified_runtimes)
    sglang = installation(sglang_probe, "SGLang", args.allow_modified_runtimes)
    infinity = infinity_binary(args.infinity_bin, args.torch_lib)
    client_versions = run_private(client_python, "--inspect-versions")
    client_identity = probe_python_identity(client_versions, "client Python")
    vllm_identity = probe_python_identity(vllm_probe, "vLLM")
    sglang_identity = probe_python_identity(sglang_probe, "SGLang")
    sdk_version, sdk_source = sdk_torch_version(args.torch_lib)
    versions = runtime_version_metadata(vllm_probe, sglang_probe, client_versions,
                                        infinity["version"], sdk_version, sdk_source)
    resolved_hardware, lanes = hardware(args.gpus, args.server_cpus, args.client_cpus, common)
    try:
        timezone = ZoneInfo(args.timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CampaignError(f"unknown --timezone {args.timezone!r}") from exc
    date_local = datetime.datetime.now(timezone).date().isoformat()
    identifier = args.id or default_id(resolved_hardware["gpu_name"], date_local)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identifier) is None:
        raise CampaignError("--id must contain only letters, digits, dot, underscore or dash")
    build = {"version": infinity["version"], "binary_sha256": infinity["binary_sha256"],
             "native_library_sha256": infinity["native_library_sha256"],
             "torch_library_tree_sha256": infinity["torch_library_tree_sha256"],
             "command": values["command"], "providers": values["providers"],
             "libraries": values["libraries"], "source_commit": source["commit"],
             "binary_version_output": infinity["version_output"],
             "source_build_relation": "not_verified", "source_build_note": SOURCE_BUILD_NOTE,
             "source_scope": SOURCE_SCOPE, "build_inputs": source["inputs"],
             "provenance_note": PROVENANCE_NOTE, "declared_extra": values["extra"]}
    installations = {"infinity": {"package_tree_sha256": infinity["package_tree_sha256"],
                                 "source_clean": True, "torch_version": sdk_version,
                                 "torch_version_source": sdk_source},
                     "vllm": vllm, "sglang": sglang}
    campaign = {"schema_version": 1, "id": identifier, "date_local": date_local,
                "timezone": args.timezone,
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source_commit": source["commit"], "source_clean": True,
                "binary_sha256": infinity["binary_sha256"], "build": build,
                "runtime_versions": versions, "runtime_installations": installations,
                "hardware": resolved_hardware,
                "method": {"seconds": 10.0, "repeats": 3, "warmup": 3.0, "concurrency": 32,
                           "long_input_concurrency": 8, "max_batch_tokens": 16384,
                           "max_sequence_length": 2048, "dtype": "bfloat16",
                           "runtimes": list(common.RUNTIMES), "request_batch_size": 1,
                           "encoding": "JSON float", "prefix_cache": False,
                           "radix_cache": False, "infinity_radix_mlp_threshold": 0,
                           "runtime_python_identity_sha256": {
                               "infinity": client_identity["sha256"],
                               "vllm": vllm_identity["sha256"],
                               "sglang": sglang_identity["sha256"]},
                           "lanes": lanes},
                "manifest_sha256": sha256_file(manifest), "fingerprint_files": {},
                "interference_note": extra.get("interference_note", "No interference note supplied.")}
    fingerprint_args = argparse.Namespace(
        infinity_bin=str(args.infinity_bin), vllm_python=str(args.vllm_python),
        sglang_python=str(args.sglang_python), client_python=str(client_python),
        torch_lib=str(args.torch_lib) if args.torch_lib else None)
    cache = {}
    common.seed_package_probe(cache, str(args.vllm_python), "vllm", vllm_probe)
    common.seed_package_probe(cache, str(args.sglang_python), "sglang", sglang_probe)
    common.seed_python_identity(cache, str(client_python), client_identity)
    campaign["fingerprint_files"] = {
        runtime: common.runtime_files(fingerprint_args, runtime, campaign, cache)
        for runtime in common.RUNTIMES}
    return campaign

def expect(error, function, *arguments):
    try:
        function(*arguments)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")

def self_test():
    common = load_common()
    with tempfile.TemporaryDirectory() as temporary:
        root, repo = Path(temporary), Path(temporary) / "repo"
        native = repo / "libs" / "infinity-v3"
        fixture = {
            ".cargo/config.toml": "[build]\n", "Cargo.toml":
            '[workspace]\nmembers = ["core", "backends"]\n', "Cargo.lock": "# lock\n",
            "rust-toolchain.toml": "[toolchain]\n", "core/Cargo.toml": "[package]\nname='core'\n",
            "core/src/lib.rs": "pub fn a() {}\n", "backends/Cargo.toml":
            "[package]\nname='backends'\n", "backends/src/lib.rs": "pub fn b() {}\n"}
        for relative, text in fixture.items():
            path = native / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        def run_git(*arguments):
            subprocess.run(["git", "-C", str(repo), *arguments], check=True,
                           capture_output=True, text=True)
        run_git("init", "-q")
        run_git("add", ".")
        run_git("-c", "user.name=Campaign Test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "initial")
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        for relative in ("docs/new.md", "scripts/new.py", "benchmarks/new.json",
                         "assets/new.bin", "LICENSES/new.txt"):
            path = native / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("new\n", encoding="utf-8")
        assert source_snapshot(repo)["commit"] == commit == source_snapshot(native)["commit"]
        for relative in ("Cargo.lock", "rust-toolchain.toml"):
            (native / relative).unlink()
            expect(CampaignError, source_snapshot, native)
            run_git("checkout", "--", f"libs/infinity-v3/{relative}")
        (native / "core/src/lib.rs").write_text("dirty\n", encoding="utf-8")
        expect(CampaignError, source_snapshot, native)
        run_git("checkout", "--", "libs/infinity-v3/core/src/lib.rs")
        (native / "core/src/new.rs").write_text("untracked\n", encoding="utf-8")
        expect(CampaignError, source_snapshot, native)
        (native / "core/src/new.rs").unlink()
        lanes = [{"gpu": 0, "server_cpus": "0-7", "client_cpus": "8-15"},
                 {"gpu": 1, "server_cpus": "16-23", "client_cpus": "24-31"}]
        validate_cpu_lanes(lanes, set(range(32)))
        expect(CampaignError, cpu_set, "8-4")
        expect(CampaignError, validate_cpu_lanes,
               [{"gpu": 0, "server_cpus": "0-7", "client_cpus": "7-15"}])
        expect(CampaignError, validate_cpu_lanes,
               [{"gpu": 0, "server_cpus": "0-7", "client_cpus": "8-15"},
                {"gpu": 1, "server_cpus": "6-13", "client_cpus": "24-31"}], set(range(32)))
        expect(CampaignError, validate_cpu_lanes, lanes, set(range(16)))
        core_map = {0: ("0", "0"), 1: ("0", "0"), 2: ("0", "1"), 3: ("0", "1")}
        assert physical_cores({0, 1, 2, 3}, core_map) == 2
        assert physical_cores({0, 1}, None) is None
        assert physical_cores({0, 9}, core_map) is None
        clean = {"version": "1", "package_tree_sha256": "a" * 64, "record_sha256": "b" * 64,
                 "file_count": 1, "record_mismatches": [], "unrecorded_files": [],
                 "missing_record_files": []}
        assert installation(clean, "vllm", False)["stock"] is True
        for key in ("record_mismatches", "missing_record_files", "unrecorded_files"):
            broken = {**clean, key: ["vllm/file.py"]}
            expect(CampaignError, installation, broken, "vllm", False)
            assert installation(broken, "vllm", True)["stock"] is False
        vllm_entry = installation({**clean, "version": "0.33.0", "torch_version": "2.14.0",
                                   "transformers_version": "5.20.2"}, "vllm", False)
        sglang_entry = installation({**clean, "version": "0.5.23", "torch_version": "2.14.1",
                                     "transformers_version": "5.19.0"}, "sglang", False)
        assert vllm_entry["transformers_version"] == "5.20.2"
        assert sglang_entry["transformers_version"] == "5.19.0"
        unknown_sdk = root / "unknown-sdk/lib"
        unknown_sdk.mkdir(parents=True)
        assert sdk_torch_version(unknown_sdk) == (None, "unknown")
        versions = runtime_version_metadata(
            {"version": "0.33.0", "torch_version": "2.14.0",
             "transformers_version": "5.20.2"},
            {"version": "0.5.23", "torch_version": "2.14.1",
             "transformers_version": "5.19.0"},
            {"torch_version": "9.9.9", "transformers_version": "5.19.0"},
            "3.0.2-beta.2", *sdk_torch_version(unknown_sdk))
        assert versions["infinity_torch"] is None
        assert versions["infinity_torch_source"] == "unknown"
        assert versions["client_torch"] == "9.9.9"
        assert versions["vllm_transformers"] == "5.20.2"
        assert versions["sglang_transformers"] == "5.19.0"
        sdk = root / "sdk"
        (sdk / "lib").mkdir(parents=True)
        (sdk / "share/cmake/Torch").mkdir(parents=True)
        (sdk / "share/cmake/Torch/TorchConfigVersion.cmake").write_text(
            'set(PACKAGE_VERSION "2.14.1")\n', encoding="utf-8")
        assert sdk_torch_version(sdk / "lib") == ("2.14.1", "sdk_cmake")
        pkg = root / "recordcheck"
        (pkg / "pkg").mkdir(parents=True)
        (pkg / "pkg/a.py").write_text("A = 1\n", encoding="utf-8")
        encoded = base64.urlsafe_b64encode(hashlib.sha256(b"A = 1\n").digest()).decode().rstrip("=")
        entry = types.SimpleNamespace(hash=types.SimpleNamespace(mode="sha256", value=encoded))
        record = {"pkg/a.py": entry}
        assert package_files(pkg / "pkg", record)[1:] == ([], [])
        (pkg / "pkg/a.py").write_text("A = 2\n", encoding="utf-8")
        assert package_files(pkg / "pkg", record)[1] == ["pkg/a.py"]
        (pkg / "pkg/extra.py").write_text("X = 1\n", encoding="utf-8")
        assert package_files(pkg / "pkg", record)[2] == ["pkg/extra.py"]
        assert missing_record_files(pkg / "pkg", {"pkg/gone.py": None}, "pkg") == ["pkg/gone.py"]
        hashes = []
        for name in ("site-a", "site-b"):
            site = root / name
            (site / "pkg/sub").mkdir(parents=True)
            (site / "pkg/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (site / "pkg/sub/module.py").write_text("X = 2\n", encoding="utf-8")
            (site / "pkg/sub/data.bin").write_bytes(b"payload")
            hashes.append(canonical_sha256(package_files(site / "pkg", {})[0]))
        assert hashes[0] == hashes[1]
        target = root / "campaign.json"
        target.write_bytes(b"original")
        expect(CampaignError, write_campaign, target, {"x": 1}, False)
        assert target.read_bytes() == b"original"
        write_campaign(target, {"x": 1}, True)
        assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}
        fresh = root / "fresh/campaign.json"
        write_campaign(fresh, {"y": 2}, False)
        assert json.loads(fresh.read_text(encoding="utf-8")) == {"y": 2}
        original_link = os.link
        def unsupported_link(*arguments, **keywords):
            raise OSError(errno.EOPNOTSUPP, "hard links unsupported")
        os.link = unsupported_link
        try:
            blocked = root / "blocked/campaign.json"
            expect(CampaignError, write_campaign, blocked, {"z": 3}, False)
            assert not blocked.exists()
        finally:
            os.link = original_link
        expect(CampaignError, load_extra, '{"source_commit": "f" * 40}')

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

        def make_wrapper(name, python, pythonpath):
            path = root / "wrappers" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"#!/bin/sh\nPYTHONPATH={shlex.quote(str(pythonpath))} "
                f"exec {shlex.quote(str(python))} \"$@\"\n", encoding="utf-8")
            path.chmod(0o755)
            return path

        def make_package(site, module):
            package = site / module
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            (package / "model.py").write_text("MODEL = 1\n", encoding="utf-8")
            dist_info = site / f"{module}-0.0.1.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.0.1\n\n",
                encoding="utf-8")
            def record_line(relative):
                data = (site / relative).read_bytes()
                encoded = base64.urlsafe_b64encode(
                    hashlib.sha256(data).digest()).decode().rstrip("=")
                return f"{relative},sha256={encoded},{len(data)}\n"
            (dist_info / "RECORD").write_text(
                "".join(record_line(item)
                        for item in (f"{module}/__init__.py", f"{module}/model.py")),
                encoding="utf-8")
            return package

        vllm_site = root / "fake-vllm-site"
        vllm_site.mkdir()
        make_package(vllm_site, "vllm")
        sglang_site = root / "fake-sglang-site"
        sglang_site.mkdir()
        make_package(sglang_site, "sglang")
        vllm_wrapper = make_wrapper(
            "campaign-vllm", make_venv("campaign-vllm-env"), vllm_site)
        sglang_wrapper = make_wrapper(
            "campaign-sglang", make_venv("campaign-sglang-env"), sglang_site)
        infinity_dir = root / "infinity-runtime"
        infinity_dir.mkdir()
        infinity = infinity_dir / "infinity"
        infinity.write_text("#!/bin/sh\necho 'Infinity 3.0.2-beta.3'\n", encoding="utf-8")
        infinity.chmod(0o755)
        for index in range(5):
            (infinity_dir / f"libinfinity_bridge_{index}.so").write_bytes(
                f"bridge-{index}".encode())
        torch_lib = root / "campaign-torch"
        (torch_lib / "nested").mkdir(parents=True)
        (torch_lib / "libtorch.so").write_bytes(b"torch")
        (torch_lib / "nested" / "libc10.so").write_bytes(b"c10")
        fake_lanes = [{"gpu": 0, "server_cpus": "0-7", "client_cpus": "8-15"}]
        fake_hardware = {"gpu_name": "Fake GPU", "gpu_memory_mib": 1,
                         "gpu_devices": [{"index": 0, "name": "Fake GPU",
                                          "memory_mib": 1}],
                         "gpu_lanes": fake_lanes, "cpu": "fixture",
                         "server_cores_per_lane": 8, "client_cores_per_lane": 8,
                         "shared_host": True}
        original_hardware = globals()["hardware"]
        globals()["hardware"] = lambda *arguments, **keywords: (
            fake_hardware, fake_lanes)
        try:
            campaign_args = argparse.Namespace(
                output=root / "campaign-output", source_dir=repo,
                infinity_bin=infinity, vllm_python=vllm_wrapper,
                sglang_python=sglang_wrapper, client_python=Path(sys.executable),
                torch_lib=torch_lib, gpus="0", server_cpus="0-7",
                client_cpus="8-15", timezone="UTC", id="fixture-campaign",
                extra_metadata=None,
                manifest=Path(__file__).resolve().parent / "benchmark-common-models.json",
                allow_modified_runtimes=False, overwrite=False)
            campaign = create_campaign(campaign_args, common)
        finally:
            globals()["hardware"] = original_hardware
        files = campaign["fingerprint_files"]
        expected_keys = {"runner_sha256", "client_sha256", "binary_sha256",
                         "native_library_sha256", "runtime_package_tree_sha256",
                         "torch_library_tree_sha256"}
        assert set(files) == set(common.RUNTIMES)
        assert all(set(files[runtime]) == expected_keys for runtime in common.RUNTIMES)
        assert files["infinity"]["torch_library_tree_sha256"] \
            == common.library_tree_sha256(torch_lib)
        assert campaign["build"]["torch_library_tree_sha256"] \
            == files["infinity"]["torch_library_tree_sha256"]
        assert files["vllm"]["torch_library_tree_sha256"] is None
        assert files["sglang"]["torch_library_tree_sha256"] is None
        for runtime in common.RUNTIMES:
            assert campaign["runtime_installations"][runtime]["package_tree_sha256"] \
                == files[runtime]["runtime_package_tree_sha256"]
        assert files["infinity"]["native_library_sha256"] \
            == files["vllm"]["native_library_sha256"] \
            == files["sglang"]["native_library_sha256"]
        identities = campaign["method"]["runtime_python_identity_sha256"]
        assert set(identities) == set(common.RUNTIMES)
        assert all(re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in identities.values())
        fingerprint_args = argparse.Namespace(
            infinity_bin=str(infinity), vllm_python=str(vllm_wrapper),
            sglang_python=str(sglang_wrapper), client_python=str(sys.executable),
            torch_lib=str(torch_lib))
        for runtime in common.RUNTIMES:
            assert common.runtime_files(fingerprint_args, runtime, campaign, {}) \
                == files[runtime]
            assert common.runtime_python_identity_sha256(
                fingerprint_args, runtime, {}) == identities[runtime]
        (vllm_site / "vllm/model.py").write_text("MODEL = 2\n", encoding="utf-8")
        expect(common.CampaignError, common.runtime_files, fingerprint_args, "vllm",
               campaign, {})
        (vllm_site / "vllm/model.py").write_text("MODEL = 1\n", encoding="utf-8")
        assert common.runtime_files(fingerprint_args, "vllm", campaign, {}) \
            == files["vllm"]
        (vllm_site / "vllm/model.py").unlink()
        expect(common.CampaignError, common.runtime_files, fingerprint_args, "vllm",
               campaign, {})
    print("self-test passed")
    return 0

def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--inspect-package" in arguments or "--inspect-versions" in arguments:
        return private_main(arguments)
    common = load_common()
    args = campaign_parser(common).parse_args(arguments)
    if args.self_test:
        return self_test()
    if args.output is None or not args.output.is_absolute():
        raise SystemExit("--output must be an absolute path")
    if args.source_dir is None:
        raise SystemExit("--source-dir is required")
    args.output = args.output.resolve()
    try:
        campaign = create_campaign(args, common)
        write_campaign(args.output / "campaign.json", campaign, args.overwrite)
    except CampaignError as exc:
        raise SystemExit(str(exc)) from exc
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
