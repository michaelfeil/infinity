#!/usr/bin/env python3
"""Portable embedding-benchmark exporter for the frozen common runner contract."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

RT = ("infinity", "vllm", "sglang")
RTS = frozenset(RT)
TERMINAL = frozenset({"measured", "client_failed", "load_failed", "checkpoint_unavailable"})
FIXTURE_SOURCE_COMMIT = "53dd6f2cb3083e72b65fbe2406913bc09ab75952"
SECONDS, WARMUP, WINDOWS, TOL = 10.0, 3.0, 3, 0.25
COS, RTOL, ATOL = 0.999, 0.02, 0.01
DTYPE = "bfloat16"
SEM = {"embeddings": ("text",), "rerank": ("query", "document")}
FILE_KEYS = {"runner_sha256", "client_sha256", "binary_sha256",
             "native_library_sha256", "runtime_package_tree_sha256"}
SET_KEYS = {"dtype", "seconds", "repeats", "warmup", "concurrency",
            "long_input_concurrency", "max_batch_tokens", "max_sequence_length",
            "lengths", "context", "input_context", "runtime_environment"}
REQUIRED_ENV = {
    "infinity": {"INFINITY_TORCH_CUDA_GRAPHS": "1",
                 "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"},
    "sglang": {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"},
    "vllm": {},
}
ARGS = {"url", "runtime", "model", "tokenizer", "task", "output", "lengths",
        "concurrency", "seconds", "repeats", "warmup", "min_latency_length",
        "self_test", "batch_size", "dtype"}
HEX64 = re.compile(r"[0-9a-f]{64}")
REV = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
CPU_DESC = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
URL = re.compile(r"https?://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?[^\s\"'`]*")
ABS = re.compile(r"(?<![A-Za-z0-9._-])/(?:root|home|tmp|opt|var|usr|etc|srv|mnt|data|workspace|scratch|private)(?:/[^\s\"'`,;:)]*)?")

class ReportError(Exception):
    pass

def require(ok, msg):
    if not ok:
        raise ReportError(msg)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def shafile(path):
    with Path(path).open("rb") as h:
        return hashlib.file_digest(h, "sha256").hexdigest()

def cjson(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)

def fp(v):
    return sha(cjson(v).encode())

def gz(data):
    b = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=b, compresslevel=9, mtime=0) as h:
        h.write(data)
    return b.getvalue()

def clean(v):
    if isinstance(v, str):
        return ABS.sub("<abs-path>", URL.sub("<local-url>", v))
    if isinstance(v, dict):
        return {clean(str(k)): clean(x) for k, x in sorted(v.items(), key=lambda p: str(p[0]))}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, bool) or v is None or isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    return clean(str(v))

def bounded(v):
    if not isinstance(v, str):
        return None
    for line in v.splitlines():
        if line.strip():
            return clean(line.strip()[:200])
    return None

def portable(v):
    if isinstance(v, dict):
        return {k: portable(x) for k, x in sorted(v.items())
                if k not in {"path", "cached_path", "cache_path", "local_dir", "local_path"}}
    if isinstance(v, list):
        return [portable(x) for x in v]
    return Path(v).name if isinstance(v, str) and v.startswith("/") else v

def num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v)) else None

def pint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0

def h64(v):
    return isinstance(v, str) and HEX64.fullmatch(v) is not None

def revok(v):
    return isinstance(v, str) and REV.fullmatch(v) is not None

def dtype(v):
    require(isinstance(v, str), f"dtype {v!r} is invalid")
    x = v.lower().replace("-", "").replace("_", "")
    require(x in {"bf16", "bfloat16"}, f"unsupported dtype {v!r}")
    return DTYPE

def place(v):
    require(not isinstance(v, bool) and v is not None, "placement is missing")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str) and v:
        return v
    if isinstance(v, list) and v:
        return ",".join(str(x) for x in v)
    raise ReportError("placement is invalid")

def norm(v):
    return math.sqrt(sum(x * x for x in v))

def cos(a, b):
    if not a or len(a) != len(b) or norm(a) == 0 or norm(b) == 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))

def allclose(a, b):
    return len(a) == len(b) and all(abs(x - y) <= ATOL + RTOL * abs(x) for x, y in zip(a, b))

def normclose(a, b):
    return abs(norm(a) - norm(b)) <= ATOL + RTOL * abs(norm(a))

def vshape(v):
    if isinstance(v, bool):
        return None, "boolean"
    if isinstance(v, (int, float)):
        return (None, "nonfinite") if not math.isfinite(float(v)) else ([], None)
    if not isinstance(v, list):
        return None, "nonscalar"
    if not v:
        return [0], None
    ss = [vshape(x) for x in v]
    for s, e in ss:
        if e:
            return None, e
    if any(s != ss[0][0] for s, _ in ss[1:]):
        return None, "ragged"
    return [len(v)] + ss[0][0], None

def flat(v):
    out, stack = [], [v]
    while stack:
        x = stack.pop()
        if isinstance(x, bool):
            return None, "boolean"
        if isinstance(x, (int, float)):
            y = float(x)
            if not math.isfinite(y):
                return None, "nonfinite"
            out.append(y)
        elif isinstance(x, list):
            stack.extend(reversed(x))
        else:
            return None, "nonscalar"
    return out, None

def probe(task, rec):
    if not isinstance(rec, dict):
        return None, "missing_probe_record"
    shape, vals = rec.get("shape"), rec.get("values")
    if not (isinstance(shape, list) and shape and all(pint(x) for x in shape)):
        return None, "invalid_shape"
    if task == "embeddings":
        if len(shape) != 1:
            return None, "dense_matrix_shape"
    elif shape != [1]:
        return None, "rerank_shape"
    if vals is None:
        return None, "missing_probe_values"
    if rec.get("all_finite") is not True:
        return None, "all_finite_not_true"
    actual, err = vshape(vals)
    if err:
        return None, err
    if actual != shape:
        return None, "shape_mismatch"
    values, err = flat(vals)
    if err:
        return None, err
    if not values:
        return None, "empty_vector"
    n = norm(values)
    if task == "embeddings" and n == 0:
        return None, "zero_norm"
    rn = rec.get("norm")
    if rn is not None:
        x = num(rn)
        if x is None:
            return None, "invalid_norm"
        if abs(x - n) > ATOL + RTOL * abs(n):
            return None, "norm_mismatch"
    return {"shape": shape, "values": vals, "flat": values, "norm": n}, None

def fields(task, sample, label):
    require(isinstance(sample, dict), f"{label} is not an object")
    out = {}
    for k in SEM[task]:
        v = sample.get(k)
        require(isinstance(v, str) and v, f"{label} {k} is missing")
        out[k] = v
    return out

def semantic_hash(task, sample, label):
    f = fields(task, sample, label)
    h = sha(json.dumps(f, sort_keys=True).encode())
    require(isinstance(sample.get("sha256"), str) and sample["sha256"] == h,
        f"{label} sample.sha256 mismatch")
    return {"sha256": h, "fields": f, "provided_sha256": sample["sha256"]}

def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, UnicodeError, ValueError) as e:
        raise ReportError(f"{path} invalid JSON: {str(e).splitlines()[0][:300]}")

def anchor(v, top_binary):
    require(isinstance(v, dict) and set(v) == RTS,
        "campaign.fingerprint_files is missing or must contain exactly infinity/vllm/sglang")
    out = {}
    for r in RT:
        e = v[r]
        require(isinstance(e, dict) and set(e) == FILE_KEYS,
            f"campaign.fingerprint_files[{r}] must have exactly {sorted(FILE_KEYS)}")
        for k in ("runner_sha256", "client_sha256", "binary_sha256"):
            require(h64(e[k]), f"fingerprint_files[{r}].{k} invalid")
        lib = e["native_library_sha256"]
        require(isinstance(lib, dict), f"fingerprint_files[{r}].native_library_sha256 invalid")
        for n, h in lib.items():
            require(isinstance(n, str) and n and "/" not in n and "\\" not in n and h64(h),
                f"fingerprint_files[{r}] library hash invalid")
        tree = e["runtime_package_tree_sha256"]
        require(tree is None or h64(tree), f"fingerprint_files[{r}].runtime_package_tree_sha256 invalid")
        if r in ("vllm", "sglang"):
            require(h64(tree), f"fingerprint_files[{r}].runtime_package_tree_sha256 must be 64-hex")
        out[r] = {k: e[k] for k in FILE_KEYS}
    require(out["infinity"]["binary_sha256"] == top_binary,
        "campaign.fingerprint_files.infinity.binary_sha256 must equal campaign.binary_sha256")
    require(len({out[r]["runner_sha256"] for r in RT}) == 1,
        "runner_sha256 must match across runtime anchors")
    require(len({out[r]["client_sha256"] for r in RT}) == 1,
        "client_sha256 must match across runtime anchors")
    require(len({cjson(out[r]["native_library_sha256"]) for r in RT}) == 1,
        "native library hashes must match across runtime anchors")
    return out

def load_campaign(root):
    d = load(Path(root) / "campaign.json")
    require(isinstance(d, dict), "campaign.json must be an object")
    require(d.get("source_clean") is True, "campaign.source_clean must be true")
    commit = d.get("source_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{40}", commit),
        "campaign.source_commit must be a full 40-hex commit")
    require(h64(d.get("binary_sha256")), "campaign.binary_sha256 invalid")
    require(h64(d.get("manifest_sha256")), "campaign.manifest_sha256 pin invalid")
    mpath = Path(root) / "manifest.json"
    require(shafile(mpath) == d["manifest_sha256"], "campaign.manifest_sha256 does not match manifest bytes")
    versions = d.get("runtime_versions")
    require(isinstance(versions, dict), "campaign.runtime_versions missing")
    vers = {}
    for r in RT:
        v = versions.get(r)
        if isinstance(v, dict):
            v = v.get("version") or v.get("commit") or v.get("sha256")
        require(isinstance(v, str) and v, f"runtime version missing for {r}")
        vers[r] = v
    hw = d.get("hardware")
    require(isinstance(hw, dict), "campaign.hardware missing")
    meth = d.get("method")
    require(isinstance(meth, dict), "campaign.method missing")
    require(set(meth.get("runtimes") or []) == RTS and len(meth.get("runtimes") or []) == 3,
        "campaign.method.runtimes must be exactly infinity/vllm/sglang")
    for k, want in (("seconds", SECONDS), ("warmup", WARMUP), ("repeats", WINDOWS)):
        require(num(meth.get(k)) == want, f"campaign.method.{k} must be {want}")
    require(meth.get("concurrency") == 32, "campaign.method.concurrency must be 32")
    require(meth.get("long_input_concurrency") == 8, "campaign.method.long_input_concurrency must be 8")
    require(meth.get("max_batch_tokens") == 16384, "campaign.method.max_batch_tokens must be 16384")
    require(meth.get("max_sequence_length") == 2048, "campaign.method.max_sequence_length must be 2048")
    require(meth.get("request_batch_size") == 1, "campaign.method.request_batch_size must be 1")
    require(dtype(meth.get("dtype")) == DTYPE, "campaign.method.dtype must be bfloat16")
    if meth.get("prefix_cache") is not None:
        require(meth["prefix_cache"] is False, "campaign.method.prefix_cache must be false")
    if meth.get("radix_cache") is not None:
        require(meth["radix_cache"] is False, "campaign.method.radix_cache must be false")
    if meth.get("infinity_radix_mlp_threshold") is not None:
        require(meth["infinity_radix_mlp_threshold"] == 0, "Infinity radix MLP threshold must be 0")
    lanes = meth.get("lanes")
    require(isinstance(lanes, list) and lanes, "campaign.method.lanes must be a non-empty list")
    lane_keys, gpus, server_sets, client_sets = set(), set(), set(), set()
    for lane in lanes:
        require(isinstance(lane, dict)
                and set(lane) == {"gpu", "server_cpus", "client_cpus"},
                "campaign.method.lanes entry schema invalid")
        gpu = lane["gpu"]
        require(isinstance(gpu, int) and not isinstance(gpu, bool) and gpu >= 0,
                "campaign.method.lanes gpu invalid")
        for key in ("server_cpus", "client_cpus"):
            require(isinstance(lane[key], str) and CPU_DESC.fullmatch(lane[key]),
                    f"campaign.method.lanes {key} invalid")
        require(str(gpu) not in gpus, "campaign.method.lanes gpu values must be unique")
        require(lane["server_cpus"] not in server_sets,
                "campaign.method.lanes server CPU descriptors must be unique")
        require(lane["client_cpus"] not in client_sets,
                "campaign.method.lanes client CPU descriptors must be unique")
        gpus.add(str(gpu)); server_sets.add(lane["server_cpus"]); client_sets.add(lane["client_cpus"])
        lane_keys.add((str(gpu), lane["server_cpus"], lane["client_cpus"]))
    if hw.get("gpu_lanes") is not None:
        require(cjson(hw["gpu_lanes"]) == cjson(lanes),
                "campaign.hardware.gpu_lanes must match campaign.method.lanes")
    files = anchor(d.get("fingerprint_files"), d["binary_sha256"])
    build = d.get("build") or {}
    if build.get("binary_sha256") is not None:
        require(build["binary_sha256"] == files["infinity"]["binary_sha256"],
            "build binary_sha256 does not match Infinity fingerprint anchor")
    if build.get("native_library_sha256") is not None:
        require(cjson(build["native_library_sha256"]) == cjson(files["infinity"]["native_library_sha256"]),
            "build native libraries do not match Infinity fingerprint anchor")
    rawinst = d.get("runtime_installations")
    require(isinstance(rawinst, dict), "campaign.runtime_installations missing")
    inst = {}
    for r in RT:
        e = rawinst.get(r)
        require(isinstance(e, dict), f"runtime installation missing for {r}")
        tree = e.get("package_tree_sha256")
        if r in ("vllm", "sglang"):
            require(h64(tree) and tree == files[r]["runtime_package_tree_sha256"],
                f"{r} package tree does not match fingerprint anchor")
            require(h64(e.get("record_sha256")), f"{r} record_sha256 missing")
            require(isinstance(e.get("record_mismatches"), list) and not e["record_mismatches"],
                f"{r} record mismatches present; stock claim rejected")
            require(isinstance(e.get("unrecorded_files"), list) and not e["unrecorded_files"],
                f"{r} unrecorded files present; stock claim rejected")
            stock = "stock"
        else:
            if tree is not None and files[r]["runtime_package_tree_sha256"] is not None:
                require(tree == files[r]["runtime_package_tree_sha256"],
                    "Infinity package tree does not match fingerprint anchor")
            stock = "source_clean" if e.get("source_clean") is True else "unverified"
        inst[r] = {"version": e.get("version"),
                   "package_tree_sha256": clean(tree),
                   "record_sha256": clean(e.get("record_sha256")),
                   "record_mismatch_count": len(e.get("record_mismatches") or []),
                   "unrecorded_count": len(e.get("unrecorded_files") or []),
                   "stock": stock}
    return {"doc": d, "sha": shafile(Path(root) / "campaign.json"), "id": d.get("id"),
            "date_local": d.get("date_local"), "timezone": d.get("timezone"),
            "commit": commit, "binary": d["binary_sha256"], "versions": vers,
            "hardware": hw, "method": {
                "seconds": SECONDS, "warmup": WARMUP, "repeats": WINDOWS,
                "concurrency": meth["concurrency"], "long": 8,
                "max_batch_tokens": meth["max_batch_tokens"],
                "max_sequence_length": meth["max_sequence_length"], "dtype": DTYPE,
                "request_batch_size": 1, "prefix_cache": meth.get("prefix_cache"),
                "lanes": clean(lanes)},
            "files": files, "build": build, "runtime_installations": inst,
            "lanes": lanes, "lane_keys": lane_keys}

def load_manifest(root, camp):
    p = Path(root) / "manifest.json"
    d = load(p)
    require(isinstance(d, dict) and isinstance(d.get("models"), list) and d["models"],
        "manifest.models missing")
    out, seen = [], set()
    for raw in d["models"]:
        require(isinstance(raw, dict), "manifest model is not an object")
        mid = raw.get("id")
        require(isinstance(mid, str) and mid and mid.strip() == mid, "manifest model id invalid")
        require(mid not in seen, f"duplicate model {mid}")
        seen.add(mid)
        require(raw.get("task") in SEM, f"{mid}.task invalid")
        require(revok(raw.get("revision")), f"{mid}.revision invalid")
        ls = raw.get("lengths")
        require(isinstance(ls, list) and ls and all(pint(x) for x in ls) and len(set(ls)) == len(ls),
            f"{mid}.lengths invalid")
        params = raw.get("parameters")
        require(params is None or (isinstance(params, int) and not isinstance(params, bool) and params >= 0),
            f"{mid}.parameters invalid")
        out.append({"id": mid, "task": raw["task"], "revision": raw["revision"],
                    "lengths": ls, "parameters": params,
                    "selection": clean(raw.get("selection")), "raw": raw})
    return {"sha": shafile(p), "models": out}

def expected(model):
    return [(n, None) for n in model["lengths"]]

def concurrency(camp, n):
    return camp["method"]["long"] if n >= 2048 else camp["method"]["concurrency"]

def ledger_errors(obj, label, lat):
    if not isinstance(obj, dict):
        return [f"{label} missing"]
    e = []
    for k in ("attempted", "responses", "ok"):
        if not pint(obj.get(k)):
            e.append(f"{label} {k} invalid")
    if not (pint(obj.get("attempted")) and obj.get("attempted") == obj.get("responses") == obj.get("ok")):
        e.append(f"{label} counters unequal")
    for k in ("errors", "unanswered", "failure_count"):
        v = obj.get(k)
        if isinstance(v, bool) or not isinstance(v, int) or v != 0:
            e.append(f"{label} {k} nonzero")
    if num(obj.get("elapsed_seconds")) is None or obj["elapsed_seconds"] <= 0:
        e.append(f"{label} elapsed invalid")
    if not isinstance(obj.get("failure_samples"), list) or obj["failure_samples"]:
        e.append(f"{label} failure samples present")
    if lat:
        lm = obj.get("latency_ms")
        if not isinstance(lm, dict):
            e.append(f"{label} latency missing")
        else:
            for k in ("p50", "p95"):
                if num(lm.get(k)) is None or lm[k] <= 0:
                    e.append(f"{label} latency {k} invalid")
            v = lm.get("samples")
            if isinstance(v, bool) or not isinstance(v, int) or v != obj.get("ok"):
                e.append(f"{label} latency samples invalid")
    return e

def serial(p):
    if not isinstance(p, dict):
        return False, ["serial probe missing"], {}
    e = ledger_errors(p, "serial", False)
    if p.get("length") != 8:
        e.append("serial length must be 8")
    if p.get("concurrency") != 1:
        e.append("serial concurrency must be 1")
    if p.get("serial_requests") != 20 or p.get("ok") != 20:
        e.append("serial_requests must be 20 and equal ok")
    s = p.get("all_latencies_ms")
    if not isinstance(s, list) or not s or len(s) != p.get("ok"):
        e.append("serial samples do not cover ok")
        s = []
    elif not all(num(x) is not None and x > 0 for x in s):
        e.append("serial samples invalid")
    m = num(p.get("min_ms"))
    if m is None or m <= 0:
        e.append("serial min invalid")
    elif s and not math.isclose(m, min(s), rel_tol=1e-12, abs_tol=1e-12):
        e.append("serial min mismatch")
    lm = p.get("latency_ms") or {}
    return not e, e, {"valid": not e, "serial_min_ms": m if not e else None,
                      "p50_ms": num(lm.get("p50")), "p95_ms": num(lm.get("p95")),
                      "serial_requests": p.get("serial_requests"), "length": p.get("length")}

def flags(cmd, label):
    require(isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd), f"{label} invalid")
    out, i = {}, 0
    while i < len(cmd):
        t = cmd[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1); out[k] = v; i += 1
            elif i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                out[t] = cmd[i + 1]; i += 2
            else:
                out[t] = True; i += 1
        else:
            i += 1
    return out

def fnum(f, k, label):
    v = f.get(k)
    require(v not in (None, True), f"{label} {k} missing")
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ReportError(f"{label} {k} not numeric")

def taskset(cmd, label):
    require(isinstance(cmd, list), f"{label} invalid")
    try:
        i = cmd.index("taskset")
    except ValueError:
        raise ReportError(f"{label} lacks taskset")
    require(i + 2 < len(cmd) and cmd[i + 1] == "-c", f"{label} taskset invalid")
    return cmd[i + 2]

def deps(v, label):
    if v is None:
        return []
    require(isinstance(v, list), f"{label} invalid")
    out = []
    for x in v:
        require(isinstance(x, dict) and isinstance(x.get("id"), str) and x["id"] and revok(x.get("revision")),
            f"{label} entry invalid")
        out.append({"id": x["id"], "revision": x["revision"]})
    return out

def validate_fingerprint(camp, model, raw, cp, revision, status):
    x = raw.get("fingerprint_inputs")
    require(isinstance(x, dict) and set(x) >= {"model", "checkpoint", "runtime", "lane", "settings", "files"},
        f"{model['id']}: fingerprint_inputs missing keys")
    require(cjson(x["model"]) == cjson(portable(model["raw"])),
        f"{model['id']}: fingerprint model mismatch")
    require(x["runtime"] == raw["runtime"], f"{model['id']}: fingerprint runtime mismatch")
    c = x["checkpoint"]
    require(isinstance(c, dict) and set(c) >= {"id", "revision", "metadata_sha256", "code_dependencies"},
        f"{model['id']}: fingerprint checkpoint invalid")
    require(c["id"] == model["id"] == cp.get("id"), f"{model['id']}: checkpoint id must equal model id")
    require(c["revision"] == revision, f"{model['id']}: checkpoint revision mismatch")
    meta = c["metadata_sha256"]
    if not (status == "checkpoint_unavailable" and meta in (None, "")):
        require(h64(meta), f"{model['id']}: checkpoint metadata_sha256 invalid")
    require(cp.get("metadata_sha256") in (None, meta), f"{model['id']}: raw metadata mismatch")
    cd = deps(c["code_dependencies"], "fingerprint dependencies")
    if cp.get("code_dependencies") is not None:
        require(cjson(cd) == cjson(deps(cp["code_dependencies"], "raw dependencies")),
            f"{model['id']}: dependency projection mismatch")
    if model["raw"].get("code_dependencies") is not None:
        require(cjson(cd) == cjson(deps(model["raw"]["code_dependencies"], "manifest dependencies")),
            f"{model['id']}: manifest dependency mismatch")
    lane = x["lane"]
    require(isinstance(lane, dict), f"{model['id']}: lane missing")
    for k in ("gpu", "cpus", "client_cpus"):
        require(k in lane and place(lane[k]) == place(raw[k]), f"{model['id']}: lane {k} mismatch")
    s = x["settings"]
    require(isinstance(s, dict) and set(s) >= SET_KEYS, f"{model['id']}: settings missing keys")
    m = camp["method"]
    for k, want in (("seconds", m["seconds"]), ("warmup", m["warmup"]),
                    ("repeats", m["repeats"]), ("concurrency", m["concurrency"]),
                    ("long_input_concurrency", m["long"]),
                    ("max_batch_tokens", m["max_batch_tokens"]),
                    ("max_sequence_length", m["max_sequence_length"])):
        require(num(s.get(k)) == float(want) if isinstance(want, float) else s.get(k) == want,
            f"{model['id']}: settings {k} mismatch")
    require(dtype(s["dtype"]) == DTYPE, f"{model['id']}: settings dtype mismatch")
    require(s["lengths"] == model["lengths"], f"{model['id']}: settings lengths mismatch")
    ctx, ictx = s["context"], s["input_context"]
    if status in ("measured", "client_failed"):
        require(pint(ctx) and pint(ictx), f"{model['id']}: context limits invalid")
    else:
        require((ctx is None or pint(ctx)) and (ictx is None or pint(ictx)),
            f"{model['id']}: context limits invalid")
    if pint(ictx):
        require(ictx <= m["max_sequence_length"], f"{model['id']}: input_context exceeds declared max")
    require(isinstance(s["runtime_environment"], dict), f"{model['id']}: runtime_environment invalid")
    require(cjson(s["runtime_environment"]) == cjson(REQUIRED_ENV[raw["runtime"]]),
            f"{model['id']}: runtime_environment policy mismatch")
    require(cjson(x["files"]) == cjson(camp["files"][raw["runtime"]]),
        f"{model['id']}: fingerprint files do not match per-runtime anchor")
    require(isinstance(raw.get("run_fingerprint"), str) and fp(x) == raw["run_fingerprint"],
        f"{model['id']}: run_fingerprint recompute mismatch")
    return x

def validate_commands(camp, model, raw, meas, fin):
    a = meas.get("arguments")
    require(isinstance(a, dict), "measurement.arguments missing")
    require(not set(a) - ARGS, f"unknown measurement argument controls: {sorted(set(a) - ARGS)}")
    require(a.get("runtime") == raw["runtime"] and a.get("task") == raw["task"],
        "measurement.arguments identity mismatch")
    require(isinstance(a.get("model"), str) and a["model"], "served model missing")
    require(num(a.get("seconds")) == SECONDS and num(a.get("warmup")) == WARMUP
        and a.get("repeats") == WINDOWS and a.get("concurrency") == camp["method"]["concurrency"],
        "measurement.arguments controls mismatch")
    require(a.get("min_latency_length") == 8, "measurement.arguments.min_latency_length must be 8")
    require(a.get("lengths") == model["lengths"], "measurement.arguments lengths mismatch")
    if a.get("batch_size") is not None:
        require(a["batch_size"] == 1, "measurement.arguments.batch_size must be 1")
    if a.get("dtype") is not None:
        require(dtype(a["dtype"]) == DTYPE, "measurement.arguments dtype mismatch")
    rt = meas.get("runtime")
    require(isinstance(rt, dict), "measurement.runtime missing")
    require(rt.get("requested") == raw["runtime"] and rt.get("model") == a["model"]
        and rt.get("task") == raw["task"], "measurement.runtime identity mismatch")
    require(place(taskset(raw["client_command"], "client_command")) == place(raw["client_cpus"]),
        "client taskset CPUs mismatch")
    require(place(taskset(raw["command"], "command")) == place(raw["cpus"]), "server taskset CPUs mismatch")
    cf = flags(raw["client_command"], "client_command")
    require(fnum(cf, "--seconds", "client") == SECONDS and fnum(cf, "--warmup", "client") == WARMUP
        and fnum(cf, "--repeats", "client") == WINDOWS
        and fnum(cf, "--concurrency", "client") == camp["method"]["concurrency"],
        "client command controls mismatch")
    if "--min-latency-length" in cf:
        require(fnum(cf, "--min-latency-length", "client") == 8, "client min latency must be 8")
    for k, want in (("--task", raw["task"]), ("--runtime", raw["runtime"]), ("--model", a["model"])):
        require(cf.get(k) == want, f"client command {k} mismatch")
    require([int(x) for x in str(cf.get("--lengths", "")).split(",") if x] == model["lengths"],
        "client command lengths mismatch")
    sf = flags(raw["command"], "command")
    require(dtype(sf.get("--dtype")) == DTYPE, "server dtype mismatch")
    s = fin["settings"]
    env = raw.get("environment_overrides") or {}
    require(isinstance(env, dict), "environment_overrides invalid")
    for k, v in REQUIRED_ENV[raw["runtime"]].items():
        require(str(env.get(k)) == str(v), f"environment override {k} mismatch")
    if "CUDA_VISIBLE_DEVICES" in env:
        require(str(env["CUDA_VISIBLE_DEVICES"]) == str(raw["gpu"]), "CUDA_VISIBLE_DEVICES mismatch")
    if raw["runtime"] == "infinity":
        require(s["context"] == s["input_context"], "Infinity context mismatch")
        require(fnum(sf, "--max-batch-tokens", "infinity") == s["max_batch_tokens"],
            "Infinity batch flag mismatch")
        require(fnum(sf, "--max-sequence-length", "infinity") == s["context"],
            "Infinity sequence flag mismatch")
        require(str(sf.get("--radix-mlp-threshold")) == "0", "Infinity radix threshold must be 0")
    elif raw["runtime"] == "vllm":
        require(s["context"] <= s["input_context"], "vLLM context exceeds input_context")
        require(fnum(sf, "--max-num-batched-tokens", "vllm") == s["max_batch_tokens"],
            "vLLM batch flag mismatch")
        require(fnum(sf, "--max-model-len", "vllm") == s["context"], "vLLM max-model-len mismatch")
        require("--no-enable-prefix-caching" in sf, "vLLM prefix cache must be disabled")
        require(sf.get("--runner") == "pooling", "vLLM runner must be pooling")
        if model["task"] == "embeddings":
            require("--pooler-config" in sf, "vLLM dense pooler config missing")
            try:
                p = json.loads(sf["--pooler-config"])
            except ValueError:
                raise ReportError("vLLM pooler config is not JSON")
            require(isinstance(p, dict) and set(p) == {"use_activation"} and p["use_activation"] is True,
                "vLLM dense pooler must only enable use_activation")
    else:
        require(s["context"] == s["input_context"] + 1, "SGLang context must be input_context + 1")
        require(fnum(sf, "--max-prefill-tokens", "sglang") == s["max_batch_tokens"],
            "SGLang prefill flag mismatch")
        require(fnum(sf, "--context-length", "sglang") == s["context"], "SGLang context flag mismatch")
        require("--disable-radix-cache" in sf, "SGLang radix cache must be disabled")

def windows_rate(ws, keys):
    for k in keys:
        total = elapsed = 0.0
        for w in ws:
            st = (w.get("tokens_per_request") or {}).get(k)
            if not pint(w.get("ok")) or not isinstance(st, dict) or st.get("count") != w["ok"]:
                break
            mean, el = num(st.get("mean")), num(w.get("elapsed_seconds"))
            if mean is None or el is None or el <= 0:
                break
            total += mean * w["ok"]; elapsed += el
        else:
            if elapsed > 0:
                return total / elapsed
    return None

def metrics(ws):
    a = r = o = f = 0
    el = 0.0
    p50, p95 = [], []
    for w in ws:
        if not isinstance(w, dict):
            continue
        a += w.get("attempted") if isinstance(w.get("attempted"), int) else 0
        r += w.get("responses") if isinstance(w.get("responses"), int) else 0
        o += w.get("ok") if isinstance(w.get("ok"), int) else 0
        f += w.get("failure_count") if isinstance(w.get("failure_count"), int) else 0
        el += num(w.get("elapsed_seconds")) or 0
        p50.append(num((w.get("latency_ms") or {}).get("p50")))
        p95.append(num((w.get("latency_ms") or {}).get("p95")))
    return {"attempted": a, "responses": r, "ok": o, "errors": a - o,
            "unanswered": a - r, "failure_count": f, "elapsed_seconds": el,
            "throughput_rps": o / el if o and el > 0 else None,
            "server_input_tokens_per_second": windows_rate(ws, ("usage_prompt", "usage_input")),
            "local_input_tokens_per_second": windows_rate(ws, ("local",)),
            "per_window_p50_ms": p50, "per_window_p95_ms": p95,
            "window_p50_median_ms": statistics.median(p50) if p50 and all(x is not None for x in p50) else None,
            "window_p95_median_ms": statistics.median(p95) if p95 and all(x is not None for x in p95) else None,
            "serial_min_ms": None, "metrics_diagnostic": True}

def workload(camp, model, raw, w, key):
    require(isinstance(w, dict), "workload invalid")
    n, v = w.get("length"), w.get("variant")
    require(v is None and pint(n) and (n, v) == key, "workload case mismatch")
    c = concurrency(camp, n)
    require(w.get("concurrency") == c, f"workload {key} concurrency mismatch")
    d = w.get("descriptor")
    require(isinstance(d, dict) and d.get("task") == raw["task"] and d.get("length") == n and d.get("concurrency") == c,
        f"workload {key} descriptor mismatch")
    ss = d.get("samples")
    require(isinstance(ss, list) and ss, f"workload {key} descriptor samples missing")
    for s in ss:
        semantic_hash(raw["task"], s, f"workload {key} sample")
        require(s.get("local_tokens") == n, f"workload {key} sample local_tokens mismatch")
    warm = w.get("warmup")
    reasons = ledger_errors(warm, f"workload {key} warmup", False)
    require(isinstance(warm, dict) and warm.get("concurrency") == c, f"workload {key} warmup concurrency")
    if isinstance(warm, dict) and num(warm.get("elapsed_seconds")) is not None \
            and warm["elapsed_seconds"] < WARMUP - TOL:
        reasons.append(f"workload {key} warmup too short")
    ws = w.get("windows")
    require(isinstance(ws, list), f"workload {key} windows missing")
    if len(ws) != WINDOWS:
        reasons.append(f"workload {key} needs {WINDOWS} windows")
    for i, x in enumerate(ws):
        require(isinstance(x, dict), f"workload {key} window invalid")
        require(x.get("concurrency") == c, f"workload {key} window {i} concurrency mismatch")
        reasons += ledger_errors(x, f"workload {key} window {i}", True)
        if num(x.get("elapsed_seconds")) is not None and x["elapsed_seconds"] < SECONDS - TOL:
            reasons.append(f"workload {key} window {i} too short")
    p = w.get("numerical_probe")
    sample = record = None
    if not isinstance(p, dict):
        reasons.append("numerical probe missing")
    else:
        if p.get("failure_count") != 0:
            reasons.append("probe failure_count nonzero")
        if not isinstance(p.get("failures"), list) or p["failures"]:
            reasons.append("probe failures present")
        s = p.get("sample")
        if not isinstance(s, dict) or not any(k in s for k in SEM[raw["task"]]):
            reasons.append("probe sample missing")
        else:
            sample = semantic_hash(raw["task"], s, f"workload {key} probe sample")
            require(s.get("local_tokens") == n, f"workload {key} probe local_tokens mismatch")
        if not reasons:
            record, err = probe(raw["task"], p.get("record"))
            if err:
                reasons.append(f"probe invalid: {err}")
    m = metrics(ws)
    return {"length": n, "variant": None, "concurrency": c, "warmup": warm, "windows": ws,
            "sample": sample, "record": p.get("record") if isinstance(p, dict) else None,
            "probe_valid": sample is not None and record is not None,
            "probe_error": "; ".join(reasons) if reasons else None,
            "metrics": m, "clean_reasons": reasons}

def parse_run(camp, model, path):
    b = path.read_bytes()
    d = json.loads(b.decode())
    require(isinstance(d, dict), f"{path.name}: run not object")
    require(d.get("model") == model["id"], f"{path.name}: model mismatch")
    require(d.get("task") == model["task"], f"{path.name}: task mismatch")
    r = d.get("runtime")
    require(r in RTS, f"{path.name}: runtime mismatch")
    st = d.get("status")
    require(st in TERMINAL, f"{path.name}: unknown status {st!r}")
    dt = dtype(d.get("dtype"))
    require(dt == DTYPE, f"{path.name}: dtype mismatch")
    cp = d.get("checkpoint")
    require(isinstance(cp, dict), f"{path.name}: checkpoint missing")
    rev = cp.get("revision")
    if st == "checkpoint_unavailable":
        require(rev is None or revok(rev), f"{path.name}: checkpoint revision invalid")
    else:
        require(revok(rev), f"{path.name}: checkpoint revision invalid")
    require(rev is None or rev == model["revision"], f"{path.name}: checkpoint revision mismatch")
    gpu, cpus, ccpu = place(d.get("gpu")), place(d.get("cpus")), place(d.get("client_cpus"))
    require((gpu, cpus, ccpu) in camp["lane_keys"],
            f"{path.name}: placement does not match a declared campaign.method.lanes entry")
    fin = validate_fingerprint(camp, model, d, cp, rev, st)
    startup = num(d.get("startup_s"))
    if st == "measured":
        require(startup is not None and startup > 0, f"{path.name}: startup_s invalid")
    meas = d.get("measurement")
    cas = {}
    ser = {"valid": False, "errors": ["serial missing"], "serial_min_ms": None}
    endpoint = False
    exp = expected(model)
    if st in ("measured", "client_failed") and meas is not None:
        require(isinstance(meas, dict), f"{path.name}: measurement invalid")
        if st == "measured":
            require(meas.get("ok") is True, f"{path.name}: measurement.ok false")
        validate_commands(camp, model, d, meas, fin)
        endpoint = any(isinstance(x, dict) and x.get("phase") in ("endpoint_resolution", "fatal")
                       for x in meas.get("failures") or [])
        ws = meas.get("workloads")
        require(isinstance(ws, list), f"{path.name}: workloads missing")
        for w in ws:
            require(isinstance(w, dict), f"{path.name}: workload invalid")
            key = (w.get("length"), w.get("variant"))
            require(key in exp, f"{path.name}: unknown case {key}")
            require(key not in cas, f"{path.name}: duplicate case {key}")
            cas[key] = workload(camp, model, d, w, key)
        if st == "measured":
            require(set(cas) == set(exp), f"{path.name}: measured run missing cases")
        ok, errs, sm = serial(meas.get("latency_probe"))
        ser = {**sm, "valid": ok, "errors": errs}
    elif st == "measured":
        raise ReportError(f"{path.name}: measured run has no measurement")
    for case in cas.values():
        if endpoint:
            case["clean_reasons"].append("endpoint/fatal failure")
        case["clean"] = not case["clean_reasons"]
        if ser["valid"]:
            case["metrics"]["serial_min_ms"] = ser["serial_min_ms"]
    error = bounded(d.get("error"))
    error_hint = bounded(d.get("error_hint"))
    failure_reason = error or (None if st == "measured" else st)
    return {"path": f"runs/{path.name}", "sha": sha(b), "doc": d, "model": model["id"],
            "task": model["task"], "runtime": r, "status": st, "revision": rev,
            "gpu": gpu, "cpus": cpus, "client_cpus": ccpu, "startup": startup,
            "fingerprint": d["run_fingerprint"], "cases": cas, "serial": ser,
            "error": error, "error_hint": error_hint, "failure_reason": failure_reason}

def loadruns(camp, models, root, allow):
    dirs = [Path(root) / "runs", Path(root) / "sweep" / "runs"]
    rd = next((x for x in dirs if x.is_dir()), None)
    files = sorted(rd.glob("*.json")) if rd is not None else []
    if not files:
        if allow:
            return {}
        raise ReportError("campaign has no runs yet (incomplete; use --allow-incomplete for PROVISIONAL)")
    runs, errs = {}, []
    byid = {m["id"]: m for m in models}
    for p in files:
        try:
            require(p.read_bytes()[:1] not in (b"",), f"{p.name}: empty")
            d = load(p)
            mid = d.get("model")
            require(mid in byid, f"{p.name}: unknown model {mid!r}")
            x = parse_run(camp, byid[mid], p)
        except (ReportError, ValueError, UnicodeError) as e:
            errs.append(str(e)); continue
        k = (x["model"], x["runtime"])
        if k in runs:
            errs.append(f"{p.name}: duplicate run {k}")
        else:
            runs[k] = x
    if errs:
        raise ReportError("malformed/unknown/duplicate runs:\n- " + "\n- ".join(errs))
    return runs

def probe_reference(x, c):
    s = c.get("sample") or {}
    rec = c.get("record") or {}
    return {"runtime": x["runtime"], "raw": {"path": x["path"], "sha256": x["sha"]},
            "checkpoint_revision": clean(x["revision"]), "semantic_sha256": s.get("sha256"),
            "shape": clean(rec.get("shape")), "gpu": x["gpu"], "cpus": x["cpus"],
            "client_cpus": x["client_cpus"]}

def compare(task, a, b):
    ra, ea = probe(task, a["record"]); rb, eb = probe(task, b["record"])
    if ea or eb:
        return {"status": "invalid_probe", "numerical_pass": False,
                "reason": f"reference={ea}, candidate={eb}"}
    if ra["shape"] != rb["shape"]:
        return {"status": "shape_mismatch", "numerical_pass": False, "reason": "shape mismatch"}
    if task == "embeddings":
        c = cos(ra["flat"], rb["flat"])
        if c is None:
            return {"status": "invalid_vector", "numerical_pass": False, "reason": "cosine undefined"}
        n = normclose(ra["flat"], rb["flat"])
        if c < COS:
            return {"status": "fail_cosine", "numerical_pass": False,
                    "reason": f"cosine {c:.8f} < {COS}", "cosine": c, "norm_close": n}
        if not n:
            return {"status": "fail_norm", "numerical_pass": False,
                    "reason": "norm mismatch", "cosine": c, "norm_close": n}
        return {"status": "pass", "numerical_pass": True, "cosine": c, "norm_close": n,
                "method": "dense_cosine"}
    ok = allclose(ra["flat"], rb["flat"])
    return {"status": "pass" if ok else "fail_values", "numerical_pass": ok,
            "reason": None if ok else "ordered scalar allclose failed",
            "method": "ordered_allclose", "allclose": ok,
            "max_abs_diff": max((abs(x - y) for x, y in zip(ra["flat"], rb["flat"])), default=None)}

def load_fp32(path, camp, models, runs, helper_path):
    try:
        raw = Path(path).read_bytes()
        d = json.loads(raw.decode())
    except (OSError, UnicodeError, ValueError) as e:
        raise ReportError(f"FP32 reference is not readable JSON: {str(e).splitlines()[0][:200]}")
    require(isinstance(d, dict), "FP32 reference must be an object")
    require(d.get("schema_version") == 1, "FP32 reference schema_version must be 1")
    require(d.get("kind") == "independent_hf_fp32_numerical_smoke_reference",
            "FP32 reference kind mismatch")
    helper_sha = shafile(helper_path)
    client_sha = camp["files"]["infinity"]["client_sha256"]
    manifest_sha = shafile(Path(camp["root"]) / "manifest.json")
    require(d.get("source_sha256") == helper_sha,
            "FP32 source_sha256 does not match accepted helper bytes")
    require(d.get("client_sha256") == client_sha,
            "FP32 client_sha256 does not match campaign fingerprint anchors")
    require(d.get("manifest_sha256") == manifest_sha,
            "FP32 manifest_sha256 does not match campaign manifest")
    sh = d.get("source_hashes")
    require(isinstance(sh, dict), "FP32 source_hashes missing")
    require(sh.get("reference_source") == helper_sha
            and sh.get("http_client_helper") == client_sha
            and sh.get("manifest") == manifest_sha,
            "FP32 source_hashes are inconsistent")
    require(d.get("dtype") == "float32", "FP32 reference dtype must be float32")
    byid = {m["id"]: m for m in models}
    expected_keys = {(m["id"], m["revision"], m["task"], n)
                     for m in models for n in m["lengths"]}
    rawrefs = d.get("references")
    rawfails = d.get("failures")
    require(isinstance(rawrefs, list) and isinstance(rawfails, list),
            "FP32 references/failures must be lists")
    refs, issues, seen = {}, [], {}
    for rawrec in rawrefs:
        if isinstance(rawrec, dict):
            mid0, rev0, task0, length0 = (rawrec.get("model"), rawrec.get("revision"),
                                          rawrec.get("task"), rawrec.get("length"))
            if isinstance(mid0, str) and isinstance(rev0, str) and isinstance(task0, str) \
                    and isinstance(length0, int) and not isinstance(length0, bool):
                key0 = (mid0, rev0, task0, length0)
                if key0 in seen:
                    require(cjson(seen[key0]) == cjson(rawrec),
                            f"conflicting duplicate FP32 reference {key0}")
                    continue
        try:
            require(isinstance(rawrec, dict), "reference record is not an object")
            mid = rawrec.get("model")
            require(mid in byid, f"unknown FP32 model {mid!r}")
            model = byid[mid]
            rev, task = rawrec.get("revision"), rawrec.get("task")
            length = rawrec.get("length")
            require(task == model["task"] and rev == model["revision"]
                    and length in model["lengths"], "FP32 identity does not match manifest")
            key = (mid, rev, task, length)
            require(key in expected_keys, "FP32 case not expected")
            seen[key] = rawrec
            sample = rawrec.get("sample_sha256")
            require(h64(sample), "FP32 sample_sha256 invalid")
            require(rawrec.get("local_tokens") == length, "FP32 local_tokens mismatch")
            require(rawrec.get("dtype") == "float32" and rawrec.get("all_finite") is True,
                    "FP32 record dtype/all_finite invalid")
            loading = rawrec.get("loading")
            require(isinstance(loading, dict) and loading.get("validated") is True,
                    "FP32 model loading was not validated")
            rec, err = probe(task, rawrec)
            require(err is None, f"FP32 values invalid: {err}")
            require(isinstance(rawrec.get("device"), str), "FP32 device missing")
            if task == "embeddings":
                require(isinstance(rawrec.get("pooling"), str) and rawrec["pooling"]
                        and rawrec.get("pooling_source") == "1_Pooling/config.json"
                        and rawrec.get("normalize") is True
                        and rawrec.get("normalization") == "l2",
                        "FP32 dense pooling/normalization invalid")
            else:
                require(rawrec.get("pooling") == "none"
                        and rawrec.get("normalize") is False
                        and rawrec.get("normalization") == "none",
                        "FP32 rerank normalization invalid")
            cp = rawrec.get("checkpoint")
            require(isinstance(cp, dict) and cp.get("id") == mid
                    and cp.get("revision") == rev and cp.get("status") == "ready",
                    "FP32 checkpoint identity/status mismatch")
            for k in ("receipt_sha256", "metadata_sha256"):
                require(h64(cp.get(k)), f"FP32 checkpoint {k} invalid")
            dep_list = deps(cp.get("code_dependencies"), "FP32 dependencies")
            matches = []
            for r in RT:
                run = runs.get((mid, r))
                case = run["cases"].get((length, None)) if run else None
                if case and (case.get("sample") or {}).get("sha256") == sample:
                    matches.append((run, case))
            for run, case in matches:
                require(run["revision"] == rev, "FP32 revision does not match current run")
                rcp = run["doc"].get("checkpoint") or {}
                require(rcp.get("id") == mid and rcp.get("metadata_sha256") == cp["metadata_sha256"],
                        "FP32 checkpoint metadata does not match current run")
                require(cjson(deps(rcp.get("code_dependencies"), "raw deps")) == cjson(dep_list),
                        "FP32 dependencies do not match current run")
                fcp = ((run["doc"].get("fingerprint_inputs") or {}).get("checkpoint") or {})
                require(fcp.get("metadata_sha256") == cp["metadata_sha256"]
                        and cjson(deps(fcp.get("code_dependencies"), "fp deps")) == cjson(dep_list),
                        "FP32 identity does not match fingerprint")
            ref = dict(rawrec)
            ref["_flat"], ref["_norm"] = rec["flat"], rec["norm"]
            ref["_match"] = "matched" if matches else "no_current_case"
            if matches:
                ref["_semantic"] = (matches[0][1].get("sample") or {}).get("fields")
            refs[key] = ref
        except (ReportError, ValueError, TypeError) as e:
            issues.append({"reason": bounded(str(e)) or "invalid FP32 reference"})
    failures = []
    for rawfail in rawfails:
        if not isinstance(rawfail, dict):
            issues.append({"reason": "malformed FP32 failure record"})
            continue
        failures.append({"model": clean(rawfail.get("model")),
                         "revision": clean(rawfail.get("revision")),
                         "task": clean(rawfail.get("task")),
                         "length": rawfail.get("length"),
                         "status": clean(rawfail.get("status")),
                         "reason": bounded(rawfail.get("reason")) or "unspecified"})
    prov = {"schema_version": 1, "kind": d["kind"], "source_sha256": helper_sha,
            "client_sha256": client_sha, "manifest_sha256": manifest_sha,
            "reference_file_sha256": sha(raw), "torch_version": clean(d.get("torch_version")),
            "transformers_version": clean(d.get("transformers_version")),
            "device": clean(d.get("device")), "threads": d.get("threads"),
            "dtype": "float32", "source_hashes": clean(sh),
            "reference_count": len(rawrefs), "failure_count": len(rawfails)}
    return {"provenance": prov, "references": refs, "issues": issues, "failures": failures}

def fp32_compare(refctx, camp, models, runs, comps):
    rel = {(x["model"], x["runtime"], x["length"]): x for x in comps}
    failkeys = {(x.get("model"), x.get("revision"), x.get("task"), x.get("length"))
                for x in refctx["failures"]}
    out, rows = {}, []
    for model in models:
        for length in model["lengths"]:
            key = (model["id"], model["revision"], model["task"], length)
            ref = refctx["references"].get(key)
            for r in RT:
                run = runs.get((model["id"], r))
                case = run["cases"].get((length, None)) if run else None
                row = {"model": model["id"], "runtime": r, "length": length}
                if ref is None:
                    status = "reference_failure" if key in failkeys else "reference_missing"
                    row.update(status=status, fp32_pass=False)
                elif case is None:
                    row.update(status="case_missing", fp32_pass=False)
                elif not case.get("clean"):
                    row.update(status="case_not_clean", fp32_pass=False)
                elif ref["_match"] != "matched":
                    row.update(status="sample_unmatched", fp32_pass=False)
                elif (case.get("sample") or {}).get("sha256") != ref.get("sample_sha256"):
                    row.update(status="sample_mismatch", fp32_pass=False)
                else:
                    cmp = compare(model["task"], {"record": ref}, case)
                    row.update(status=cmp["status"], fp32_pass=cmp["numerical_pass"],
                               cosine=cmp.get("cosine"), norm_close=cmp.get("norm_close"),
                               allclose=cmp.get("allclose"),
                               max_abs_diff=cmp.get("max_abs_diff"))
                out[(model["id"], r, length)] = row
                if case is not None:
                    case["fp32"] = row
    for model in models:
        for length in model["lengths"]:
            inf = out[(model["id"], "infinity", length)]
            inf_run = runs.get((model["id"], "infinity"))
            inf_case = inf_run["cases"].get((length, None)) if inf_run else None
            inf["fully_eligible"] = bool(
                inf_case is not None and inf_case.get("clean") and inf.get("fp32_pass"))
            inf["fp32_baseline_eligible"] = inf["fully_eligible"]
            for r in ("vllm", "sglang"):
                row = out[(model["id"], r, length)]
                run = runs.get((model["id"], r))
                case = run["cases"].get((length, None)) if run else None
                row["fully_eligible"] = bool(
                    case is not None and case.get("clean")
                    and (rel.get((model["id"], r, length)) or {}).get("numerical_pass") is True
                    and inf.get("fp32_pass") and row.get("fp32_pass"))
    counts = {}
    for row in out.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    coverage = {"reference_file_sha256": refctx["provenance"]["reference_file_sha256"],
                "reference_records": len(refctx["references"]),
                "reference_issues": len(refctx["issues"]),
                "reference_failures": len(refctx["failures"]),
                "runtime_cases": len(out),
                "status_counts": dict(sorted(counts.items())),
                "fp32_passes": sum(1 for x in out.values() if x.get("fp32_pass")),
                "fully_eligible_cases": sum(1 for x in out.values() if x.get("fully_eligible")),
                "eligible_comparisons": sum(1 for x in out.values()
                                            if x.get("fully_eligible")
                                            and x.get("runtime") != "infinity")}
    for row in out.values():
        rows.append({k: clean(v) for k, v in row.items() if k != "_flat"})
    return {"provenance": refctx["provenance"],
            "coverage": coverage,
            "issues": refctx["issues"],
            "failures": refctx["failures"],
            "comparisons": rows}, out

def comparisons(camp, runs, models):
    out = []
    for m in models:
        ir = runs.get((m["id"], "infinity"))
        for key in expected(m):
            ic = ir["cases"].get(key) if ir else None
            for r in ("vllm", "sglang"):
                cr = runs.get((m["id"], r)); cc = cr["cases"].get(key) if cr else None
                base = {"model": m["id"], "task": m["task"], "runtime": r,
                        "length": key[0], "variant": None, "numerical_pass": False,
                        "reference": probe_reference(ir, ic) if ir and ic else None,
                        "candidate": probe_reference(cr, cc) if cr and cc else None}
                if ic is None or not ic.get("clean"):
                    out.append({**base, "status": "reference_not_eligible",
                                "reason": "no clean Infinity reference"}); continue
                if cc is None or not cc.get("clean"):
                    out.append({**base, "status": "candidate_not_eligible",
                                "reason": "candidate missing or not clean"}); continue
                if ir["revision"] != cr["revision"]:
                    out.append({**base, "status": "checkpoint_sha_mismatch",
                                "reason": "checkpoint revision differs"}); continue
                if ic["sample"]["sha256"] != cc["sample"]["sha256"]:
                    out.append({**base, "status": "input_hash_mismatch",
                                "reason": "semantic hash differs"}); continue
                if (ic["record"] or {}).get("shape") != (cc["record"] or {}).get("shape"):
                    out.append({**base, "status": "shape_mismatch", "reason": "shape differs"}); continue
                if (ir["gpu"], ir["cpus"], ir["client_cpus"]) != (cr["gpu"], cr["cpus"], cr["client_cpus"]):
                    out.append({**base, "status": "placement_mismatch",
                                "reason": "physical placement differs"}); continue
                out.append({**base, **compare(m["task"], ic, cc)})
    return out

def public_campaign(camp):
    b = camp["build"]
    inst = {}
    for r in RT:
        e = camp["runtime_installations"].get(r) or {}
        inst[r] = {k: clean(e.get(k)) for k in
                   ("version", "package_tree_sha256", "record_sha256",
                    "record_mismatch_count", "unrecorded_count", "stock")
                   if e.get(k) is not None}
    return {"id": clean(camp["id"]), "date_local": clean(camp["date_local"]),
            "timezone": clean(camp["timezone"]), "source_commit": camp["commit"],
            "source_clean": True, "binary_sha256": camp["binary"],
            "runtime_versions": clean(camp["versions"]),
            "hardware": clean(camp["hardware"]), "method": clean(camp["method"]),
            "build": {"version": clean(b.get("version")),
                      "binary_sha256": clean(b.get("binary_sha256")),
                      "native_library_count": len(b.get("native_library_sha256") or {}),
                      "native_library_sha256": clean(b.get("native_library_sha256") or {}),
                      "command": clean(b.get("command")),
                      "providers": clean(b.get("providers") or {})},
            "runtime_installations": inst,
            "fingerprint_files": clean(camp["files"])}

def casepublic(c, x, cmp):
    s = c.get("sample") or {}; rec = c.get("record") or {}
    vals = rec.get("values")
    if rec.get("all_finite") is not True or vals is None:
        vals = None
    else:
        _, err = flat(vals)
        if err:
            vals = None
    return {"length": c["length"], "variant": None, "present": True,
            "clean": bool(c.get("clean")), "clean_reasons": [clean(r) for r in c["clean_reasons"]],
            "probe": {"valid": c["probe_valid"], "error": clean(c["probe_error"]),
                      "semantic_sha256": s.get("sha256"),
                      "values_sha256": sha(cjson(vals).encode()) if vals is not None else None,
                      "shape": clean(rec.get("shape")), "norm": num(rec.get("norm"))},
            "metrics": clean(c["metrics"]),
            "comparison": None if cmp is None else {k: clean(cmp.get(k)) for k in
                ("status", "numerical_pass", "reason", "method", "cosine", "norm_close",
                 "allclose", "max_abs_diff")},
            "fp32": None if not c.get("fp32") else {
                k: clean(c["fp32"].get(k)) for k in
                ("status", "fp32_pass", "cosine", "norm_close", "allclose",
                 "max_abs_diff", "fully_eligible", "fp32_baseline_eligible")},
            "raw": {"path": x["path"], "sha256": x["sha"]}}

def modelrecords(camp, runs, comps, models):
    ci = {(x["model"], x["runtime"], x["length"]): x for x in comps}
    out = []
    for m in sorted(models, key=lambda x: (-(x["parameters"] or 0), x["id"])):
        ats, clean_n = [], 0
        for r in RT:
            x = runs.get((m["id"], r))
            if x is None:
                ats.append({"runtime": r, "terminal_status": "missing", "checkpoint_revision": None,
                            "run_fingerprint": None, "gpu": None, "cpus": None, "client_cpus": None,
                            "startup_s": None, "error": None, "error_hint": None,
                            "failure_reason": "run missing",
                            "serial_probe": {"valid": False, "errors": ["missing"]},
                            "cases": [{"length": n, "variant": None, "present": False, "clean": False,
                                       "clean_reasons": ["run missing"], "probe": None,
                                       "metrics": None, "comparison": None, "raw": None}
                                      for n, _ in expected(m)]})
                continue
            cs = []
            for key in expected(m):
                c = x["cases"].get(key); cmp = ci.get((m["id"], r, key[0]))
                if c is None:
                    cs.append({"length": key[0], "variant": None, "present": False, "clean": False,
                               "clean_reasons": [f"case missing in {x['status']}"], "probe": None,
                               "metrics": None, "comparison": None if cmp is None else {
                                   "status": cmp["status"], "numerical_pass": cmp["numerical_pass"],
                                   "reason": clean(cmp.get("reason"))},
                               "raw": {"path": x["path"], "sha256": x["sha"]}})
                else:
                    clean_n += bool(c.get("clean"))
                    fp = c.get("fp32")
                    if fp is not None:
                        diag = not fp.get("fully_eligible")
                    else:
                        diag = not (c.get("clean") and (
                            r == "infinity" or (cmp or {}).get("numerical_pass") is True))
                    c["metrics"]["metrics_diagnostic"] = diag
                    cs.append(casepublic(c, x, cmp))
            ats.append({"runtime": r, "terminal_status": x["status"],
                        "checkpoint_revision": clean(x["revision"]),
                        "run_fingerprint": x["fingerprint"], "gpu": x["gpu"], "cpus": x["cpus"],
                        "client_cpus": x["client_cpus"], "startup_s": x["startup"],
                        "error": x["error"], "error_hint": x["error_hint"],
                        "failure_reason": x["failure_reason"],
                        "serial_probe": clean(x["serial"]), "cases": cs})
        out.append({"id": m["id"], "task": m["task"], "parameters": m["parameters"],
                    "selection": m["selection"], "lengths": m["lengths"],
                    "clean_case_count": clean_n, "attempts": ats})
    return out

def runstatus(camp, runs, models):
    out = {}
    for r in RT:
        c = {k: 0 for k in sorted(TERMINAL)}; c["missing"] = 0; clean = 0; exp = 0
        for m in models:
            exp += len(m["lengths"])
            x = runs.get((m["id"], r))
            if x is None:
                c["missing"] += 1
            else:
                c[x["status"]] += 1
                clean += sum(1 for z in x["cases"].values() if z.get("clean"))
        out[r] = {**c, "clean_cases": clean, "expected_cases": exp}
    return out

def complete(runs, models):
    exp = [(m["id"], r) for m in models for r in RT]
    miss = [k for k in exp if k not in runs]
    unknown = sorted(set(runs) - set(exp))
    tc = {}
    for k in runs:
        if k in exp:
            tc[runs[k]["status"]] = tc.get(runs[k]["status"], 0) + 1
    return {"expected_pairs": len(exp), "observed_pairs": len(exp) - len(miss),
            "missing_pairs": [{"model": a, "runtime": b} for a, b in miss],
            "unknown_pairs": [{"model": a, "runtime": b} for a, b in unknown],
            "terminal_status_counts": dict(sorted(tc.items())),
            "complete": not miss and not unknown}

def report(camp, models, runs, allow, refctx=None):
    comps = comparisons(camp, runs, models)
    comp = complete(runs, models)
    if not comp["complete"] and not allow:
        raise ReportError(f"campaign incomplete: {len(comp['missing_pairs'])} pairs missing; "
                  "use --allow-incomplete for PROVISIONAL")
    st = runstatus(camp, runs, models)
    fp32 = None
    if refctx is not None:
        fp32, _ = fp32_compare(refctx, camp, models, runs, comps)
    out = {"schema_version": 1, "kind": "infinity_embedding_benchmark_export",
           "status": "complete" if comp["complete"] else "provisional",
           "campaign": public_campaign(camp),
           "manifest_sha256": shafile(Path(camp["root"]) / "manifest.json"),
           "completeness": comp, "runtime_status": st,
           "models": modelrecords(camp, runs, comps, models), "comparisons": comps,
           "raw_runs": [{"path": x["path"], "sha256": x["sha"], "model": x["model"],
                         "runtime": x["runtime"], "status": x["status"]}
                        for _, x in sorted(runs.items())],
           "notes": ["No aggregate winner or ranking.",
                     "Token rates recomputed from per-window count==ok means.",
                     "Latency JSON/CSV values are medians of per-window p50/p95.",
                     "Evidence carries its own artifact checksums; raw SHA is provenance only.",
                     "BF16 agreement is not FP32 or task-quality validation."]}
    if fp32 is not None:
        out["independent_fp32"] = fp32
        out["notes"].append("Optional independent HF float32 smoke gates are additional; "
                            "they are not task-quality or ranking evidence.")
    return out

def evidence(camp, models, runs, refctx=None):
    out = []
    for m in models:
        for r in RT:
            x = runs.get((m["id"], r))
            if not x:
                continue
            for key in expected(m):
                c = x["cases"].get(key)
                if not c:
                    continue
                s = c.get("sample") or {}; rec = c.get("record") or {}
                vals, finite = None, False
                if rec.get("all_finite") is True and rec.get("values") is not None:
                    _, err = flat(rec["values"])
                    finite = err is None
                if finite:
                    vals = rec["values"]
                pr = {"shape": clean(rec.get("shape")), "norm": num(rec.get("norm")),
                      "response_format": clean(rec.get("response_format")),
                      "all_finite": rec.get("all_finite"),
                      "semantic_sha256": s.get("sha256"),
                      "provided_sha256": s.get("provided_sha256"),
                      "values": vals}
                if vals is None:
                    pr["values_omitted"] = clean(c.get("probe_error") or "nonfinite_or_unavailable")
                elif not c["probe_valid"]:
                    pr["values_diagnostic"] = clean(c.get("probe_error") or "probe_invalid")
                out.append({"schema_version": 1, "record_kind": "runtime_probe",
                            "model": m["id"], "task": m["task"],
                            "runtime": r, "length": key[0], "variant": None,
                            "raw": {"path": x["path"], "sha256": x["sha"]},
                            "semantic_input": None if not s else {"sha256": s["sha256"], "fields": s["fields"]},
                            "probe": pr,
                            "warmup": ledgerev(c.get("warmup")),
                            "windows": [ledgerev(w) for w in c.get("windows") or []],
                            "serial_probe": {k: clean(v) for k, v in (x["serial"] or {}).items()
                                             if k not in {"errors", "all_latencies_ms"}},
                            "evidence_note": "Portable subset: full available finite probes, exact "
                                             "semantic inputs, warm-up/window counters, latency "
                                             "aggregates with artifact checksums. Raw SHA is provenance "
                                             "only; full raw runs/logs are not published."})
    provenance = None
    if refctx is not None:
        provenance = {"schema_version": 1, "record_kind": "hf_fp32_reference_provenance",
                      **clean(refctx["provenance"]),
                      "reference_issues": clean(refctx["issues"]),
                      "reference_failures": clean(refctx["failures"])}
        for ref in refctx["references"].values():
            out.append({
                "schema_version": 1, "record_kind": "hf_fp32_reference",
                "model": ref["model"], "task": ref["task"], "length": ref["length"],
                "variant": None, "sample_sha256": ref["sample_sha256"],
                "local_tokens": ref["local_tokens"], "pooling": clean(ref.get("pooling")),
                "pooling_source": clean(ref.get("pooling_source")),
                "normalize": ref.get("normalize"),
                "normalization": clean(ref.get("normalization")),
                "values": ref["values"], "shape": clean(ref.get("shape")),
                "all_finite": ref.get("all_finite"), "dtype": ref.get("dtype"),
                "device": clean(ref.get("device")),
                "checkpoint": clean(ref.get("checkpoint")),
                "loading": clean(ref.get("loading")),
                "semantic_input": None if not ref.get("_semantic") else {
                    "sha256": ref["sample_sha256"], "fields": ref["_semantic"]},
                "evidence_note": "Independent HF float32 numerical smoke vector; not training, "
                                 "task-quality, or ranking evidence."})
    out.sort(key=lambda z: (z["record_kind"], z["model"], z.get("runtime", ""), z["length"]))
    body = "".join(cjson(z) + "\n" for z in out).encode()
    if provenance is not None:
        body = (cjson(provenance) + "\n").encode() + body
    return body

def ledgerev(x):
    if not isinstance(x, dict):
        return None
    keys = ("attempted", "responses", "ok", "errors", "unanswered", "elapsed_seconds",
            "throughput_rps", "latency_ms", "tokens_per_request", "token_count_comparison",
            "response_formats", "concurrency", "failure_count")
    return {k: clean(x.get(k)) for k in keys if k in x}

def csvtext(camp, models, runs, rep):
    ci = {(x["model"], x["runtime"], x["length"]): x for x in rep["comparisons"]}
    b = io.StringIO(); w = csv.writer(b, lineterminator="\n")
    fp32_mode = "independent_fp32" in rep
    header = ["model_id", "task", "runtime", "length", "terminal_status", "case_present",
              "case_clean", "clean_reasons", "concurrency", "attempted", "responses", "ok",
              "errors", "unanswered", "failure_count", "elapsed_seconds", "throughput_rps",
              "server_input_tokens_per_second", "local_input_tokens_per_second",
              "window_p50_median_ms", "window_p95_median_ms", "per_window_p50_ms",
              "per_window_p95_ms", "serial_min_ms", "serial_probe_valid", "startup_s",
              "run_fingerprint", "probe_valid", "semantic_sha256", "probe_shape",
              "comparison_status", "numerical_pass", "cosine", "norm_close", "allclose",
              "max_abs_diff", "metric_diagnostic", "failure_reason"]
    if fp32_mode:
        header += ["fp32_status", "fp32_cosine", "fp32_norm_close", "fp32_allclose",
                   "fp32_max_abs_diff", "fully_eligible"]
    w.writerow(header)
    for m in sorted(models, key=lambda z: (-(z["parameters"] or 0), z["id"])):
        for r in RT:
            x = runs.get((m["id"], r))
            for n in m["lengths"]:
                c = x["cases"].get((n, None)) if x else None
                cm = ci.get((m["id"], r, n)); mt = (c or {}).get("metrics") or {}
                fp = (c or {}).get("fp32") or {}
                if fp:
                    diag = not fp.get("fully_eligible")
                else:
                    diag = not (c and c.get("clean") and
                                (r == "infinity" or (cm or {}).get("numerical_pass") is True))
                if not x:
                    failure = "run missing"
                elif x["failure_reason"]:
                    failure = x["failure_reason"]
                elif c is None:
                    failure = f"{x['status']} case missing"
                else:
                    failure = ""
                row = [m["id"], m["task"], r, n, x["status"] if x else "missing",
                            bool(c), bool(c and c.get("clean")),
                            "; ".join(clean(z) for z in (c or {}).get("clean_reasons", [])),
                            (c or {}).get("concurrency"), mt.get("attempted"), mt.get("responses"),
                            mt.get("ok"), mt.get("errors"), mt.get("unanswered"),
                            mt.get("failure_count"), mt.get("elapsed_seconds"), mt.get("throughput_rps"),
                            mt.get("server_input_tokens_per_second"),
                            mt.get("local_input_tokens_per_second"), mt.get("window_p50_median_ms"),
                            mt.get("window_p95_median_ms"),
                            ";".join(str(v) for v in mt.get("per_window_p50_ms") or []),
                            ";".join(str(v) for v in mt.get("per_window_p95_ms") or []),
                            mt.get("serial_min_ms"), bool(x and x["serial"].get("valid")),
                            x["startup"] if x else None, x["fingerprint"] if x else None,
                            bool(c and c.get("probe_valid")),
                            ((c or {}).get("sample") or {}).get("sha256"),
                            ((c or {}).get("record") or {}).get("shape"),
                            (cm or {}).get("status"), bool((cm or {}).get("numerical_pass")),
                            (cm or {}).get("cosine"), (cm or {}).get("norm_close"),
                            (cm or {}).get("allclose"), (cm or {}).get("max_abs_diff"),
                            diag, failure]
                if fp32_mode:
                    row += [fp.get("status"), fp.get("cosine"), fp.get("norm_close"),
                            fp.get("allclose"), fp.get("max_abs_diff"),
                            fp.get("fully_eligible")]
                w.writerow(row)
    return b.getvalue()

def cell(x, c, cm, r):
    if x is None:
        return "missing"
    if c is None:
        return x["status"]
    mt = c.get("metrics") or {}
    tok, rps = mt.get("server_input_tokens_per_second"), mt.get("throughput_rps")
    s = ((f"{tok:,.2f} tok/s" if tok is not None else "tok/s n/a") + " / " +
         (f"{rps:,.2f} req/s" if rps is not None else "req/s n/a"))
    rel = ("baseline" if c.get("clean") else "diagnostic") if r == "infinity" else \
          ("diagnostic" if not c.get("clean") else (cm or {}).get("status", "no-comparison"))
    fp = c.get("fp32")
    if fp is not None:
        prefix = "ELIGIBLE" if fp.get("fully_eligible") else "DIAGNOSTIC"
        tag = f"{prefix} rel={rel} fp32={fp.get('status')}"
        if r == "infinity":
            tag += f" baseline={fp.get('fp32_baseline_eligible')}"
    else:
        tag = rel
    return f"{s} [{tag}]"

def markdown(camp, models, runs, rep):
    comp = rep["completeness"]; st = rep["runtime_status"]
    ci = {(x["model"], x["runtime"], x["length"]): x for x in rep["comparisons"]}
    hw = camp["hardware"]; b = camp["build"]
    vi = camp["runtime_installations"]["vllm"]; si = camp["runtime_installations"]["sglang"]
    lines = [f"# Infinity embedding benchmark export — {rep['status'].upper()}", "",
             "No aggregate winner or ranking. Failed/diagnostic/missing cells are not ranked.", "",
             ("Data: [results.json](results.json) · [results.csv](results.csv) · "
              "[evidence.jsonl.gz](evidence.jsonl.gz) · [artifacts.json](artifacts.json)."), "",
             "## Provenance", "",
             (f"- Campaign `{clean(camp['id'] or '')}`; source `{camp['commit']}` clean; "
              f"manifest SHA256 `{shafile(Path(camp['root']) / 'manifest.json')}`."),
             (f"- Recorded campaign date `{clean(camp['date_local'])}` local timezone "
              f"`{clean(camp['timezone'])}` (recorded campaign date, not generation clock)."),
             (f"- Infinity `{camp['versions']['infinity']}` build `{clean(b.get('version'))}`, "
              f"binary `{camp['binary']}`, native libraries `{len(b.get('native_library_sha256') or {})}`."),
             (f"- Stock runtimes: vLLM `{camp['versions']['vllm']}`, SGLang `{camp['versions']['sglang']}`."),
             (f"- Recorded packages: vLLM record `{vi.get('record_sha256')}` mismatches "
              f"`{vi.get('record_mismatch_count')}` unrecorded `{vi.get('unrecorded_count')}`; "
              f"SGLang record `{si.get('record_sha256')}` mismatches "
              f"`{si.get('record_mismatch_count')}` unrecorded `{si.get('unrecorded_count')}`."),
             (f"- Hardware: `{clean(hw.get('gpu_name'))}` {hw.get('gpu_memory_mib')} MiB; "
              f"`{clean(hw.get('cpu'))}`; {hw.get('server_cores_per_lane')} server cores/lane; "
              f"{hw.get('client_cores_per_lane')} client cores/lane; shared host `{hw.get('shared_host')}`."),
             (f"- Method: {WINDOWS}×{SECONDS:g}s after {WARMUP:g}s warm-up, batch "
              f"{camp['method'].get('request_batch_size', 1)}, dtype {DTYPE}."), "",
             "## Completeness", "",
             (f"- Expected pairs **{comp['expected_pairs']}**, observed **{comp['observed_pairs']}**, "
              f"missing **{len(comp['missing_pairs'])}**, complete **{comp['complete']}**."),
             f"- Terminal statuses `{comp['terminal_status_counts']}`.", "",
             "| Runtime | measured | client_failed | load_failed | checkpoint_unavailable | missing | clean |",
             "|---|---|---|---|---|---|---|"]
    for r in RT:
        x = st[r]
        lines.append(f"| {r} | {x['measured']} | {x['client_failed']} | {x['load_failed']} | "
                     f"{x['checkpoint_unavailable']} | {x['missing']} | "
                     f"{x['clean_cases']}/{x['expected_cases']} |")
    lines += ["", "## Full model × length matrix (larger parameters first)", "",
              ("Cells: recomputed server input tok/s / successful req/s and explicit tag. "
               "Latency is per-window p50/p95 medians in results.csv/results.json."), "",
              "| Model | Parameters (B) | Tokens | C | Infinity | vLLM | SGLang |",
              "|---|---|---|---|---|---|---|"]
    for m in sorted(models, key=lambda z: (-(z["parameters"] or 0), z["id"])):
        for n in sorted(m["lengths"]):
            x = runs.get((m["id"], "infinity"))
            cells = []
            for r in RT:
                rr = runs.get((m["id"], r))
                c = rr["cases"].get((n, None)) if rr else None
                cells.append(cell(rr, c, ci.get((m["id"], r, n)), r))
            p = f"{m['parameters'] / 1e9:,.2f}" if m["parameters"] is not None else ""
            tag = f" · {m['selection']}" if m.get("selection") else ""
            lines.append(f"| `{m['id']}` {m['task']}{tag} | {p} | {n} | {concurrency(camp, n)} | "
                         + " | ".join(cells) + " |")
    fp32 = rep.get("independent_fp32")
    if fp32 is not None:
        pv = fp32["provenance"]; cv = fp32["coverage"]
        lines += ["", "## Independent FP32 smoke coverage", "",
                  (f"- Reference SHA `{pv.get('reference_file_sha256')}`; helper "
                   f"`{pv.get('source_sha256')}`; client `{pv.get('client_sha256')}`; "
                   f"manifest `{pv.get('manifest_sha256')}`; device `{pv.get('device')}`, "
                   f"dtype `{pv.get('dtype')}`."),
                  (f"- References `{cv.get('reference_records')}`, issues "
                   f"`{cv.get('reference_issues')}`, failures `{cv.get('reference_failures')}`, "
                   f"runtime cases `{cv.get('runtime_cases')}`, FP32 passes "
                   f"`{cv.get('fp32_passes')}`, fully eligible `{cv.get('fully_eligible_cases')}`; "
                   f"statuses `{cv.get('status_counts')}`."),
                  ("- Independent HF float32 is an additional numerical smoke gate, not "
                   "task-quality, retrieval-quality, or ranking evidence."), ""]
    lines += ["", "## Diagnostics", "",
              f"- Comparison statuses `{dict(sorted({k: sum(1 for x in rep['comparisons'] if x['status'] == k) for k in {x['status'] for x in rep['comparisons']}}.items()))}`.",
              ("- Full per-case status, token rates and latency medians are in results.csv; "
               "normalized records are in results.json; probe evidence is in evidence.jsonl.gz."),
              ("- Evidence carries its own checksums and full available finite probes. Raw run "
               "SHA256 is provenance only; full raw runs/logs are not published."),
              ("- BF16 agreement is not FP32 validation or task-quality accuracy."), ""]
    return "\n".join(lines)

def outputs(root, rep, md, csvb, ev):
    files = {"results.json": (cjson(rep) + "\n").encode(), "results.md": md.encode(),
             "results.csv": csvb.encode(), "evidence.jsonl.gz": ev}
    for name, data in list(files.items()):
        txt = gzip.decompress(data).decode() if name.endswith(".gz") else data.decode()
        for bad in ("localhost", "127.0.0.1", "/root/", "/home/", "/tmp/", "/private/"):
            require(bad not in txt, f"{name} contains non-portable {bad}")
    Path(root).mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (Path(root) / name).write_bytes(data)
    art = {"schema_version": 1, "kind": "artifact_manifest",
           "artifacts": [{"path": n, "sha256": sha(d), "bytes": len(d)}
                         for n, d in sorted(files.items())]}
    (Path(root) / "artifacts.json").write_text(cjson(art) + "\n")
    return art

def export(root, out, allow=False, reference=None):
    camp = load_campaign(root); camp["root"] = str(root)
    mods = load_manifest(root, camp)["models"]
    runs = loadruns(camp, mods, root, allow)
    refctx = None
    if reference is not None:
        helper = Path(__file__).resolve().parent / "benchmark-fp32-reference.py"
        refctx = load_fp32(reference, camp, mods, runs, helper)
    rep = report(camp, mods, runs, allow, refctx)
    ev = gz(evidence(camp, mods, runs, refctx))
    md = markdown(camp, mods, runs, rep)
    art = outputs(out, rep, md, csvtext(camp, mods, runs, rep), ev)
    summary = {"status": rep["status"], "complete": rep["completeness"]["complete"],
               "models": len(mods), "expected_pairs": rep["completeness"]["expected_pairs"],
               "observed_pairs": rep["completeness"]["observed_pairs"],
               "missing_pairs": len(rep["completeness"]["missing_pairs"]),
               "clean_cases": sum(x["clean_cases"] for x in rep["runtime_status"].values()),
               "comparisons": len(rep["comparisons"]),
               "numerical_passes": sum(1 for x in rep["comparisons"] if x["numerical_pass"]),
               "artifacts": art["artifacts"], "json_bytes": len(cjson(rep))}
    if refctx is not None:
        summary["independent_fp32"] = rep["independent_fp32"]["coverage"]
    return summary

def fixture(root, *, short=False, client_failed=False, unavailable=False,
            broken_probe=False, missing_sample=False, usage=None, zero=False,
            nan=False, placement=False, order=False, serial_bad=False):
    root = Path(root); (root / "runs").mkdir(parents=True, exist_ok=True)
    models = [
        {"id": "acme/dense-a", "revision": "a" * 40, "task": "embeddings",
         "parameters": 7_500_000_000, "selection": "comparison_candidate",
         "new_checkpoint": False, "lengths": [128, 512, 2048]},
        {"id": "acme/rerank-a", "revision": "b" * 40, "task": "rerank",
         "parameters": 500_000_000, "selection": "diagnostic",
         "new_checkpoint": False, "lengths": [128, 512]},
    ]
    (root / "manifest.json").write_text(json.dumps(
        {"schema_version": 1, "campaign": "fixture", "models": models}, indent=2) + "\n")
    msha = shafile(root / "manifest.json")
    files = {
        "infinity": {"runner_sha256": "1" * 64, "client_sha256": "2" * 64,
                     "binary_sha256": "a" * 64,
                     "native_library_sha256": {"libinfinity.so": "3" * 64},
                     "runtime_package_tree_sha256": None},
        "vllm": {"runner_sha256": "1" * 64, "client_sha256": "2" * 64,
                 "binary_sha256": "b" * 64,
                 "native_library_sha256": {"libinfinity.so": "3" * 64},
                 "runtime_package_tree_sha256": "4" * 64},
        "sglang": {"runner_sha256": "1" * 64, "client_sha256": "2" * 64,
                   "binary_sha256": "c" * 64,
                   "native_library_sha256": {"libinfinity.so": "3" * 64},
                   "runtime_package_tree_sha256": "5" * 64},
    }
    lanes = [{"gpu": 0, "server_cpus": "0-7", "client_cpus": "32-39"},
             {"gpu": 1, "server_cpus": "8-15", "client_cpus": "40-47"}]
    camp = {"schema_version": 1, "id": "fixture", "date_local": "2026-11-08",
            "timezone": "America/Los_Angeles", "source_commit": FIXTURE_SOURCE_COMMIT,
            "source_clean": True, "binary_sha256": "a" * 64, "manifest_sha256": msha,
            "runtime_versions": {"infinity": "3.0.2-beta.2", "vllm": "0.33.0",
                                 "sglang": "0.5.23"},
            "hardware": {"gpu_name": "NVIDIA B200", "gpu_memory_mib": 1,
                         "cpu": "fixture", "server_cores_per_lane": 8,
                         "client_cores_per_lane": 8, "shared_host": True,
                         "gpu_lanes": lanes},
            "method": {"runtimes": list(RT), "seconds": 10, "repeats": 3, "warmup": 3,
                       "concurrency": 32, "long_input_concurrency": 8,
                       "max_batch_tokens": 16384, "max_sequence_length": 2048,
                       "request_batch_size": 1, "dtype": "bfloat16",
                       "prefix_cache": False, "radix_cache": False,
                       "infinity_radix_mlp_threshold": 0, "lanes": lanes},
            "fingerprint_files": files,
            "build": {"version": "3.0.2-beta.2", "binary_sha256": "a" * 64,
                      "native_library_sha256": {"libinfinity.so": "3" * 64}},
            "runtime_installations": {
                "infinity": {"version": "3.0.2-beta.2", "source_clean": True,
                             "package_tree_sha256": "6" * 64},
                "vllm": {"version": "0.33.0", "package_tree_sha256": "4" * 64,
                         "record_sha256": "7" * 64, "record_mismatches": [],
                         "unrecorded_files": []},
                "sglang": {"version": "0.5.23", "package_tree_sha256": "5" * 64,
                           "record_sha256": "8" * 64, "record_mismatches": [],
                           "unrecorded_files": []}}}
    (root / "campaign.json").write_text(json.dumps(camp, indent=2) + "\n")

    def led(c, start=0):
        ok = 10 + start
        return {"attempted": ok, "responses": ok, "ok": ok, "errors": 0, "unanswered": 0,
                "elapsed_seconds": 10.0,
                "throughput_rps": ok / 10.0,
                "latency_ms": {"p50": 1.0 + start, "p95": 2.0 + start, "samples": ok},
                "tokens_per_request": {"local": {"count": ok, "mean": 8.0},
                                       "usage_prompt": {"count": ok, "mean": 8.0},
                                       "usage_total": {"count": ok, "mean": 8.0},
                                       "usage_input": None, "target": 8},
                "token_count_comparison": {"compared": ok, "equal": ok, "different": 0,
                                           "usage_prompt_unavailable": 0},
                "response_formats": ["fixture"], "uuid_requests": 0, "concurrency": c,
                "failure_count": 0, "failure_samples": []}

    for model in models:
        for r in RT:
            st = "measured"
            if client_failed and r == "vllm": st = "client_failed"
            if unavailable and r == "sglang": st = "checkpoint_unavailable"
            rev = None if st == "checkpoint_unavailable" else model["revision"]
            cp = {"id": model["id"], "status": "ready" if rev else "unavailable",
                  "revision": rev, "path": "/root/private/checkpoint",
                  "metadata_sha256": "" if rev is None else "9" * 64,
                  "code_dependencies": [{"id": "dep", "revision": "8" * 40,
                                         "path": "/root/private/dep"}]}
            context = None if rev is None else (2049 if r == "sglang" else 2048)
            settings = {"dtype": "bfloat16", "seconds": 10.0, "repeats": 3, "warmup": 3.0,
                        "concurrency": 32, "long_input_concurrency": 8,
                        "max_batch_tokens": 16384, "max_sequence_length": 2048,
                        "lengths": model["lengths"], "context": context,
                        "input_context": None if context is None else 2048,
                        "runtime_environment": (
                            {"INFINITY_TORCH_CUDA_GRAPHS": "1",
                             "INFINITY_TORCH_PREFIX_CACHE_SIZE": "0"} if r == "infinity"
                            else {"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1"}
                            if r == "sglang" else {})}
            if placement and r != "infinity":
                lane_gpu, lane_cpus, lane_client = 1, "8-15", "40-47"
            else:
                lane_gpu, lane_cpus, lane_client = 0, "0-7", "32-39"
            fpv = {"model": portable(model), "runtime": r,
                   "checkpoint": {"id": model["id"], "revision": rev,
                                  "metadata_sha256": cp["metadata_sha256"],
                                  "code_dependencies": [{"id": "dep", "revision": "8" * 40}]},
                   "lane": {"gpu": lane_gpu, "cpus": lane_cpus,
                            "client_cpus": lane_client},
                   "settings": settings, "files": files[r]}
            command = (["taskset", "-c", lane_cpus, "/root/private/server", "--dtype", "bfloat16"]
                       if st != "checkpoint_unavailable" else [])
            measurement = None
            client_command = []
            if st != "checkpoint_unavailable":
                if r == "infinity":
                    command += ["--max-batch-tokens", "16384", "--max-sequence-length",
                                "2048", "--radix-mlp-threshold", "0"]
                elif r == "vllm":
                    command += ["--max-num-batched-tokens", "16384", "--max-model-len",
                                "2048", "--no-enable-prefix-caching", "--runner", "pooling"]
                    if model["task"] == "embeddings":
                        command += ["--pooler-config", '{"use_activation": true}']
                else:
                    command += ["--max-prefill-tokens", "16384", "--context-length",
                                "2049", "--disable-radix-cache"]
                client_command = ["taskset", "-c", fpv["lane"]["client_cpus"],
                                  "/root/private/client", "--runtime", r, "--model", "benchmark",
                                  "--task", model["task"], "--lengths",
                                  ",".join(str(n) for n in model["lengths"]),
                                  "--concurrency", "32", "--seconds", "10", "--repeats",
                                  "3", "--warmup", "3", "--output", "/root/private/out"]
                workloads = []
                for n in model["lengths"]:
                    c = concurrency({"method": {"long": 8, "concurrency": 32}}, n)
                    if model["task"] == "embeddings":
                        fields = {"text": f"text {model['id']} {n}"}
                        vals = [0.1, 0.2, -0.3, 0.4 + (0.001 if r != "infinity" else 0)]
                        shape = [4]
                    else:
                        fields = {"query": f"q {model['id']} {n}",
                                  "document": f"d {model['id']} {n}"}
                        vals = [0.0] if zero else [0.5]
                        shape = [1]
                        if order and r != "infinity":
                            vals = [-0.5]
                    h = sha(json.dumps(fields, sort_keys=True).encode())
                    sample = {**fields, "sha256": h, "local_tokens": n,
                              "expected_rows": 1, "body_sha256": "e" * 64}
                    if nan and r == "vllm" and n == model["lengths"][0] and model["task"] == "embeddings":
                        vals[0] = float("nan")
                    wrec = {"shape": shape, "values": vals, "norm": norm([x for x in vals if math.isfinite(x)]),
                            "all_finite": not any(not math.isfinite(x) for x in vals),
                            "response_format": "fixture"}
                    probe = {"sample": sample, "record": wrec, "failures": [], "failure_count": 0}
                    if broken_probe and r == "vllm" and n == model["lengths"][0] and model["task"] == "embeddings":
                        probe["failure_count"] = 1
                        probe["failures"] = [{"kind": "fixture"}]
                    if missing_sample and r == "vllm" and n == model["lengths"][0] and model["task"] == "embeddings":
                        probe.pop("sample")
                    wl = {"length": n, "variant": None, "concurrency": c,
                          "descriptor": {"task": model["task"], "length": n, "concurrency": c,
                                         "samples": [dict(sample)]},
                          "warmup": {**led(c), "concurrency": c},
                          "windows": [{**led(c, i), "concurrency": c} for i in range(3)],
                          "numerical_probe": probe}
                    if short and r != "infinity":
                        wl["warmup"]["elapsed_seconds"] = 1.0
                        wl["windows"][0]["elapsed_seconds"] = 5.0
                    if usage and r == "vllm" and n == model["lengths"][0] and model["task"] == "embeddings":
                        for w in wl["windows"]:
                            w["tokens_per_request"]["usage_input"] = {"count": w["ok"], "mean": 9.0}
                        if usage == "prompt_incomplete_input_complete":
                            wl["windows"][0]["tokens_per_request"]["usage_prompt"]["count"] -= 1
                        elif usage == "both_incomplete":
                            wl["windows"][0]["tokens_per_request"]["usage_prompt"]["count"] -= 1
                            wl["windows"][0]["tokens_per_request"]["usage_input"]["count"] -= 1
                    workloads.append(wl)
                latency = {"attempted": 20, "responses": 20, "ok": 20, "errors": 0,
                           "unanswered": 0, "elapsed_seconds": 0.5,
                           "latency_ms": {"p50": 1.0, "p95": 2.0, "samples": 20},
                           "failure_count": 0, "failure_samples": [],
                           "min_ms": 1.0, "all_latencies_ms": [1.0 + i for i in range(20)],
                           "serial_requests": 20, "length": 8, "concurrency": 1}
                if serial_bad and r == "vllm":
                    latency["all_latencies_ms"] = latency["all_latencies_ms"][:-1]
                measurement = {"ok": st == "measured", "arguments": {
                    "runtime": r, "task": model["task"], "model": "benchmark",
                    "lengths": model["lengths"], "concurrency": 32, "seconds": 10,
                    "repeats": 3, "warmup": 3, "min_latency_length": 8,
                    "self_test": False}, "runtime": {"requested": r, "model": "benchmark",
                    "task": model["task"]}, "workloads": workloads,
                    "latency_probe": latency, "failures": []}
                if st == "client_failed":
                    measurement["workloads"] = measurement["workloads"][:-1]
                    measurement["failures"] = [{"phase": "window_1"}]
            doc = {"model": model["id"], "task": model["task"], "runtime": r,
                   "gpu": lane_gpu, "cpus": lane_cpus,
                   "client_cpus": lane_client, "dtype": "bfloat16",
                   "checkpoint": cp, "command": command, "client_command": client_command,
                   "environment_overrides": {"CUDA_VISIBLE_DEVICES": str(lane_gpu),
                                             **settings["runtime_environment"],
                                             "LD_LIBRARY_PATH_injected": ["/root/private/lib"]},
                   "fingerprint_inputs": fpv, "run_fingerprint": fp(fpv),
                   "startup_s": 1.0, "status": st}
            if measurement is not None:
                doc["measurement"] = measurement
            if st != "measured":
                doc["error"] = "RuntimeError: fixture failure at /root/private/path\nTraceback details hidden"
            name = f"{model['id'].replace('/', '--')}--{r}.json"
            (root / "runs" / name).write_text(json.dumps(doc) + "\n")
    return root

def prep(**kw):
    t = tempfile.TemporaryDirectory(prefix="report-")
    root, out = Path(t.name) / "c", Path(t.name) / "o"
    fixture(root, **kw)
    return t, root, out

def dense(rep, r, n):
    m = next(x for x in rep["models"] if x["id"] == "acme/dense-a")
    a = next(x for x in m["attempts"] if x["runtime"] == r)
    return next(x for x in a["cases"] if x["length"] == n)

def make_fp32_ref(root, *, omit=(), mutate=None, failures=None, duplicate=False):
    manifest = json.loads((root / "manifest.json").read_text())
    camp = json.loads((root / "campaign.json").read_text())
    docs = [json.loads(p.read_text()) for p in sorted((root / "runs").glob("*.json"))]
    helper = shafile(Path(__file__).resolve().parent / "benchmark-fp32-reference.py")
    client = camp["fingerprint_files"]["infinity"]["client_sha256"]
    msha = shafile(root / "manifest.json")
    refs = []
    for model in manifest["models"]:
        for n in model["lengths"]:
            if (model["id"], n) in omit:
                continue
            found = None
            for d in docs:
                if d.get("model") != model["id"] or not isinstance(d.get("measurement"), dict):
                    continue
                for w in d["measurement"].get("workloads", []):
                    if w.get("length") == n:
                        found = (d, w)
                        break
                if found:
                    break
            if not found:
                continue
            d, w = found
            probe = w["numerical_probe"]
            rec = probe["record"]
            cp = d["checkpoint"]
            fcp = d["fingerprint_inputs"]["checkpoint"]
            item = {"model": model["id"], "revision": model["revision"],
                    "task": model["task"], "length": n,
                    "sample_sha256": probe["sample"]["sha256"],
                    "local_tokens": probe["sample"]["local_tokens"],
                    "pooling": "mean" if model["task"] == "embeddings" else "none",
                    "pooling_source": "1_Pooling/config.json" if model["task"] == "embeddings" else None,
                    "normalize": model["task"] == "embeddings",
                    "normalization": "l2" if model["task"] == "embeddings" else "none",
                    "values": rec["values"], "shape": rec["shape"],
                    "all_finite": True, "dtype": "float32", "device": "cpu",
                    "loading": {"model_class": "fixture.Model", "validated": True,
                                "missing_keys": [], "unexpected_keys": [],
                                "mismatched_keys": [], "error_msgs": [],
                                "allowed_unused_pooler_keys": []},
                    "checkpoint": {"id": cp["id"], "revision": cp["revision"],
                                   "status": "ready", "receipt_sha256": "a" * 64,
                                   "metadata_sha256": cp["metadata_sha256"],
                                   "code_dependencies": deps(fcp.get("code_dependencies"), "fixture fp deps")}}
            if item["task"] == "rerank":
                item.pop("pooling_source")
            if mutate:
                mutate(item)
            refs.append(item)
    fails = []
    for mid, n in omit:
        m = next(x for x in manifest["models"] if x["id"] == mid)
        fails.append({"model": mid, "revision": m["revision"], "task": m["task"],
                      "length": n, "status": "unavailable",
                      "reason": "AttributeError: fixture unavailable at /root/private/path"})
    if failures:
        fails.extend(failures)
    if duplicate and refs:
        extra = json.loads(json.dumps(refs[0]))
        if extra["task"] == "embeddings":
            extra["values"] = [0.9] * len(extra["values"])
        else:
            extra["values"] = [0.9]
        refs.append(extra)
    doc = {"schema_version": 1, "kind": "independent_hf_fp32_numerical_smoke_reference",
           "method": "fixture independent HF float32", "source_sha256": helper,
           "client_sha256": client, "manifest_sha256": msha,
           "torch_version": "fixture", "transformers_version": "fixture",
           "device": "cpu", "threads": 8, "dtype": "float32",
           "source_hashes": {"reference_source": helper, "http_client_helper": client,
                             "manifest": msha},
           "input_layout": {"manifest": "manifest.json", "checkpoints": "checkpoints"},
           "model_counts": {"selected": len(manifest["models"]), "embeddings": 1, "rerank": 1},
           "counts": {"references": len(refs), "failures": len(fails),
                      "models_attempted": len(manifest["models"])},
           "references": refs, "failures": fails}
    path = root / "fp32-reference.json"
    path.write_text(json.dumps(doc))
    return path

def mutate(root, kind):
    cp = root / "campaign.json"
    if kind == "dirty":
        d = json.loads(cp.read_text()); d["source_clean"] = False; cp.write_text(json.dumps(d))
    elif kind == "malformed_commit":
        d = json.loads(cp.read_text()); d["source_commit"] = "53dd"; cp.write_text(json.dumps(d))
    elif kind == "manifest_pin":
        d = json.loads(cp.read_text()); d["manifest_sha256"] = "0" * 64; cp.write_text(json.dumps(d))
    elif kind == "runtime_set":
        d = json.loads(cp.read_text()); d["method"]["runtimes"] = ["infinity", "vllm", "other"]; cp.write_text(json.dumps(d))
    elif kind == "anchor_missing":
        d = json.loads(cp.read_text()); d.pop("fingerprint_files"); cp.write_text(json.dumps(d))
    elif kind == "anchor_tamper":
        d = json.loads(cp.read_text()); d["fingerprint_files"]["vllm"]["binary_sha256"] = "f" * 64; cp.write_text(json.dumps(d))
    elif kind == "duplicate":
        p = root / "runs" / "acme--dense-a--infinity.json"
        (root / "runs" / "dup.json").write_text(p.read_text())
    elif kind == "status":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["status"] = "weird"; p.write_text(json.dumps(d))
    elif kind == "malformed":
        (root / "runs" / "bad.json").write_text("{not json")
    elif kind == "hash":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["measurement"]["workloads"][0]["numerical_probe"]["sample"]["sha256"] = "e" * 64; p.write_text(json.dumps(d))
    elif kind == "concurrency":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["measurement"]["workloads"][0]["concurrency"] = 16; p.write_text(json.dumps(d))
    elif kind == "control":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["measurement"]["arguments"]["mystery"] = 1; p.write_text(json.dumps(d))
    elif kind == "pooler":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text())
        d["command"][d["command"].index("--pooler-config") + 1] = '{"use_activation": false}'
        p.write_text(json.dumps(d))
    elif kind == "settings":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["fingerprint_inputs"]["settings"]["warmup"] = 9.0
        d["run_fingerprint"] = fp(d["fingerprint_inputs"]); p.write_text(json.dumps(d))
    elif kind == "fp":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["run_fingerprint"] = "e" * 64; p.write_text(json.dumps(d))
    elif kind == "deps":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text()); d["fingerprint_inputs"]["checkpoint"]["code_dependencies"] = []
        d["run_fingerprint"] = fp(d["fingerprint_inputs"]); p.write_text(json.dumps(d))
    elif kind == "env_empty":
        p = root / "runs" / "acme--dense-a--infinity.json"
        d = json.loads(p.read_text()); d["fingerprint_inputs"]["settings"]["runtime_environment"] = {}
        d["environment_overrides"] = {}
        d["run_fingerprint"] = fp(d["fingerprint_inputs"]); p.write_text(json.dumps(d))
    elif kind == "latency_samples":
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text())
        d["measurement"]["workloads"][0]["windows"][0]["latency_ms"]["samples"] += 1
        p.write_text(json.dumps(d))
    elif kind == "stock_record":
        p = root / "campaign.json"; d = json.loads(p.read_text())
        d["runtime_installations"]["vllm"]["record_mismatches"] = ["changed.py"]
        p.write_text(json.dumps(d))
    elif kind == "budget":
        p = root / "campaign.json"; d = json.loads(p.read_text())
        d["method"]["max_batch_tokens"] = 8192; p.write_text(json.dumps(d))
    elif kind in ("lane_gpu", "lane_client"):
        p = root / "runs" / "acme--dense-a--vllm.json"
        d = json.loads(p.read_text())
        if kind == "lane_gpu":
            d.update(gpu=2, cpus="16-23", client_cpus="48-55")
            d["environment_overrides"]["CUDA_VISIBLE_DEVICES"] = "2"
            d["fingerprint_inputs"]["lane"].update(gpu=2, cpus="16-23", client_cpus="48-55")
        else:
            d["client_cpus"] = "200-207"
            d["fingerprint_inputs"]["lane"]["client_cpus"] = "200-207"
        d["run_fingerprint"] = fp(d["fingerprint_inputs"])
        p.write_text(json.dumps(d))
    else:
        raise AssertionError(kind)

def selftest():
    tests = []

    def add(fn):
        tests.append(fn)
        return fn

    @add
    def complete_export_matrix_anchor_and_size():
        t, root, out = prep()
        with t:
            s = export(root, out)
            assert s["status"] == "complete" and s["numerical_passes"] == 10, s
            assert s["json_bytes"] < 500_000
            assert dense(json.loads((out / "results.json").read_text()), "vllm", 128)["metrics"]["metrics_diagnostic"] is False
            assert dense(json.loads((out / "results.json").read_text()), "infinity", 128)["metrics"]["metrics_diagnostic"] is False
            md = (out / "results.md").read_text(); rows = [x for x in md.splitlines() if x.startswith("| `acme/")]
            assert len(rows) == 5 and rows[0].startswith("| `acme/dense-a`"), rows
            assert "Parameters (B)" in md and "[baseline]" in md and "[pass]" in md
            assert "tok/s" in md and "req/s" in md
            assert "[results.json](results.json)" in md
            assert "Recorded campaign date `2026-11-08` local timezone `America/Los_Angeles`" in md
            d = json.loads((root / "campaign.json").read_text())
            assert d["fingerprint_files"]["infinity"]["runtime_package_tree_sha256"] is None
            assert len({d["fingerprint_files"][r]["binary_sha256"] for r in RT}) == 3
            run = json.loads((root / "runs" / "acme--dense-a--vllm.json").read_text())
            assert "--min-latency-length" not in run["client_command"]
            assert "path" in run["checkpoint"]["code_dependencies"][0]

    @add
    def portable_build_provenance_is_published():
        t, root, out = prep()
        with t:
            path = root / "campaign.json"
            campaign = json.loads(path.read_text())
            campaign["build"]["command"] = 'LIBTORCH="$LIBTORCH" cargo build --release'
            campaign["build"]["providers"] = {
                "fa4": {"manifest_sha256": "d" * 64,
                        "manifest": {"abi_version": 7, "architecture": "sm_100a"}},
                "gdn": {"provenance": {"upstream_revision": "e" * 40}}}
            path.write_text(json.dumps(campaign))
            export(root, out)
            build = json.loads((out / "results.json").read_text())["campaign"]["build"]
            for key in ("command", "providers", "native_library_sha256"):
                assert build[key] == campaign["build"][key], key
            assert build["native_library_count"] == len(build["native_library_sha256"])

    @add
    def provisional_and_default_reject():
        t, root, out = prep()
        with t:
            (root / "runs" / "acme--dense-a--sglang.json").unlink()
            try:
                export(root, out)
            except ReportError as e:
                assert "incomplete" in str(e)
            else:
                raise AssertionError("incomplete accepted")
            s = export(root, out, True)
            assert s["status"] == "provisional" and s["missing_pairs"] == 1
            assert len((out / "results.csv").read_text().splitlines()) == 1 + 3 * 5

    @add
    def usage_weighted_fallback_none():
        for mode, want in (("prompt_incomplete_input_complete", 9.9), ("both_incomplete", None)):
            t, root, out = prep(usage=mode)
            with t:
                export(root, out)
                m = dense(json.loads((out / "results.json").read_text()), "vllm", 128)["metrics"]
                if want is None:
                    assert m["server_input_tokens_per_second"] is None
                else:
                    assert math.isclose(m["server_input_tokens_per_second"], want)
                assert math.isclose(m["local_input_tokens_per_second"], 8.8)
        t, root, out = prep()
        with t:
            export(root, out)
            m = dense(json.loads((out / "results.json").read_text()), "vllm", 128)["metrics"]
            assert math.isclose(m["server_input_tokens_per_second"], 8.8)

    @add
    def partial_probe_absence_and_finite_evidence():
        t, root, out = prep(broken_probe=True, client_failed=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            c = dense(rep, "vllm", 128)
            assert c["present"] and not c["clean"] and c["comparison"]["status"] == "candidate_not_eligible"
            ev = [json.loads(x) for x in gzip.decompress((out / "evidence.jsonl.gz").read_bytes()).splitlines()]
            e = next(x for x in ev if x["model"] == "acme/dense-a" and x["runtime"] == "vllm" and x["length"] == 128)
            assert isinstance(e["probe"]["values"], list) and e["probe"]["values_diagnostic"]
        t, root, out = prep(missing_sample=True, client_failed=True)
        with t:
            export(root, out)
            c = dense(json.loads((out / "results.json").read_text()), "vllm", 128)
            assert "probe sample missing" in "; ".join(c["clean_reasons"])
            assert c["probe"]["semantic_sha256"] is None
            att = next(a for m in json.loads((out / "results.json").read_text())["models"]
                       if m["id"] == "acme/dense-a" for a in m["attempts"] if a["runtime"] == "vllm")
            assert att["failure_reason"] and "/root/" not in att["failure_reason"]

    @add
    def nonfinite_and_zero_and_shape_gates():
        t, root, out = prep(nan=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            assert not dense(rep, "vllm", 128)["clean"]
            ev = gzip.decompress((out / "evidence.jsonl.gz").read_bytes()).decode()
            assert "NaN" not in ev and "Infinity" not in ev
        t, root, out = prep(zero=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            m = next(x for x in rep["models"] if x["id"] == "acme/rerank-a")
            assert all(c["clean"] for a in m["attempts"] for c in a["cases"])
        rec = {"shape": [1], "values": [0], "all_finite": True, "norm": 0}
        assert probe("rerank", rec)[1] is None
        assert probe("embeddings", rec)[1] == "zero_norm"
        assert probe("embeddings", {"shape": [2, 2], "values": [[1, 2], [3, 4]],
                                     "all_finite": True, "norm": norm([1, 2, 3, 4])})[1] == "dense_matrix_shape"
        assert probe("embeddings", {"shape": [4], "values": [[1, 2], [3]],
                                     "all_finite": True, "norm": norm([1, 2, 3])})[1] == "ragged"

    @add
    def short_windows_and_serial_are_diagnostic():
        t, root, out = prep(short=True, serial_bad=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            c = dense(rep, "vllm", 128)
            assert not c["clean"] and "warmup too short" in "; ".join(c["clean_reasons"])
            assert "window 0 too short" in "; ".join(c["clean_reasons"])
            assert c["metrics"]["serial_min_ms"] is None
            assert c["metrics"]["metrics_diagnostic"] is True

    @add
    def latency_sample_counter_is_diagnostic():
        t, root, out = prep()
        with t:
            mutate(root, "latency_samples")
            export(root, out)
            c = dense(json.loads((out / "results.json").read_text()), "vllm", 128)
            assert not c["clean"] and "latency samples invalid" in "; ".join(c["clean_reasons"])

    @add
    def placement_and_order_mismatch():
        t, root, out = prep(placement=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            assert any(x["status"] == "placement_mismatch" for x in rep["comparisons"])
        t, root, out = prep(order=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            rr = [x for x in rep["comparisons"] if x["task"] == "rerank"]
            assert rr and all(x["status"] == "fail_values" for x in rr)

    @add
    def unavailable_checkpoint_and_portability():
        t, root, out = prep(unavailable=True)
        with t:
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            assert rep["runtime_status"]["sglang"]["checkpoint_unavailable"] == 2
            a = next(x for m in rep["models"] for x in m["attempts"] if x["runtime"] == "sglang")
            assert a["checkpoint_revision"] is None and all(not c["present"] for c in a["cases"])
            assert a["failure_reason"] and "/root/" not in a["failure_reason"]
            first = (out / "evidence.jsonl.gz").read_bytes()
            out2 = Path(t.name) / "o2"; export(root, out2)
            assert first == (out2 / "evidence.jsonl.gz").read_bytes()
            assert first[4:8] == b"\x00\x00\x00\x00"
            for p in out.iterdir():
                txt = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
                for bad in ("localhost", "127.0.0.1", "/root/", "/private/"):
                    assert bad not in txt, (p.name, bad)
            art = json.loads((out / "artifacts.json").read_text())
            for x in art["artifacts"]:
                assert shafile(out / x["path"]) == x["sha256"]

    rejects = [
        ("dirty", "source_clean"), ("malformed_commit", "source_commit"),
        ("manifest_pin", "manifest_sha256"), ("runtime_set", "runtimes"),
        ("anchor_missing", "fingerprint_files"), ("anchor_tamper", "anchor"),
        ("duplicate", "duplicate"), ("status", "unknown status"),
        ("malformed", "invalid JSON"), ("hash", "sha256 mismatch"),
        ("concurrency", "concurrency"), ("control", "unknown") if False else ("control", "control"),
        ("pooler", "pooler"), ("settings", "settings"), ("fp", "run_fingerprint"),
        ("deps", "dependency"), ("env_empty", "runtime_environment policy"),
        ("stock_record", "stock claim rejected"), ("budget", "max_batch_tokens must be 16384"),
        ("lane_gpu", "declared campaign.method.lanes"),
        ("lane_client", "declared campaign.method.lanes"),
    ]
    for kind, want in rejects:
        @add
        def reject_case(kind=kind, want=want):
            t, root, out = prep()
            with t:
                mutate(root, kind)
                try:
                    export(root, out, True)
                except ReportError as e:
                    msg = str(e)
                else:
                    raise AssertionError(f"{kind} accepted")
                assert want in msg, (kind, want, msg)

    @add
    def fp32_valid_dense_zero_rerank_evidence_and_checksums():
        t, root, out = prep(zero=True)
        with t:
            ref = make_fp32_ref(root)
            s = export(root, out, reference=ref)
            assert "independent_fp32" in s and s["independent_fp32"]["fp32_passes"] > 0
            assert s["independent_fp32"]["fully_eligible_cases"] > 0
            rep = json.loads((out / "results.json").read_text())
            assert "values" not in cjson(rep["independent_fp32"])
            assert dense(rep, "vllm", 128)["fp32"]["status"] == "pass"
            assert dense(rep, "vllm", 128)["fp32"]["fully_eligible"] is True
            assert dense(rep, "vllm", 128)["metrics"]["metrics_diagnostic"] is False
            model = next(x for x in rep["models"] if x["id"] == "acme/rerank-a")
            assert all(c["fp32"]["status"] == "pass" for a in model["attempts"]
                       for c in a["cases"] if c["present"])
            ev = [json.loads(x) for x in gzip.decompress(
                (out / "evidence.jsonl.gz").read_bytes()).splitlines()]
            assert any(x["record_kind"] == "hf_fp32_reference_provenance" for x in ev)
            refs = [x for x in ev if x["record_kind"] == "hf_fp32_reference"]
            assert refs and all(isinstance(x["values"], list) and x["values"] for x in refs)
            assert all(x["loading"]["validated"] is True and
                       x["loading"]["model_class"] == "fixture.Model" for x in refs)
            art = json.loads((out / "artifacts.json").read_text())
            for x in art["artifacts"]:
                assert shafile(out / x["path"]) == x["sha256"]
            for x in refs:
                assert "/root/" not in cjson(x)

    @add
    def fp32_invalid_records_are_diagnostic():
        mutations = [
            lambda x: x.__setitem__("revision", "c" * 40),
            lambda x: x.__setitem__("sample_sha256", "e" * 64),
            lambda x: x.__setitem__("shape", [2]),
            lambda x: x["checkpoint"].__setitem__("metadata_sha256", "0" * 64),
            lambda x: x["checkpoint"].__setitem__("status", "unavailable"),
            lambda x: x.__setitem__("dtype", "float64"),
            lambda x: x.pop("loading"),
            lambda x: x["loading"].__setitem__("validated", False),
        ]
        for fn in mutations:
            t, root, out = prep()
            with t:
                ref = make_fp32_ref(root, mutate=fn)
                export(root, out, reference=ref)
                rep = json.loads((out / "results.json").read_text())
                case = dense(rep, "vllm", 128)
                assert case["fp32"]["status"] != "pass", case["fp32"]
                assert case["metrics"]["metrics_diagnostic"] is True

    @add
    def fp32_numeric_failures_are_separate():
        def bad_dense(x):
            if x["task"] == "embeddings":
                x["values"] = [0.9, -0.9, 0.9, -0.9]
        def bad_rerank(x):
            if x["task"] == "rerank":
                x["values"] = [0.9]
        def bad_norm(x):
            if x["task"] == "embeddings":
                x["values"] = [1.5 * v for v in x["values"]]
        for fn, want in ((bad_dense, "fail_cosine"), (bad_norm, "fail_norm"),
                         (bad_rerank, "fail_values")):
            t, root, out = prep()
            with t:
                ref = make_fp32_ref(root, mutate=fn)
                export(root, out, reference=ref)
                rep = json.loads((out / "results.json").read_text())
                if want in ("fail_cosine", "fail_norm"):
                    case = dense(rep, "vllm", 128)
                else:
                    model = next(x for x in rep["models"] if x["id"] == "acme/rerank-a")
                    case = next(c for a in model["attempts"] if a["runtime"] == "vllm"
                                for c in a["cases"] if c["present"])
                assert case["fp32"]["status"] == want, case["fp32"]

    @add
    def fp32_missing_failure_is_diagnostic_not_completeness_change():
        t, root, out = prep()
        with t:
            export(root, out)
            base = json.loads((out / "results.json").read_text())
            ref = make_fp32_ref(root, omit=(("acme/dense-a", 128),))
            out2 = Path(t.name) / "o2"
            export(root, out2, reference=ref)
            rep = json.loads((out2 / "results.json").read_text())
            assert rep["completeness"]["complete"] == base["completeness"]["complete"]
            assert rep["independent_fp32"]["coverage"]["reference_failures"] == 1
            case = dense(rep, "vllm", 128)
            assert case["fp32"]["status"] == "reference_failure"
            assert case["metrics"]["metrics_diagnostic"] is True

    @add
    def fp32_global_provenance_tamper_and_duplicate_reject():
        for field, value in (("source_sha256", "0" * 64), ("client_sha256", "0" * 64),
                             ("manifest_sha256", "0" * 64)):
            t, root, out = prep()
            with t:
                ref = make_fp32_ref(root)
                d = json.loads(ref.read_text()); d[field] = value; ref.write_text(json.dumps(d))
                try:
                    export(root, out, reference=ref)
                except ReportError as e:
                    assert field in str(e)
                else:
                    raise AssertionError(f"FP32 {field} tamper accepted")
        t, root, out = prep()
        with t:
            ref = make_fp32_ref(root, duplicate=True)
            try:
                export(root, out, reference=ref)
            except ReportError as e:
                assert "duplicate" in str(e)
            else:
                raise AssertionError("conflicting FP32 duplicate accepted")

    @add
    def fp32_hf_is_asymmetric_reference_order():
        hf_dense = {"record": {"shape": [2], "values": [1.0, 0.0], "all_finite": True}}
        rt_dense = {"record": {"shape": [2], "values": [1.0305, 0.0], "all_finite": True}}
        assert compare("embeddings", hf_dense, rt_dense)["status"] == "fail_norm"
        hf_rerank = {"record": {"shape": [1], "values": [1.0], "all_finite": True}}
        rt_rerank = {"record": {"shape": [1], "values": [1.0305], "all_finite": True}}
        assert compare("rerank", hf_rerank, rt_rerank)["status"] == "fail_values"

    @add
    def fp32_infinity_failure_forces_candidate_diagnostic_and_uniform_csv():
        model = {"id": "m", "revision": "a" * 40, "task": "embeddings", "lengths": [128]}
        inf_case = {"clean": True, "sample": {"sha256": "s"},
                    "record": {"shape": [2], "values": [1.0, 0.0], "all_finite": True}}
        cand_case = {"clean": True, "sample": {"sha256": "s"},
                     "record": {"shape": [2], "values": [1.025, 0.0], "all_finite": True}}
        runs = {("m", "infinity"): {"cases": {(128, None): inf_case}},
                ("m", "vllm"): {"cases": {(128, None): cand_case}},
                ("m", "sglang"): {"cases": {(128, None): cand_case}}}
        comps = [{"model": "m", "runtime": "vllm", "length": 128,
                  "status": "pass", "numerical_pass": True},
                 {"model": "m", "runtime": "sglang", "length": 128,
                  "status": "pass", "numerical_pass": True}]
        ref = {"model": "m", "revision": "a" * 40, "task": "embeddings", "length": 128,
               "sample_sha256": "s", "values": [1.05, 0.0], "shape": [2],
               "all_finite": True, "_match": "matched"}
        refctx = {"references": {("m", "a" * 40, "embeddings", 128): ref},
                  "failures": [], "issues": [],
                  "provenance": {"reference_file_sha256": "x"}}
        _, out = fp32_compare(refctx, {}, [model], runs, comps)
        assert out[("m", "infinity", 128)]["fp32_pass"] is False
        assert out[("m", "vllm", 128)]["fp32_pass"] is True
        assert out[("m", "vllm", 128)]["fully_eligible"] is False
        rendered = cell(runs[("m", "vllm")], cand_case, comps[0], "vllm")
        assert rendered.startswith("0.00 tok/s") or "tok/s" in rendered
        assert "[DIAGNOSTIC rel=pass fp32=pass]" in rendered
        missing, _ = fp32_compare({"references": {}, "failures": [], "issues": [],
                                   "provenance": {"reference_file_sha256": "x"}},
                                  {}, [model], runs, [])
        assert missing["coverage"]["fully_eligible_cases"] == 0
        rendered_inf = cell(runs[("m", "infinity")], inf_case, None, "infinity")
        assert "[DIAGNOSTIC rel=baseline fp32=reference_missing" in rendered_inf
        t, root, outdir = prep()
        with t:
            refpath = make_fp32_ref(root)
            export(root, outdir, reference=refpath)
            rows = list(csv.DictReader((outdir / "results.csv").open()))
            inf = next(x for x in rows if x["model_id"] == "acme/dense-a"
                       and x["runtime"] == "infinity" and x["length"] == "128")
            assert inf["fully_eligible"] in ("True", "true"), inf["fully_eligible"]

    @add
    def alternate_full_source_commit_accepted():
        t, root, out = prep()
        with t:
            alt = "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
            d = json.loads((root / "campaign.json").read_text())
            d["source_commit"] = alt
            (root / "campaign.json").write_text(json.dumps(d))
            export(root, out)
            rep = json.loads((out / "results.json").read_text())
            assert rep["campaign"]["source_commit"] == alt

    @add
    def manifest_actual_schema_read_only():
        p = Path("/root/infinity-benchmark-upstream-20261108/manifest.json")
        if p.is_file():
            d = json.loads(p.read_text())
            assert len(d["models"]) == 30
            assert any(x["id"] == "microsoft/harrier-oss-v1-27b" and x["parameters"] for x in d["models"])

    bad = []
    for fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            bad.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        else:
            print("ok", fn.__name__)
    print(json.dumps({"self_test": "pass" if not bad else "fail", "tests": len(tests),
                      "passed": len(tests) - len(bad), "failed": len(bad)}))
    for x in bad:
        print("fail", x, file=sys.stderr)
    return 0 if not bad else 1

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return selftest()
    if not a.input or not a.output:
        p.error("--input and --output are required")
    try:
        print(cjson(export(a.input, a.output, a.allow_incomplete, a.reference)))
    except ReportError as e:
        print(clean(str(e)), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
