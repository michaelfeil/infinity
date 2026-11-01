#!/usr/bin/env python3
"""Source-only, reproducible builder for the standalone MPNet attention shim."""
from __future__ import annotations
import argparse, concurrent.futures, dataclasses, hashlib, json, os, re, shutil
import subprocess, sys, tempfile, urllib.parse, urllib.request
from pathlib import Path
PYTORCH_COMMIT = "5c4886908584029761b579af026dcfb627c84070"
CUTLASS_COMMIT = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
EXPECTED_FILE_COUNT = 908
ARCHITECTURES = (90, 100, 120)
EXPORTS = ("mpnet_attn_prepare_bf16_h12d64", "mpnet_attn_validate",
           "mpnet_attn_launch_bf16_h12d64", "mpnet_attn_smem_bytes")
ADAPTER = "standalone_mpnet_attention.cu"
PATCH_HEADER = ("pytorch/aten/src/ATen/native/transformers/cuda/mem_eff_attention/"
                "kernel_forward.h")
DEFAULT_CUDA = "/usr/local/cuda-13.1"
RAW_HOST = "raw.githubusercontent.com"
FETCH_WORKERS, MAX_FETCH_BYTES = 4, 16 * 1024 * 1024
SHA_RE, COMMIT_RE = re.compile(r"[0-9a-f]{64}\Z"), re.compile(r"[0-9a-f]{40}\Z")
FORBIDDEN = ("torch", "c10", "ATen", "at::")
MANGLED_FORBIDDEN = ("_ZN2at", "_ZN3c10", "_ZN5torch", "_ZN4ATen", "_ZN5c10")
NOTICE = """Standalone MPNet attention packaging notice

Locally authored adapter plus a patched PyTorch mem_eff_attention
kernel_forward.h. Public PyTorch and CUTLASS headers are consumed from the
verified source tree and are not vendored here.

PyTorch and CUTLASS use BSD-3-Clause licenses; see licenses/.
No numerical correctness or performance qualification is claimed.
"""
class BuildError(RuntimeError):
    pass
@dataclasses.dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    url: str | None = None
@dataclasses.dataclass(frozen=True)
class Manifest:
    pytorch_commit: str
    cutlass_commit: str
    files: dict[str, SourceFile]
    cutlass_archive: dict = dataclasses.field(default_factory=dict)
def validate_relative_path(rel: str) -> str:
    if not isinstance(rel, str) or not rel: raise BuildError("path must be a non-empty string")
    if "\x00" in rel or "\\" in rel or ":" in rel: raise BuildError(f"unsafe path: {rel!r}")
    parts = rel.split("/")
    if rel.startswith("/") or any(p in ("", ".", "..") for p in parts): raise BuildError(f"unsafe relative path: {rel!r}")
    if Path(rel).is_absolute(): raise BuildError(f"absolute path rejected: {rel!r}")
    return rel
def resolve_source_file(root, rel: str) -> Path:
    validate_relative_path(rel)
    try: root_real = Path(root).resolve(strict=True)
    except OSError as exc: raise BuildError(f"source root is unavailable: {root}") from exc
    if not root_real.is_dir(): raise BuildError(f"source root is not a directory: {root}")
    current = root_real
    for part in rel.split("/"):
        current = current / part
        if current.is_symlink(): raise BuildError(f"symlink path rejected: {rel}")
    try:
        real = current.resolve(strict=False)
        inside = os.path.commonpath((str(root_real), str(real))) == str(root_real)
    except (OSError, ValueError): inside = False
    if not inside: raise BuildError(f"path escapes source root: {rel}")
    return current
def sha256_file(path, *, max_bytes: int | None = None) -> str:
    digest, total = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            total += len(chunk)
            if max_bytes is not None and total > max_bytes: raise BuildError(f"file exceeds byte limit: {path}")
            digest.update(chunk)
    return digest.hexdigest()
def validate_download_url(url: str) -> str:
    if not isinstance(url, str) or not url: raise BuildError("download URL is missing")
    try: parsed = urllib.parse.urlsplit(url)
    except ValueError as exc: raise BuildError(f"invalid download URL: {url}") from exc
    if parsed.scheme != "https" or parsed.hostname != RAW_HOST: raise BuildError(f"only pinned {RAW_HOST} URLs are allowed: {url}")
    if parsed.username or parsed.password or parsed.port is not None: raise BuildError(f"credentials/ports are not allowed in URL: {url}")
    decoded = urllib.parse.unquote(parsed.path)
    if "\\" in decoded or any(part == ".." for part in decoded.split("/")): raise BuildError(f"unsafe path in URL: {url}")
    return url
def source_url(source: SourceFile, manifest: Manifest) -> str:
    if source.url: return source.url
    if source.path.startswith("cutlass/"):
        return (f"https://raw.githubusercontent.com/NVIDIA/cutlass/{manifest.cutlass_commit}/"
                f"{source.path[len('cutlass/'):]}")
    raise BuildError(f"manifest entry lacks a URL: {source.path}")
def load_manifest(path, *, expected_pytorch_commit=PYTORCH_COMMIT,
                  expected_cutlass_commit=CUTLASS_COMMIT, allow_subset=False,
                  expected_file_count=EXPECTED_FILE_COUNT) -> Manifest:
    try: doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc: raise BuildError(f"cannot read manifest: {path}") from exc
    except json.JSONDecodeError as exc: raise BuildError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(doc, dict): raise BuildError("manifest root must be an object")
    pc, cc = doc.get("pytorch_commit"), doc.get("cutlass_commit")
    if not isinstance(pc, str) or not COMMIT_RE.match(pc): raise BuildError("manifest pytorch_commit must be a 40-hex commit")
    if not isinstance(cc, str) or not COMMIT_RE.match(cc): raise BuildError("manifest cutlass_commit must be a 40-hex commit")
    if expected_pytorch_commit is not None and pc != expected_pytorch_commit: raise BuildError(f"unexpected PyTorch commit: {pc}")
    if expected_cutlass_commit is not None and cc != expected_cutlass_commit: raise BuildError(f"unexpected CUTLASS commit: {cc}")
    raw = doc.get("files")
    if not isinstance(raw, dict) or not raw: raise BuildError("manifest files must be a non-empty object")
    files = {}
    for rel, meta in raw.items():
        validate_relative_path(rel)
        if not isinstance(meta, dict): raise BuildError(f"manifest entry must be an object: {rel}")
        sha, url = meta.get("sha256"), meta.get("url")
        if not isinstance(sha, str) or not SHA_RE.match(sha): raise BuildError(f"invalid sha256 for {rel}")
        if url is not None and not isinstance(url, str): raise BuildError(f"invalid url for {rel}")
        files[rel] = SourceFile(rel, sha, url)
    if not allow_subset and expected_file_count is not None and len(files) != expected_file_count:
        raise BuildError(f"manifest has {len(files)} files, expected {expected_file_count}")
    archive = doc.get("cutlass_archive")
    return Manifest(pc, cc, files, archive if isinstance(archive, dict) else {})
def source_tree_digest(manifest: Manifest) -> str:
    digest = hashlib.sha256()
    for rel in sorted(manifest.files): digest.update(f"{rel}\0{manifest.files[rel].sha256}\n".encode())
    return digest.hexdigest()
def verify_source_tree(root, manifest: Manifest) -> dict[str, str]:
    missing, corrupt = [], []
    for rel, source in sorted(manifest.files.items()):
        path = resolve_source_file(root, rel)
        if not path.is_file(): missing.append(rel)
        elif sha256_file(path) != source.sha256: corrupt.append(rel)
    if missing: raise BuildError("missing source files: " + ", ".join(missing[:5]))
    if corrupt: raise BuildError("source hash mismatch: " + ", ".join(corrupt[:5]))
    return {rel: manifest.files[rel].sha256 for rel in sorted(manifest.files)}
def _fetch_one(root, source: SourceFile, url: str, opener, max_bytes: int) -> None:
    dest = resolve_source_file(root, source.path)
    if dest.exists(): raise BuildError(f"refusing to overwrite existing file: {source.path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest = resolve_source_file(root, source.path)
    if dest.exists(): raise BuildError(f"refusing to overwrite existing file: {source.path}")
    fd, tmp_name = tempfile.mkstemp(prefix=".mpnet-fetch-", dir=str(dest.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with opener(url, timeout=60) as response:
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk: break
                    total += len(chunk)
                    if total > max_bytes: raise BuildError(f"download exceeds byte limit: {source.path}")
                    handle.write(chunk)
        if sha256_file(tmp, max_bytes=max_bytes) != source.sha256: raise BuildError(f"downloaded hash mismatch: {source.path}")
        if dest.exists(): raise BuildError(f"refusing to overwrite existing file: {source.path}")
        os.replace(tmp, dest)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
def fetch_missing(root, manifest: Manifest, *, opener=None, workers=FETCH_WORKERS,
                  max_bytes=MAX_FETCH_BYTES) -> int:
    opener = opener or urllib.request.urlopen
    pending = []
    for rel, source in sorted(manifest.files.items()):
        path = resolve_source_file(root, rel)
        if path.exists():
            if path.is_symlink() or not path.is_file(): raise BuildError(f"unsafe existing path: {rel}")
            if sha256_file(path, max_bytes=max_bytes) != source.sha256: raise BuildError(f"existing corrupt file refuses overwrite: {rel}")
        else: pending.append(source)
    if not pending: return 0
    urls = {s.path: validate_download_url(source_url(s, manifest)) for s in pending}
    bounded = min(FETCH_WORKERS, max(1, int(workers)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=bounded) as pool:
        futures = [pool.submit(_fetch_one, root, s, urls[s.path], opener, max_bytes) for s in pending]
        for future in futures: future.result()
    return len(pending)
def check_only(package_dir, *, manifest_path, source_root=None, fetch=False,
               opener=None, allow_subset=False, expected_file_count=EXPECTED_FILE_COUNT) -> dict:
    manifest = load_manifest(manifest_path, allow_subset=allow_subset, expected_file_count=expected_file_count)
    package = Path(package_dir)
    adapter, header = package / ADAPTER, package / "kernel_forward.h"
    if not adapter.is_file(): raise BuildError(f"missing adapter: {adapter}")
    if not header.is_file(): raise BuildError(f"missing patched header: {header}")
    text = header.read_text(encoding="utf-8", errors="replace")
    for marker in ("bias_table_ptr", "relative_bucket"):
        if marker not in text: raise BuildError("kernel_forward.h is not bucket-bias patched")
    if fetch and source_root is None: raise BuildError("--fetch requires --source-root")
    fetched = verified = 0
    if source_root is not None:
        if fetch: fetched = fetch_missing(source_root, manifest, opener=opener)
        verified = len(verify_source_tree(source_root, manifest))
    return {"mode": "check-only", "manifest": str(manifest_path),
            "pytorch_commit": manifest.pytorch_commit, "cutlass_commit": manifest.cutlass_commit,
            "file_count": len(manifest.files), "verified_files": verified,
            "fetched_files": fetched, "network_used": bool(fetch),
            "cuda_used": False, "gpu_queried": False}
def parse_architectures(values) -> list[int]:
    if isinstance(values, (str, int)): values = [values]
    if not values: raise BuildError("--arch is required with --build")
    flat = []
    for value in values: flat.extend(value if isinstance(value, (list, tuple)) else (value,))
    result = []
    for value in flat:
        for raw in str(value).split(","):
            raw = raw.strip()
            if raw.startswith("sm_"): raw = raw[3:]
            if not raw.isdigit(): raise BuildError(f"invalid architecture: {raw}")
            arch = int(raw)
            if arch not in ARCHITECTURES: raise BuildError(f"unsupported architecture: {arch}")
            if arch not in result: result.append(arch)
    if not result: raise BuildError("at least one architecture is required")
    return result
def require_fresh_output(output) -> Path:
    path = Path(output)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise BuildError(f"output directory is not fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
def install_patch_tree(package_dir, output) -> Path:
    dest = Path(output) / PATCH_HEADER
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(package_dir) / "kernel_forward.h", dest)
    return dest
def install_licenses(package_dir, source_root, output) -> list[str]:
    out = Path(output) / "licenses"
    out.mkdir(parents=True, exist_ok=True)
    package, public = Path(package_dir), Path(source_root)
    choices = {
        "LICENSE-PyTorch-BSD3": [package / "licenses" / "LICENSE-PyTorch-BSD3",
                                 package / "LICENSE-PyTorch-BSD3", public / "pytorch" / "LICENSE"],
        "LICENSE-CUTLASS-BSD3": [package / "licenses" / "LICENSE-CUTLASS-BSD3",
                                 package / "LICENSE-CUTLASS-BSD3", public / "cutlass" / "LICENSE.txt"]}
    copied = []
    for name, candidates in choices.items():
        for candidate in candidates:
            if candidate.is_file():
                shutil.copyfile(candidate, out / name)
                copied.append(f"licenses/{name}")
                break
        else: raise BuildError(f"license not found: {name}")
    notice_text = (package / "NOTICE").read_text(encoding="utf-8") if (package / "NOTICE").is_file() else NOTICE
    (Path(output) / "NOTICE").write_text(notice_text, encoding="utf-8")
    (out / "NOTICE").write_text(notice_text, encoding="utf-8")
    return copied
def compiler_version(nvcc, *, runner=subprocess.run, env=None) -> str:
    proc = runner([str(nvcc), "--version"], capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0: raise BuildError(f"cannot query nvcc version: {proc.stderr.strip()}")
    return " ".join((proc.stdout or proc.stderr).split())
def build_command(nvcc, arch: int, output, source_root, adapter, dest) -> list[str]:
    public = Path(source_root)
    cmd = [str(nvcc), "-std=c++17", "-O3", "--expt-relaxed-constexpr", "-shared", "--cudart=shared",
           "-Xcompiler", "-fPIC", "-Xlinker", "-z", "-Xlinker", "defs",
           "-DMPNET_ATTENTION_HAVE_BUCKET_BIAS=1", "-DC10_USING_CUSTOM_GENERATED_MACROS",
           f"-arch=sm_{arch}", "-I", str(Path(output) / "pytorch" / "aten" / "src"),
           "-I", str(output), "-I", str(public / "pytorch" / "aten" / "src"),
           "-I", str(public / "pytorch"), "-I", str(public / "cutlass" / "include")]
    stubs = Path(nvcc).parent.parent / "lib64" / "stubs"
    if (stubs / "libcuda.so").is_file(): cmd.extend(["-L", str(stubs)])
    cmd.extend([str(adapter), "-lcuda", "-o", str(dest)])
    return cmd
def audit_text(dynamic_output: str, undefined_output: str, defined_output: str | None = None) -> dict:
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", dynamic_output)
    bad = [n for n in needed if any(m.lower() in n.lower() for m in ("torch", "c10", "aten"))]
    if bad: raise BuildError("forbidden ELF dependencies: " + ", ".join(bad))
    bad_syms = []
    for line in undefined_output.splitlines():
        parts = line.split()
        if parts and any(marker in parts[-1] for marker in FORBIDDEN + MANGLED_FORBIDDEN): bad_syms.append(parts[-1])
    if bad_syms: raise BuildError("forbidden undefined symbols: " + ", ".join(bad_syms[:5]))
    if defined_output is not None:
        missing = [name for name in EXPORTS if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", defined_output)]
        if missing: raise BuildError("missing exported symbols: " + ", ".join(missing))
    return {"needed": needed, "undefined_checked": True, "exports_checked": defined_output is not None}
def audit_library(path, *, runner=subprocess.run) -> dict:
    readelf, nm = shutil.which("readelf"), shutil.which("nm")
    if not readelf or not nm: raise BuildError("readelf and nm are required for the ELF audit")
    commands = {"dynamic": [readelf, "-d", str(path)], "undefined": [nm, "-u", "--demangle", str(path)],
                "defined": [nm, "-D", "--defined-only", "--demangle", str(path)]}
    output = {}
    for label, command in commands.items():
        proc = runner(command, capture_output=True, text=True, check=False)
        if proc.returncode != 0: raise BuildError(f"{label} audit failed: {proc.stderr.strip()}")
        output[label] = proc.stdout
    return audit_text(output["dynamic"], output["undefined"], output["defined"])
def make_arch_manifest(*, arch: int, architectures, manifest: Manifest, library_sha: str, command,
                       tool_version: str, provenance: dict, cuda_toolkit, licenses) -> dict:
    abi = {"version": 1, "dtype": "BF16", "num_heads": 12, "head_dim": 64,
           "qkv_row_stride": 2304, "bias_num_buckets": 32, "causal": False, "exports": list(EXPORTS)}
    source = {"pytorch_commit": manifest.pytorch_commit, "cutlass_commit": manifest.cutlass_commit}
    return {"schema": "mpnet-attention-manifest", "format_version": 1, "abi_version": 1,
            "dtype": "bf16", "num_heads": 12, "head_dim": 64, "qkv_row_stride": 2304,
            "bias_num_buckets": 32, "causal": False, "abi": abi,
            "architectures": list(architectures), "architecture": arch, "arch": f"sm_{arch}",
            "pytorch_commit": manifest.pytorch_commit, "cutlass_commit": manifest.cutlass_commit,
            "source": source, "symbols": dict(zip(("prepare","validate","launch","smem_bytes"),EXPORTS)),
            "library": {"file": "libmpnet_attention.so", "sha256": library_sha},
            "library_sha256": library_sha, "library_sha": library_sha,
            "build": {"command": list(command), "tool_version": tool_version, "nvcc_version": tool_version,
                      "cuda_toolkit": str(cuda_toolkit), "gpu_queried": False,
                      "environment": {"CUDA_VISIBLE_DEVICES": "-1"}},
            "provenance": provenance, "licenses": list(licenses), "notice": "NOTICE",
            "qualification": {"numerical": False, "performance": False, "status": "not-qualified"}}
def write_json(path, document) -> None:
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def run_build(*, package_dir, manifest_path, source_root, output, cuda_toolkit, arches,
              fetch=False, opener=None, allow_subset=False,
              expected_file_count=EXPECTED_FILE_COUNT, runner=subprocess.run) -> dict:
    if source_root is None: raise BuildError("--build requires --source-root")
    arches = parse_architectures(arches)
    manifest = load_manifest(manifest_path, allow_subset=allow_subset, expected_file_count=expected_file_count)
    if fetch: fetch_missing(source_root, manifest, opener=opener)
    verify_source_tree(source_root, manifest)
    package, toolkit, nvcc = Path(package_dir), Path(cuda_toolkit), Path(cuda_toolkit) / "bin" / "nvcc"
    if not nvcc.is_file(): raise BuildError(f"nvcc not found under --cuda-toolkit: {toolkit}")
    out = require_fresh_output(output)
    patch_header = install_patch_tree(package, out)
    licenses = install_licenses(package, source_root, out)
    adapter_out = out / "src" / ADAPTER
    adapter_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(package / ADAPTER, adapter_out)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "-1"
    version = compiler_version(nvcc, runner=runner, env=env)
    provenance = {"manifest_sha256": sha256_file(manifest_path), "manifest_file_count": len(manifest.files),
                  "source_tree_sha256": source_tree_digest(manifest), "adapter_sha256": sha256_file(adapter_out),
                  "patched_kernel_forward_sha256": sha256_file(patch_header), "cuda_visible_devices": "-1"}
    artifacts = {}
    for arch in arches:
        arch_dir = out / f"sm_{arch}"; arch_dir.mkdir()
        library = arch_dir / "libmpnet_attention.so"
        command = build_command(nvcc, arch, out, source_root, adapter_out, library)
        proc = runner(command, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0: raise BuildError(f"nvcc failed for sm_{arch}:\n{proc.stderr[-4000:]}")
        audit_library(library, runner=runner)
        library_sha = sha256_file(library)
        write_json(arch_dir / "manifest.json", make_arch_manifest(
            arch=arch, architectures=ARCHITECTURES, manifest=manifest, library_sha=library_sha,
            command=command, tool_version=version, provenance=provenance,
            cuda_toolkit=toolkit, licenses=licenses))
        artifacts[f"sm_{arch}"] = {"architecture": arch, "library": f"sm_{arch}/libmpnet_attention.so",
                                   "sha256": library_sha, "manifest": f"sm_{arch}/manifest.json"}
    write_json(out / "manifest.json", {
        "schema": "mpnet-attention-build-index", "format_version": 1, "abi_version": 1,
        "dtype": "bf16", "num_heads": 12, "head_dim": 64, "qkv_row_stride": 2304,
        "bias_num_buckets": 32, "causal": False, "architectures": list(ARCHITECTURES),
        "architecture": arches[0] if len(arches) == 1 else None, "built_architectures": list(arches), "pytorch_commit": manifest.pytorch_commit,
        "cutlass_commit": manifest.cutlass_commit, "symbols": list(EXPORTS), "artifacts": artifacts,
        "source": {"pytorch": manifest.pytorch_commit, "cutlass": manifest.cutlass_commit,
                   "pytorch_commit": manifest.pytorch_commit, "cutlass_commit": manifest.cutlass_commit},
        "build": {"nvcc_version": version, "tool_version": version, "cuda_toolkit": str(toolkit), "gpu_queried": False},
        "provenance": provenance, "licenses": licenses, "notice": "NOTICE",
        "qualification": {"numerical": False, "performance": False, "status": "not-qualified"}})
    return {"mode": "build", "output": str(out), "architectures": list(arches),
            "artifacts": artifacts, "network_used": bool(fetch), "gpu_queried": False}
def parser() -> argparse.ArgumentParser:
    argp = argparse.ArgumentParser(description=__doc__)
    argp.add_argument("--manifest"); argp.add_argument("--source-root"); argp.add_argument("--package-dir")
    argp.add_argument("--fetch", action="store_true")
    mode = argp.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true"); mode.add_argument("--build", action="store_true")
    argp.add_argument("--output"); argp.add_argument("--cuda-toolkit", default=DEFAULT_CUDA)
    argp.add_argument("--arch", action="append", nargs="+", default=[])
    return argp
def main(argv=None) -> int:
    args = parser().parse_args(argv)
    package = Path(args.package_dir) if args.package_dir else Path(__file__).resolve().parent
    manifest_path = Path(args.manifest) if args.manifest else package / "public-source-manifest.json"
    try:
        if args.build:
            if not args.output: raise BuildError("--build requires --output")
            result = run_build(package_dir=package, manifest_path=manifest_path, source_root=args.source_root,
                               output=args.output, cuda_toolkit=args.cuda_toolkit,
                               arches=parse_architectures(args.arch), fetch=args.fetch)
        else:
            result = check_only(package, manifest_path=manifest_path, source_root=args.source_root, fetch=args.fetch)
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
