#!/usr/bin/env python3
"""Fingerprint the complete active Python serving environment.

The default command writes one JSON object.  It covers the interpreter binary,
the full standard library (including ``lib-dynload``), every active
site/dist-packages root, every external import root found on ``sys.path``, and
explicit ``--include-root`` entries.  Actual probes require
``PYTHONSAFEPATH=1``: implicit CWD/script-directory entries are then absent and
every nonempty ``sys.path`` root, including explicit wrapper ``PYTHONPATH``
roots, is bound.  The non-safe-path fake-root view excludes implicit
CWD/script-directory entries so tests can exercise that distinction.

Only ``__pycache__`` directories and ``.pyc`` files are skipped.  Nested
site/dist-packages directories are represented by their own roots rather than
duplicated inside the standard-library root.  Selected native SDK/loader inputs
are bound separately by the runner; remaining OS/driver files are outside this
recorded Python-content scope.  Distribution names and versions come from
``importlib.metadata`` without importing the distributions themselves.  The
helper source bytes are hashed into ``integrity_source_sha256`` and the public
digest.

The public JSON omits absolute host paths and per-file paths.  ``--private-roots``
adds separate absolute root and interpreter paths that do not participate in the
digest.  ``--self-test`` runs the bundled ``unittest`` suite against temporary
fake roots and never probes the active environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
import sysconfig
import tempfile
import unittest
from unittest import mock
import zipfile


SCHEMA_VERSION = 1
KIND = "python_environment_content_fingerprint"
SCOPE = (
    "Python import roots (stdlib, active site/dist-packages, external import "
    "overrides, and explicit includes) plus the interpreter binary. Selected "
    "native SDK/loader inputs are bound separately by the runner; remaining "
    "OS/driver files are outside this recorded Python-content scope."
)
EXCLUSIONS = (
    "'__pycache__' directories are skipped",
    "'.pyc' files are skipped",
)
SITE_DIRECTORY_NAMES = frozenset({"site-packages", "dist-packages"})
STDLIB_DIRECTORY_NAMES = frozenset({"site-packages", "dist-packages"})
CHUNK_SIZE = 1024 * 1024
MAX_ERROR_LENGTH = 2000


class FingerprintError(RuntimeError):
    """The active Python environment could not be fingerprinted safely."""


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _real(path):
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _is_within(child, parent):
    child = os.path.normpath(child)
    parent = os.path.normpath(parent)
    if child == parent:
        return True
    if parent == os.sep:
        return True
    return child.startswith(parent + os.sep)


def _is_broad_root(real):
    home = os.path.expanduser("~")
    if not home or home == "~":
        return real == os.sep
    home = _real(home)
    if home == os.sep:
        return real == os.sep
    return real == os.sep or _is_within(home, real)


def _stdlib_zip_names():
    major, minor = sys.version_info[:2]
    return frozenset(
        {"python%d%d.zip" % (major, minor), "python%d.%d.zip" % (major, minor)}
    )


def _stat_signature(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_file(path, context):
    try:
        before = os.stat(path, follow_symlinks=True)
    except OSError as exc:
        raise FingerprintError(
            "cannot access %s: %s" % (context, exc.strerror or exc)
        ) from exc
    if not stat.S_ISREG(before.st_mode):
        raise FingerprintError("not a regular file: %s" % context)
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = os.stat(path, follow_symlinks=True)
    except OSError as exc:
        raise FingerprintError(
            "cannot read %s: %s" % (context, exc.strerror or exc)
        ) from exc
    if _stat_signature(before) != _stat_signature(after) or size != before.st_size:
        raise FingerprintError("file changed while being hashed: %s" % context)
    return size, digest.hexdigest()


def _is_zip_archive(path, context):
    try:
        with open(path, "rb") as handle:
            return zipfile.is_zipfile(handle)
    except OSError as exc:
        raise FingerprintError(
            "cannot inspect archive %s: %s" % (context, exc.strerror or exc)
        ) from exc


def _walk_directory(
    path,
    prefix,
    ancestors,
    root_real,
    other_root_reals,
    files,
    skip_nested_site,
):
    try:
        with os.scandir(path) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
    except OSError as exc:
        raise FingerprintError(
            "cannot read directory %s: %s"
            % (prefix or "<root>", exc.strerror or exc)
        ) from exc
    for entry in entries:
        name = entry.name
        relative = prefix + name
        try:
            link_stat = entry.stat(follow_symlinks=False)
            target_stat = (
                entry.stat(follow_symlinks=True)
                if stat.S_ISLNK(link_stat.st_mode)
                else link_stat
            )
        except OSError as exc:
            raise FingerprintError(
                "cannot follow %s: %s" % (relative, exc.strerror or exc)
            ) from exc
        real = os.path.realpath(entry.path)
        if real in other_root_reals and real != root_real:
            continue
        mode = target_stat.st_mode
        if stat.S_ISDIR(mode):
            if name == "__pycache__":
                continue
            if skip_nested_site and name in STDLIB_DIRECTORY_NAMES:
                continue
            if real in ancestors:
                raise FingerprintError("directory cycle at %s" % relative)
            _walk_directory(
                entry.path,
                relative + "/",
                ancestors | {real},
                root_real,
                other_root_reals,
                files,
                skip_nested_site,
            )
        elif stat.S_ISREG(mode):
            if name.endswith(".pyc"):
                continue
            size, digest = _hash_file(entry.path, relative)
            files.append({"path": relative, "size": size, "sha256": digest})
        else:
            raise FingerprintError("special file in environment: %s" % relative)


def _root_files(spec, other_root_reals):
    try:
        current_real = _real(spec["path"])
    except OSError as exc:
        raise FingerprintError(
            "cannot resolve root %s: %s" % (spec["label"], exc.strerror or exc)
        ) from exc
    if current_real != spec["real"]:
        raise FingerprintError("root changed during discovery: %s" % spec["label"])
    if not spec["is_dir"]:
        size, digest = _hash_file(spec["path"], spec["label"])
        logical = "<zip-archive>" if spec["is_zip"] else "<file>"
        return [{"path": logical, "size": size, "sha256": digest}]
    files = []
    _walk_directory(
        spec["path"],
        "",
        {spec["real"]},
        spec["real"],
        other_root_reals,
        files,
        spec["kind"] == "stdlib",
    )
    files.sort(key=lambda item: item["path"])
    return files


def _unique_paths(values):
    seen = set()
    result = []
    for value in values:
        if not value:
            continue
        text = os.fspath(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _site_directories():
    try:
        import site
    except Exception:
        return []
    result = []
    if getattr(site, "ENABLE_USER_SITE", True):
        try:
            result.append(site.getusersitepackages())
        except Exception:
            pass
    try:
        result.extend(site.getsitepackages())
    except Exception:
        pass
    return _unique_paths(result)


def _classify_root(raw, real, site_reals, stdlib_reals):
    raw_name = os.path.basename(os.path.normpath(os.fspath(raw)))
    real_name = os.path.basename(real.rstrip(os.sep))
    if (
        raw_name in SITE_DIRECTORY_NAMES
        or real_name in SITE_DIRECTORY_NAMES
        or any(_is_within(real, root) for root in site_reals)
    ):
        return "site_packages"
    stdlib_parents = {os.path.dirname(root) for root in stdlib_reals}
    if real_name in _stdlib_zip_names() and os.path.dirname(real) in stdlib_parents:
        return "stdlib"
    if any(_is_within(real, root) for root in stdlib_reals):
        return "stdlib"
    return "import_override"


def _root_label_for(distribution_real, selected):
    best = None
    for item in selected:
        if _is_within(distribution_real, item["real"]):
            if best is None or len(item["real"]) > len(best["real"]):
                best = item
    return None if best is None else best["label"]


def _collect_distributions(selected):
    paths = [item["path"] for item in selected if item["is_dir"]]
    if not paths:
        return []
    try:
        distributions = importlib.metadata.distributions(path=paths)
        found = []
        for distribution in distributions:
            try:
                name = distribution.metadata.get("Name") or getattr(
                    distribution, "name", None
                )
                version = distribution.version or distribution.metadata.get("Version")
                location = getattr(distribution, "_path", None)
                if location is None:
                    location = distribution.locate_file("")
                location_real = _real(location)
            except Exception:
                continue
            if not name or version is None:
                continue
            root_label = _root_label_for(location_real, selected)
            if root_label is None:
                continue
            found.append(
                {
                    "name": str(name),
                    "version": str(version),
                    "root_label": root_label,
                }
            )
    except Exception as exc:
        raise FingerprintError(
            "cannot inspect distribution metadata: %s" % (exc,)
        ) from exc
    found.sort(
        key=lambda item: (
            item["name"].lower(),
            item["name"],
            item["version"],
            item["root_label"],
        )
    )
    deduplicated = []
    seen = set()
    for item in found:
        key = (item["name"], item["version"], item["root_label"])
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


def _interpreter_info(executable, safe_path):
    if not executable:
        raise FingerprintError("interpreter executable is not available")
    try:
        executable_absolute = os.path.abspath(os.fspath(executable))
    except OSError as exc:
        raise FingerprintError(
            "cannot identify interpreter executable: %s" % (exc.strerror or exc)
        ) from exc
    _, executable_sha256 = _hash_file(
        executable_absolute, "interpreter executable"
    )
    implementation = platform.python_implementation()
    version = platform.python_version()
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    return {
        "implementation": implementation,
        "version": version,
        "cache_tag": cache_tag,
        "safe_path": bool(safe_path),
        "executable_sha256": executable_sha256,
    }


def _require_safe_path(flags):
    if not bool(getattr(flags, "safe_path", False)):
        raise FingerprintError(
            "PYTHONSAFEPATH=1 is required for environment fingerprinting"
        )


def fingerprint(
    *,
    sys_path,
    executable,
    script_path,
    cwd,
    site_dirs=(),
    sysconfig_paths=None,
    include_roots=(),
    no_site=False,
    private=False,
    safe_path=None,
):
    """Return the public fingerprint for the supplied interpreter view."""
    sysconfig_paths = {} if sysconfig_paths is None else dict(sysconfig_paths)
    sys_path = list(sys_path)
    include_roots = list(include_roots)
    site_dirs = list(site_dirs)
    if safe_path is None:
        safe_path = bool(getattr(sys.flags, "safe_path", False))
    safe_path = bool(safe_path)
    script_absolute = os.path.abspath(os.fspath(script_path))
    excluded_reals = {_real(os.path.dirname(script_absolute)), _real(cwd)}
    stdlib_raws = _unique_paths(
        [sysconfig_paths.get("stdlib"), sysconfig_paths.get("platstdlib")]
    )
    site_raws = _unique_paths(
        site_dirs
        + [sysconfig_paths.get("purelib"), sysconfig_paths.get("platlib")]
    )
    if no_site:
        sys_path_reals = {_real(path) for path in sys_path if path}
        site_raws = [path for path in site_raws if _real(path) in sys_path_reals]
    site_reals = {_real(path) for path in site_raws}
    stdlib_reals = {_real(path) for path in stdlib_raws}

    candidates = []
    for path in sys_path:
        candidates.append((path, "sys_path"))
    for path in stdlib_raws:
        candidates.append((path, "stdlib"))
    for path in site_raws:
        candidates.append((path, "site"))
    for path in include_roots:
        candidates.append((path, "include"))

    selected = []
    counters = {}
    for raw, source in candidates:
        if not raw:
            continue
        absolute = os.path.abspath(os.fspath(raw))
        try:
            link_stat = os.lstat(absolute)
        except FileNotFoundError:
            if source == "include":
                raise FingerprintError(
                    "explicit include root does not exist: %s" % absolute
                )
            continue
        except OSError as exc:
            raise FingerprintError(
                "cannot access root %s: %s"
                % (absolute, exc.strerror or exc)
            ) from exc
        if stat.S_ISLNK(link_stat.st_mode):
            try:
                root_stat = os.stat(absolute, follow_symlinks=True)
            except FileNotFoundError:
                if source == "include":
                    raise FingerprintError(
                        "explicit include root is dangling: %s" % absolute
                    )
                continue
            except OSError as exc:
                raise FingerprintError(
                    "cannot follow root %s: %s"
                    % (absolute, exc.strerror or exc)
                ) from exc
        else:
            root_stat = link_stat
        real = _real(absolute)
        if source != "include" and not safe_path and real in excluded_reals:
            continue
        if _is_broad_root(real):
            raise FingerprintError("refusing broad import root: %s" % absolute)
        if any(item["real"] == real for item in selected):
            continue
        is_dir = stat.S_ISDIR(root_stat.st_mode)
        if not is_dir:
            if not stat.S_ISREG(root_stat.st_mode):
                raise FingerprintError(
                    "root is not a regular file or directory: %s" % absolute
                )
            is_zip = _is_zip_archive(absolute, absolute)
            if not is_zip:
                if source == "include":
                    raise FingerprintError(
                        "explicit include root is not a directory or zip archive: %s"
                        % absolute
                    )
                continue
        else:
            is_zip = False
        kind = (
            "explicit"
            if source == "include"
            else _classify_root(absolute, real, site_reals, stdlib_reals)
        )
        counters[kind] = counters.get(kind, 0) + 1
        selected.append(
            {
                "label": "%s:%d" % (kind, counters[kind]),
                "kind": kind,
                "path": absolute,
                "real": real,
                "is_dir": is_dir,
                "is_zip": is_zip,
            }
        )

    other_root_reals = {item["real"] for item in selected}
    content_roots = []
    total_file_count = 0
    total_byte_count = 0
    for item in selected:
        files = _root_files(item, other_root_reals)
        root_digest = _sha256_bytes(
            _canonical_json({"files": files}).encode("utf-8")
        )
        file_count = len(files)
        byte_count = sum(entry["size"] for entry in files)
        total_file_count += file_count
        total_byte_count += byte_count
        content_roots.append(
            {
                "label": item["label"],
                "kind": item["kind"],
                "file_count": file_count,
                "byte_count": byte_count,
                "digest_sha256": root_digest,
            }
        )

    _, integrity_source_sha256 = _hash_file(
        script_absolute, "integrity helper source"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "scope": SCOPE,
        "exclusions": list(EXCLUSIONS),
        "interpreter": _interpreter_info(executable, safe_path),
        "integrity_source_sha256": integrity_source_sha256,
        "content_roots": content_roots,
        "distributions": _collect_distributions(selected),
        "total_file_count": total_file_count,
        "total_byte_count": total_byte_count,
    }
    result["digest_sha256"] = _sha256_bytes(
        _canonical_json(result).encode("utf-8")
    )
    if private:
        result["private_roots"] = [
            {"label": item["label"], "path": item["path"]} for item in selected
        ]
        result["private_executable"] = os.path.abspath(os.fspath(executable))
    return result


def fingerprint_runtime(include_roots=(), private=False):
    _require_safe_path(sys.flags)
    return fingerprint(
        sys_path=sys.path,
        executable=sys.executable,
        script_path=os.path.abspath(__file__),
        cwd=os.getcwd(),
        site_dirs=_site_directories(),
        sysconfig_paths=sysconfig.get_paths(),
        include_roots=include_roots,
        no_site=bool(getattr(sys.flags, "no_site", False)),
        private=private,
        safe_path=True,
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Fingerprint the complete active Python serving environment. "
            "Default output is one JSON object on stdout."
        )
    )
    parser.add_argument(
        "--include-root",
        action="append",
        default=[],
        metavar="PATH",
        help="add an explicit known import root (repeatable)",
    )
    parser.add_argument(
        "--private-roots",
        action="store_true",
        help=(
            "also emit absolute root and interpreter paths outside the digest"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the bundled stdlib unittest suite without probing the environment",
    )
    return parser


def _write_error(message):
    text = " ".join(str(message).split())
    sys.stderr.write("benchmark-integrity: %s\n" % text[:MAX_ERROR_LENGTH])


class _SelfTests(unittest.TestCase):
    def _env(self, base, create=True):
        base = os.fspath(base)
        stdlib = os.path.join(base, "python3.12")
        libdyn = os.path.join(stdlib, "lib-dynload")
        site = os.path.join(base, "venv", "lib", "python3.12", "site-packages")
        project = os.path.join(base, "project")
        scripts = os.path.join(project, "scripts")
        executable = os.path.join(base, "venv", "bin", "python")
        if create:
            for directory in (stdlib, libdyn, site, project, scripts):
                os.makedirs(directory, exist_ok=True)
            self._write(os.path.join(stdlib, "os.py"), "stdlib-v1\n")
            self._write(
                os.path.join(stdlib, "encodings", "__init__.py"), "enc-v1\n"
            )
            self._write(os.path.join(libdyn, "_native.so"), b"native-v1")
            self._write(os.path.join(site, "torch", "__init__.py"), "torch-v1\n")
            self._write(os.path.join(site, "torch", "version.py"), "torch-version-v1\n")
            self._write(os.path.join(site, "triton", "__init__.py"), "triton-v1\n")
            self._write(
                os.path.join(site, "transformers", "modeling.py"),
                "transformers-v1\n",
            )
            self._write(
                os.path.join(site, "torch-2.0.0.dist-info", "METADATA"),
                "Metadata-Version: 2.1\nName: torch\nVersion: 2.0.0\n\n",
            )
            os.makedirs(os.path.dirname(executable), exist_ok=True)
            self._write(executable, b"fake-python-binary-v1")
            self._write(
                os.path.join(scripts, "benchmark-integrity.py"), "# fixture\n"
            )
            self._write(os.path.join(project, "cwd_module.py"), "cwd-v1\n")
        return {
            "base": base,
            "stdlib": stdlib,
            "libdyn": libdyn,
            "site": site,
            "project": project,
            "scripts": scripts,
            "executable": executable,
            "sys_path": [
                scripts,
                project,
                stdlib,
                libdyn,
                site,
            ],
            "sysconfig": {
                "stdlib": stdlib,
                "platstdlib": stdlib,
                "purelib": site,
                "platlib": site,
            },
        }

    def _write(self, path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kwargs = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(path, mode, **kwargs) as handle:
            handle.write(content)

    def _read(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _fingerprint(self, env, **overrides):
        values = {
            "sys_path": env["sys_path"],
            "executable": env["executable"],
            "script_path": os.path.join(env["scripts"], "benchmark-integrity.py"),
            "cwd": env["project"],
            "site_dirs": [env["site"]],
            "sysconfig_paths": env["sysconfig"],
            "include_roots": [],
            "no_site": False,
            "private": False,
            "safe_path": True,
        }
        values.update(overrides)
        return fingerprint(**values)

    def _root_for_path(self, result, path):
        label = next(
            item["label"]
            for item in result["private_roots"]
            if item["path"] == path
        )
        return next(
            item for item in result["content_roots"] if item["label"] == label
        )

    def test_source_edits_without_version_change_alter_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            previous = self._fingerprint(env)["digest_sha256"]
            for relative in (
                os.path.join("torch", "version.py"),
                os.path.join("triton", "__init__.py"),
                os.path.join("transformers", "modeling.py"),
            ):
                path = os.path.join(env["site"], relative)
                self._write(path, self._read(path) + "edited\n")
                current = self._fingerprint(env)["digest_sha256"]
                self.assertNotEqual(previous, current)
                previous = current

    def test_file_addition_and_deletion_alter_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env)["digest_sha256"]
            added = os.path.join(env["site"], "torch", "extra.py")
            self._write(added, "extra\n")
            during = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before, during)
            os.unlink(added)
            after = self._fingerprint(env)["digest_sha256"]
            self.assertEqual(before, after)

    def test_native_binary_change_alters_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env)["digest_sha256"]
            self._write(os.path.join(env["libdyn"], "_native.so"), b"native-v2")
            after = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before, after)

    def test_metadata_and_pth_changes_alter_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            metadata = os.path.join(
                env["site"], "torch-2.0.0.dist-info", "METADATA"
            )
            before = self._fingerprint(env)["digest_sha256"]
            self._write(
                metadata,
                "Metadata-Version: 2.1\nName: torch\nVersion: 2.0.0\n\nchanged\n",
            )
            after_metadata = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before, after_metadata)
            self._write(os.path.join(env["site"], "stock.pth"), "stock-copy\n")
            after_pth = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(after_metadata, after_pth)

    def test_symlink_file_target_change_alters_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            outside = os.path.join(directory, "outside")
            os.makedirs(outside, exist_ok=True)
            first = os.path.join(outside, "first.txt")
            second = os.path.join(outside, "second.txt")
            self._write(first, "first\n")
            self._write(second, "second\n")
            link = os.path.join(env["site"], "torch", "linked.txt")
            os.symlink(first, link)
            before = self._fingerprint(env)["digest_sha256"]
            self._write(first, "first-changed\n")
            after_content = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before, after_content)
            os.unlink(link)
            os.symlink(second, link)
            after_target = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(after_content, after_target)

    def test_symlink_directory_target_change_alters_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            first = os.path.join(directory, "outside", "first")
            second = os.path.join(directory, "outside", "second")
            self._write(os.path.join(first, "data.txt"), "first\n")
            self._write(os.path.join(second, "data.txt"), "second\n")
            link = os.path.join(env["site"], "torch", "data")
            os.symlink(first, link, target_is_directory=True)
            before = self._fingerprint(env)["digest_sha256"]
            os.unlink(link)
            os.symlink(second, link, target_is_directory=True)
            after = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before, after)

    def test_symlink_cycle_and_dangling_link_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            cycle = os.path.join(env["site"], "torch", "cycle")
            os.symlink(env["site"], cycle, target_is_directory=True)
            with self.assertRaises(FingerprintError):
                self._fingerprint(env)
            os.unlink(cycle)
            dangling = os.path.join(env["site"], "torch", "dangling.txt")
            os.symlink(os.path.join(directory, "missing"), dangling)
            with self.assertRaises(FingerprintError):
                self._fingerprint(env)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "mkfifo is unavailable")
    def test_special_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            os.mkfifo(os.path.join(env["site"], "torch", "pipe"))
            with self.assertRaises(FingerprintError):
                self._fingerprint(env)

    def test_bytecode_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env)
            cache = os.path.join(env["site"], "torch", "__pycache__")
            os.makedirs(cache, exist_ok=True)
            self._write(os.path.join(cache, "version.cpython-312.pyc"), b"bytecode")
            self._write(os.path.join(env["site"], "torch", "orphan.pyc"), b"bytecode")
            after = self._fingerprint(env)
            self.assertEqual(before, after)

    def test_import_root_precedence_and_repeated_calls_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            first = os.path.join(directory, "override-one")
            second = os.path.join(directory, "override-two")
            self._write(os.path.join(first, "one.py"), "one\n")
            self._write(os.path.join(second, "two.py"), "two\n")
            sys_path = env["sys_path"] + [first, second]
            public = self._fingerprint(env, sys_path=sys_path)
            private = self._fingerprint(env, sys_path=sys_path, private=True)
            roots = [
                item
                for item in private["content_roots"]
                if item["kind"] == "import_override"
            ]
            self.assertEqual(
                [item["label"] for item in roots],
                [
                    "import_override:1",
                    "import_override:2",
                    "import_override:3",
                    "import_override:4",
                ],
            )
            self.assertEqual(self._root_for_path(private, first)["label"], "import_override:3")
            self.assertEqual(self._root_for_path(private, second)["label"], "import_override:4")
            repeated = self._fingerprint(env, sys_path=sys_path)
            self.assertEqual(_canonical_json(public), _canonical_json(repeated))

    def test_nested_roots_are_independent_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            parent = os.path.join(directory, "parent-root")
            nested = os.path.join(parent, "nested")
            self._write(os.path.join(parent, "top.py"), "top\n")
            self._write(os.path.join(nested, "child.py"), "child\n")
            parent_only = self._fingerprint(
                env, sys_path=env["sys_path"] + [parent]
            )
            parent_and_nested = self._fingerprint(
                env, sys_path=env["sys_path"] + [parent, nested]
            )
            self.assertNotEqual(
                parent_only["digest_sha256"],
                parent_and_nested["digest_sha256"],
            )
            private = self._fingerprint(
                env, sys_path=env["sys_path"] + [parent, nested], private=True
            )
            paths = [item["path"] for item in private["private_roots"]]
            self.assertIn(parent, paths)
            self.assertIn(nested, paths)
            self.assertLess(paths.index(parent), paths.index(nested))

    def test_reordered_external_roots_change_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            first = os.path.join(directory, "override-first")
            second = os.path.join(directory, "override-second")
            self._write(os.path.join(first, "first.py"), "first\n")
            self._write(os.path.join(second, "second.py"), "second\n")
            forward = self._fingerprint(
                env, sys_path=env["sys_path"] + [first, second]
            )["digest_sha256"]
            reverse = self._fingerprint(
                env, sys_path=env["sys_path"] + [second, first]
            )["digest_sha256"]
            self.assertNotEqual(forward, reverse)

    def test_relocated_environment_is_portable(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            env_first = self._env(first)
            result_first = self._fingerprint(env_first)
            shutil.copytree(first, second, dirs_exist_ok=True)
            env_second = self._env(second)
            result_second = self._fingerprint(env_second)
            helper_first = os.path.join(
                env_first["scripts"], "benchmark-integrity.py"
            )
            helper_second = os.path.join(
                env_second["scripts"], "benchmark-integrity.py"
            )
            with open(helper_first, "rb") as handle:
                first_bytes = handle.read()
            with open(helper_second, "rb") as handle:
                second_bytes = handle.read()
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(
                result_first["integrity_source_sha256"],
                result_second["integrity_source_sha256"],
            )
            self.assertEqual(result_first, result_second)

    def test_helper_source_edit_invalidates_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            helper = os.path.join(env["scripts"], "benchmark-integrity.py")
            before = self._fingerprint(env)
            self._write(helper, "# changed fixture\n")
            after = self._fingerprint(env)
            self.assertNotEqual(
                before["integrity_source_sha256"],
                after["integrity_source_sha256"],
            )
            self.assertNotEqual(before["digest_sha256"], after["digest_sha256"])

    def test_broad_roots_are_rejected_without_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            for broad in (os.sep, os.path.expanduser("~")):
                with self.subTest(root=broad):
                    with self.assertRaises(FingerprintError):
                        self._fingerprint(env, include_roots=[broad])

    def test_safe_path_policy_fixture_is_enforced_for_real_probes(self):
        with mock.patch.object(sys, "flags", mock.Mock(safe_path=False)):
            with self.assertRaises(FingerprintError):
                fingerprint_runtime()
        _require_safe_path(mock.Mock(safe_path=True))
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            result = self._fingerprint(env)
            self.assertTrue(result["interpreter"]["safe_path"])

    def test_wrapper_and_explicit_roots_under_home_are_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            home = os.path.join(directory, "home")
            wrapper = os.path.join(home, "stock-vllm")
            explicit = os.path.join(home, "sdk")
            self._write(os.path.join(wrapper, "module.py"), "wrapper\n")
            self._write(os.path.join(explicit, "native.txt"), "sdk\n")
            with mock.patch("os.path.expanduser", return_value=home):
                result = self._fingerprint(
                    env,
                    sys_path=env["sys_path"] + [wrapper],
                    include_roots=[explicit],
                    private=True,
                )
            private_paths = [
                item["path"] for item in result["private_roots"]
            ]
            self.assertIn(env["site"], private_paths)
            self.assertIn(wrapper, private_paths)
            self.assertIn(explicit, private_paths)
            self.assertLess(
                private_paths.index(env["site"]), private_paths.index(wrapper)
            )
            self.assertLess(
                private_paths.index(wrapper), private_paths.index(explicit)
            )
            wrapper_root = self._root_for_path(result, wrapper)
            self.assertEqual(wrapper_root["kind"], "import_override")
            self.assertGreater(wrapper_root["file_count"], 0)
            explicit_roots = [
                item
                for item in result["content_roots"]
                if item["kind"] == "explicit"
            ]
            self.assertEqual(
                [item["label"] for item in explicit_roots], ["explicit:1"]
            )
            self.assertGreater(explicit_roots[0]["file_count"], 0)

    def test_zip_import_root_is_hashed_and_missing_zip_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            archive = os.path.join(directory, "wheels", "stock.zip")
            os.makedirs(os.path.dirname(archive), exist_ok=True)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("stock_module.py", "stock-v1\n")
            base_path = env["sys_path"] + [archive]
            before = self._fingerprint(env, sys_path=base_path)
            private = self._fingerprint(
                env, sys_path=base_path, private=True
            )
            archive_root = self._root_for_path(private, archive)
            self.assertEqual(archive_root["kind"], "import_override")
            self.assertEqual(archive_root["file_count"], 1)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("stock_module.py", "stock-v2\n")
            after = self._fingerprint(env, sys_path=base_path)
            self.assertNotEqual(before["digest_sha256"], after["digest_sha256"])
            missing = os.path.join(directory, "missing.zip")
            skipped = self._fingerprint(env, sys_path=base_path + [missing])
            self.assertEqual(after["digest_sha256"], skipped["digest_sha256"])

    def test_no_site_discovers_only_active_site_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            without_site_path = [
                path for path in env["sys_path"] if path != env["site"]
            ]
            active = self._fingerprint(env, sys_path=without_site_path)
            self.assertIn("site_packages:1", [item["label"] for item in active["content_roots"]])
            inactive = self._fingerprint(
                env, sys_path=without_site_path, no_site=True
            )
            self.assertNotIn("site_packages:1", [item["label"] for item in inactive["content_roots"]])
            on_path = self._fingerprint(
                env,
                sys_path=without_site_path + [env["site"]],
                no_site=True,
            )
            self.assertIn("site_packages:1", [item["label"] for item in on_path["content_roots"]])

    def test_private_roots_are_extra_and_not_in_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            public = self._fingerprint(env)
            private = self._fingerprint(env, private=True)
            self.assertNotIn("private_roots", public)
            self.assertNotIn("private_executable", public)
            self.assertIn("private_roots", private)
            self.assertEqual(private["private_executable"], env["executable"])
            self.assertNotIn(env["base"], _canonical_json(public))
            self.assertEqual(public["digest_sha256"], private["digest_sha256"])

    def test_implicit_cwd_and_script_directory_are_excluded_without_safe_path(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env, safe_path=False)["digest_sha256"]
            self._write(os.path.join(env["project"], "secret.txt"), "not an import\n")
            self._write(os.path.join(env["scripts"], "secret.txt"), "not an import\n")
            after = self._fingerprint(env, safe_path=False)["digest_sha256"]
            self.assertEqual(before, after)
            private = self._fingerprint(
                env, safe_path=False, private=True
            )
            paths = [item["path"] for item in private["private_roots"]]
            self.assertNotIn(env["project"], paths)
            self.assertNotIn(env["scripts"], paths)

    def test_safe_path_binds_cwd_and_script_directory_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env)
            private = self._fingerprint(env, private=True)
            paths = [item["path"] for item in private["private_roots"]]
            self.assertIn(env["project"], paths)
            self.assertIn(env["scripts"], paths)
            self.assertGreater(
                self._root_for_path(private, env["project"])["file_count"], 0
            )
            self.assertGreater(
                self._root_for_path(private, env["scripts"])["file_count"], 0
            )
            self._write(
                os.path.join(env["project"], "cwd-root.txt"), "cwd root\n"
            )
            with_cwd_file = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before["digest_sha256"], with_cwd_file)
            os.unlink(os.path.join(env["project"], "cwd-root.txt"))
            self._write(
                os.path.join(env["scripts"], "script-root.txt"), "script root\n"
            )
            with_script_file = self._fingerprint(env)["digest_sha256"]
            self.assertNotEqual(before["digest_sha256"], with_script_file)

    def test_distributions_are_read_without_importing_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            self._write(
                os.path.join(env["site"], "torch", "__init__.py"),
                "raise RuntimeError('package must not be imported')\n",
            )
            result = self._fingerprint(env)
            self.assertIn(
                {"name": "torch", "version": "2.0.0", "root_label": "site_packages:1"},
                result["distributions"],
            )

    def test_interpreter_hash_change_alters_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self._env(directory)
            before = self._fingerprint(env)
            self._write(env["executable"], b"fake-python-binary-v2")
            after = self._fingerprint(env)
            self.assertNotEqual(
                before["interpreter"]["executable_sha256"],
                after["interpreter"]["executable_sha256"],
            )
            self.assertNotEqual(before["digest_sha256"], after["digest_sha256"])


def _run_self_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_SelfTests)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stderr).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return _run_self_tests()
    try:
        result = fingerprint_runtime(
            include_roots=args.include_root, private=args.private_roots
        )
        sys.stdout.write(_canonical_json(result) + "\n")
        return 0
    except Exception as exc:
        _write_error("%s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
