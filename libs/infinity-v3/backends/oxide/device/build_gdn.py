# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Cold-build the public raw8 and aux3 Qwen3.5 GDN cubin bundles."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
NATIVE_DIR = HERE / "native_gdn"
AOT_DIR = HERE / "native_gdn_aot"
REPO_LICENSE = HERE.parents[2] / "LICENSE"

ARCHITECTURES = ("sm_90", "sm_100", "sm_120")
REQUIRED_FLAGS = (
    "-std=c++17", "-O3", "--cubin", "--fmad=true", "--ftz=false",
    "--prec-div=true", "--prec-sqrt=true", "-Xptxas", "-v",
)
RAW_EXPORTS = (
    "qwen35_conv_bf16", "qwen35_gdn_prepare", "qwen35_gdn_recurrent",
    "qwen35_gdn_norm", "infinity_qwen35_packed_conv_fresh",
    "infinity_qwen35_packed_conv_cached", "infinity_qwen35_gdn_cached_lanes8",
    "infinity_qwen35_gdn_cached_lanes16",
)
AUX_EXPORTS = (
    "infinity_qwen35_gdn_aot_pack_bf16",
    "infinity_qwen35_gdn_aot_state_kv_to_vk",
    "infinity_qwen35_gdn_aot_state_vk_to_kv",
)
RAW_SOURCE_NAME = "qwen35_gdn_native.cu"
AUX_SOURCE_NAME = "qwen35_gdn_aot.cu"


class BuildError(RuntimeError):
    """An actionable cold-build failure."""


def _raw_sources():
    return (
        (RAW_SOURCE_NAME, NATIVE_DIR / RAW_SOURCE_NAME),
        ("NOTICE", NATIVE_DIR / "NOTICE"),
        ("LICENSE.candle-MIT", NATIVE_DIR / "LICENSE.candle-MIT"),
    )


def _aux_sources():
    return (
        (AUX_SOURCE_NAME, AOT_DIR / AUX_SOURCE_NAME),
        ("NOTICE", AOT_DIR / "NOTICE"),
        ("LICENSE", REPO_LICENSE),
        ("LICENSE.candle-MIT", NATIVE_DIR / "LICENSE.candle-MIT"),
    )


def _default_nvcc(value):
    if value:
        return str(value)
    toolkit = os.environ.get("CUDA_TOOLKIT_PATH")
    if toolkit:
        return str(pathlib.Path(toolkit) / "bin" / "nvcc")
    return "nvcc"


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with pathlib.Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise BuildError(f"cannot read {path}: {error}") from error
    return digest.hexdigest()


def _check_outputs(raw_output, aux_output):
    # Check lexical paths before resolving, so even dangling output symlinks
    # cannot redirect a build. Normalize aliases for same/nested destinations.
    for flag, path in (("--raw-output", raw_output), ("--aux-output", aux_output)):
        if os.path.lexists(path):
            raise BuildError(f"{flag} must be absent before any compilation: {path}")
    raw_output = raw_output.resolve()
    aux_output = aux_output.resolve()
    if raw_output == aux_output:
        raise BuildError("--raw-output and --aux-output must be different paths")
    if raw_output in aux_output.parents or aux_output in raw_output.parents:
        raise BuildError("--raw-output and --aux-output must not be nested")
    return raw_output, aux_output


def _stage(output, label, sources):
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = pathlib.Path(tempfile.mkdtemp(
            prefix=f".{output.name or 'gdn'}.{label}-", dir=str(output.parent)))
    except OSError as error:
        raise BuildError(f"cannot create {label} staging directory beside {output}: {error}") from error
    try:
        hashes = {}
        for name, source in sources:
            destination = stage / name
            try:
                shutil.copyfile(source, destination)
            except OSError as error:
                raise BuildError(
                    f"required {label} source {name} is unavailable at {source}: {error}") from error
            if not destination.is_file() or destination.stat().st_size == 0:
                raise BuildError(f"required {label} source {name} is empty at {source}")
            hashes[name] = _sha256(destination)
        return stage, hashes
    except BaseException as error:
        failures = _cleanup((stage,))
        if failures:
            raise BuildError(f"{error}; cleanup incomplete: {'; '.join(failures)}") from error
        raise


def _cleanup(paths):
    failures = []
    for path in paths:
        if path is None:
            continue
        try:
            shutil.rmtree(path)
            if os.path.lexists(path):
                failures.append(f"residual directory {path}")
        except (OSError, ValueError) as error:
            failures.append(f"cannot remove {path}: {error}")
    return failures


def _run(argv, description):
    try:
        return subprocess.run(argv, capture_output=True, text=True, check=False)
    except OSError as error:
        raise BuildError(f"cannot run {description} {argv[0]}: {error}") from error


def _toolkit_version(nvcc):
    result = _run([str(nvcc), "--version"], "CUDA compiler")
    text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise BuildError(f"CUDA compiler version query failed: {text.strip() or 'no output'}")
    match = re.search(r"release\s+([0-9][0-9A-Za-z._-]*)", text, re.IGNORECASE)
    version = match.group(1) if match else next(
        (line.strip() for line in text.splitlines() if line.strip()), "")
    if not version:
        raise BuildError("CUDA compiler --version returned no version")
    return version


def _torch_symbol(name):
    lowered = name.lower()
    return ("torch" in lowered or "at::" in lowered or "c10::" in lowered
            or (name.startswith("_Z") and ("N2at" in name or "N3c10" in name))
            or lowered.startswith(
        ("c10", "_zn3c10", "at::", "_zn2at", "aten", "_zn4aten"))
            )


def _elf_kernels(path):
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as error:
        raise BuildError(f"cannot read compiled cubin {path}: {error}") from error
    if (len(data) < 64 or data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1
            or struct.unpack_from("<H", data, 18)[0] != 190):
        raise BuildError(f"compiled cubin {path} is not a nonempty little-endian ELF64 file")
    try:
        shoff, = struct.unpack_from("<Q", data, 40)
        shentsize, shnum, _ = struct.unpack_from("<HHH", data, 58)
    except struct.error as error:
        raise BuildError(f"compiled cubin {path} has a truncated ELF header") from error
    if shoff == 0 or shnum == 0 or shentsize < 64 or shoff + shentsize * shnum > len(data):
        raise BuildError(f"compiled cubin {path} has no usable ELF section table")
    symtabs = []
    for index in range(shnum):
        header = shoff + index * shentsize
        sh_type, = struct.unpack_from("<I", data, header + 4)
        sh_offset, sh_size = struct.unpack_from("<QQ", data, header + 24)
        sh_link, = struct.unpack_from("<I", data, header + 40)
        sh_entsize, = struct.unpack_from("<Q", data, header + 56)
        if sh_type != 2:
            continue
        if (sh_entsize < 24 or sh_size % sh_entsize or sh_offset + sh_size > len(data)
                or sh_link >= shnum):
            raise BuildError(f"compiled cubin {path} has a malformed ELF symbol table")
        symtabs.append((sh_offset, sh_size, sh_link, sh_entsize))
    if not symtabs:
        raise BuildError(f"compiled cubin {path} has no ELF symbol table")
    kernels, torch = [], []
    for sh_offset, sh_size, sh_link, sh_entsize in symtabs:
        table = shoff + sh_link * shentsize
        string_type, = struct.unpack_from("<I", data, table + 4)
        string_offset, string_size = struct.unpack_from("<QQ", data, table + 24)
        if string_type != 3 or string_offset + string_size > len(data):
            raise BuildError(f"compiled cubin {path} has a malformed ELF string table")
        strings = data[string_offset:string_offset + string_size]
        for entry in range(sh_offset, sh_offset + sh_size, sh_entsize):
            name_offset, info, other, section = struct.unpack_from("<IBBH", data, entry)
            if name_offset >= len(strings):
                raise BuildError(f"compiled cubin {path} has an invalid ELF symbol name")
            end = strings.find(b"\0", name_offset)
            if end < 0:
                raise BuildError(f"compiled cubin {path} has an unterminated ELF symbol name")
            name = strings[name_offset:end].decode("utf-8", "replace")
            if not name:
                continue
            if _torch_symbol(name):
                torch.append(name)
            # CUDA entrypoints are global STT_FUNC symbols with
            # STO_CUDA_ENTRY(0x10); ordinary helper functions are not exports.
            if (info & 0xF) == 2 and (info >> 4) == 1 and other & 0x10 and 0 < section < shnum:
                kernels.append(name)
    return kernels, sorted(set(torch))


def _validate_cubin(cubin, exports, label):
    kernels, torch = _elf_kernels(cubin)
    if torch:
        raise BuildError(f"{label} cubin {cubin.name} records Torch symbols: {', '.join(torch)}")
    if len(kernels) != len(set(kernels)):
        raise BuildError(f"{label} cubin {cubin.name} has duplicate CUDA entrypoint symbols")
    actual, expected = set(kernels), set(exports)
    if actual != expected or len(kernels) != len(exports):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BuildError(
            f"{label} {cubin.parent.name} kernel symbols do not match the contract; "
            f"missing={missing}, extra={extra}, count={len(kernels)}, expected={len(exports)}")


def _compile(stage, hashes, source_name, exports, nvcc, label):
    source = stage / source_name
    baseline = hashes[source_name]
    records = []
    for architecture in ARCHITECTURES:
        directory = stage / architecture
        try:
            directory.mkdir()
        except OSError as error:
            raise BuildError(f"cannot create {label} {architecture} directory: {error}") from error
        cubin = directory / "kernels.cubin"
        argv = [str(nvcc), *REQUIRED_FLAGS, f"-arch={architecture}", str(source),
                "-o", str(cubin)]
        result = _run(argv, f"{label} {architecture} CUDA compiler")
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no compiler output").strip()
            raise BuildError(
                f"{label} {architecture} compilation failed; argv={argv}; output={detail}")
        if not cubin.is_file() or cubin.stat().st_size == 0:
            raise BuildError(f"{label} {architecture} compiler produced no nonempty cubin")
        _validate_cubin(cubin, exports, label)
        if _sha256(source) != baseline:
            raise BuildError(
                f"{label} source changed after snapshot, refusing compiled manifest: {source}")
        records.append({
            "architecture": architecture,
            "cubin": f"{architecture}/kernels.cubin",
            "cubin_sha256": _sha256(cubin),
            "cubin_bytes": cubin.stat().st_size,
            "exports": list(exports),
            "torch_symbols": [],
        })
    return records


def _manifest(operator, exports, hashes, version, compilations):
    return {
        "abi_version": 1,
        "operator": operator,
        "compile_only": True,
        "device_calls": False,
        "architectures": list(ARCHITECTURES),
        "exports": list(exports),
        "sources": dict(hashes),
        "toolkit": {"version": version},
        "flags": list(REQUIRED_FLAGS),
        "torch_symbols": [],
        "compilations": compilations,
    }


def _write_manifest(stage, manifest):
    try:
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except OSError as error:
        raise BuildError(f"cannot write staged manifest beside {stage}: {error}") from error


def _publish(stage, output, label):
    if os.path.lexists(output):
        raise BuildError(f"refusing to publish {label}: destination appeared at {output}")
    try:
        os.replace(stage, output)
    except OSError as error:
        raise BuildError(f"cannot publish complete {label} bundle to {output}: {error}") from error


def build(raw_output, aux_output, nvcc=None):
    raw_output, aux_output = _check_outputs(
        pathlib.Path(raw_output), pathlib.Path(aux_output))
    nvcc = _default_nvcc(nvcc)
    raw_stage = aux_stage = None
    raw_published = False
    try:
        raw_stage, raw_hashes = _stage(raw_output, "raw", _raw_sources())
        aux_stage, aux_hashes = _stage(aux_output, "aux", _aux_sources())
        # The ABI1 raw sources list is exactly three files. Retain the full
        # Infinity license alongside them without changing that loader schema.
        shutil.copyfile(aux_stage / "LICENSE", raw_stage / "LICENSE")
        version = _toolkit_version(nvcc)
        raw_records = _compile(raw_stage, raw_hashes, RAW_SOURCE_NAME, RAW_EXPORTS, nvcc, "raw")
        aux_records = _compile(aux_stage, aux_hashes, AUX_SOURCE_NAME, AUX_EXPORTS, nvcc, "aux")
        raw_manifest = _manifest(
            "qwen35_gdn_native", RAW_EXPORTS, raw_hashes, version, raw_records)
        raw_manifest["additional_licenses"] = {"LICENSE": _sha256(raw_stage / "LICENSE")}
        _write_manifest(raw_stage, raw_manifest)
        _write_manifest(aux_stage, _manifest(
            "qwen35_gdn_aot_pack", AUX_EXPORTS, aux_hashes, version, aux_records))
        _publish(raw_stage, raw_output, "raw")
        raw_stage = None
        raw_published = True
        _publish(aux_stage, aux_output, "aux")
        aux_stage = None
        return raw_output, aux_output
    except BaseException as error:
        failures = _cleanup((raw_stage, aux_stage, raw_output if raw_published else None))
        if failures:
            raise BuildError(f"{error}; cleanup incomplete: {'; '.join(failures)}") from error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-output", type=pathlib.Path, required=True)
    parser.add_argument("--aux-output", type=pathlib.Path, required=True)
    parser.add_argument("--nvcc", help="nvcc path; defaults to CUDA_TOOLKIT_PATH/bin/nvcc or PATH")
    args = parser.parse_args(argv)
    try:
        build(args.raw_output, args.aux_output, args.nvcc)
    except (BuildError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
