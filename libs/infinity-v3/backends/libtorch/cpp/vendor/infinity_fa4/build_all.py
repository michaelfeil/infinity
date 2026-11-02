"""Build InfinityFA4 ahead of time; serving needs neither Python nor a JIT."""
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
import argparse
import json
import pathlib
import re
import subprocess
import sys

import build_contract

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--architectures", default="90;100;120")
parser.add_argument("--runtime-dir", type=pathlib.Path)
parser.add_argument("--deberta-exact-mul", action=argparse.BooleanOptionalAction, default=True, help="Include Oxide BF16 exact-mul DeBERTa attention (default: on)")
parser.add_argument("--alibi", action=argparse.BooleanOptionalAction, default=True, help="Include published Jina8 ALiBi attention (default: on)")
parser.add_argument("--jinacode", action=argparse.BooleanOptionalAction, default=True, help="Include native JinaCode ALiBi attention (default: on)")
args = parser.parse_args()
disabled_helpers = [
    flag for enabled, flag in (
        (args.deberta_exact_mul, "--no-deberta-exact-mul"),
        (args.alibi, "--no-alibi"),
        (args.jinacode, "--no-jinacode"),
    ) if not enabled
]
if disabled_helpers:
    parser.error(
        "canonical26 requires all 8 helper families; "
        f"these disable flags are not supported: {', '.join(disabled_helpers)}")
targets = []
for arch in args.architectures.split(";"):
    match = re.fullmatch(r"(80|86|89|90|100|120)a?(?:-real|-virtual)?", arch)
    if not match:
        parser.error(f"Unsupported InfinityFA4 architecture: {arch}")
    sm = match[1]
    if sm not in targets:
        targets.append(sm)

import cutlass
import tvm_ffi

runtime = args.runtime_dir or pathlib.Path(cutlass.__file__).resolve().parents[2] / "cu13/lib"
ffi = pathlib.Path(tvm_ffi.__file__).resolve().parent
for sm in targets:
    arch = f"sm_{sm}" + ("a" if sm in {"90", "100"} else "")
    output = args.output / f"sm_{sm}"
    subprocess.run([
        sys.executable, str(pathlib.Path(__file__).with_name("build.py")),
        str(output), "--arch", arch, "--deberta", "--compile-only",
        "--runtime-dir", str(runtime), "--ffi-root", str(ffi),
        *(["--deberta-exact-mul"] if args.deberta_exact_mul else []),
        *(["--alibi"] if args.alibi else []),
        *(["--jinacode"] if args.jinacode else []),
    ], check=True)
    manifest = json.loads((output / "manifest.json").read_text())
    profile_rows = build_contract.validate_profiles_inc(
        pathlib.Path(__file__).with_name("profiles.inc"))
    profile_names = [row[0] for row in profile_rows]
    flags = {
        "include_deberta": True,
        "include_deberta_exact_mul": args.deberta_exact_mul,
        "include_alibi": args.alibi,
        "include_jinacode": args.jinacode,
    }
    symbols = build_contract.expected_kernel_symbols(profile_names, **flags)
    expected_link_order = [
        f"{symbol}.o" for symbol in build_contract.expected_link_symbols(
            profile_names, **flags)]
    inventory = build_contract.validate_manifest_inventory(manifest, profile_rows)
    if inventory["order_kind"] != "canonical":
        raise RuntimeError(f"{arch}: canonical26 metadata order is not canonical")
    if inventory["observed_symbols"] != symbols:
        raise RuntimeError(f"{arch}: observed canonical26 symbols differ")
    if inventory["declared_symbols"] != symbols:
        raise RuntimeError(f"{arch}: declared canonical26 symbols differ")
    if inventory["link_object_order"] != expected_link_order:
        raise RuntimeError(f"{arch}: canonical26 link object order differs")
    print(f"{arch}: {len(symbols)} observed canonical objects; "
          f"{len(expected_link_order)} preserved link entries", flush=True)
