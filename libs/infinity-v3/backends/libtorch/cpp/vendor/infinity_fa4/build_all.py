"""Build InfinityFA4 ahead of time; serving needs neither Python nor a JIT."""
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
import argparse
import pathlib
import re
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument("--architectures", default="80;86;89;90;100;120")
parser.add_argument("--runtime-dir", type=pathlib.Path)
parser.add_argument("--deberta-exact-mul", action="store_true", help="Include Oxide BF16 exact-mul DeBERTa attention")
parser.add_argument("--alibi", action="store_true", help="Include published Jina8 ALiBi attention")
parser.add_argument("--jinacode", action="store_true", help="Include native JinaCode ALiBi attention")
args = parser.parse_args()
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
    subprocess.run([
        sys.executable, str(pathlib.Path(__file__).with_name("build.py")),
        str(args.output / f"sm_{sm}"), "--arch", arch, "--deberta", "--compile-only",
        "--runtime-dir", str(runtime), "--ffi-root", str(ffi),
        *(["--deberta-exact-mul"] if args.deberta_exact_mul else []),
        *(["--alibi"] if args.alibi else []),
        *(["--jinacode"] if args.jinacode else []),
    ], check=True)
