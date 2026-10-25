# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Export FlashInfer GDN prefill for supported GPU families; no runtime Python/JIT."""
import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import re
import shutil
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("output", type=pathlib.Path)
p.add_argument("--architectures", default="90;100;120")
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
# Each compiler invocation has its own CuTe target and compiler state.
families = set()
for target in a.architectures.split(";"):
    match = re.fullmatch(r"(\d+)[af]?(?:-real|-virtual)?", target)
    if not match:
        p.error(f"Invalid CUDA architecture: {target}")
    family = int(match[1]) // 10
    if family in (9, 10, 12):
        families.add(family)
for family in sorted(families):
    subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("export.py")),
                    str(a.output), str(family)], check=True)

import cutlass
import tvm_ffi

vendor = pathlib.Path(__file__).parent
sys.path.insert(0, str(vendor.parent / "infinity_fa4"))
from portable_host import validate_host_object
objects = [a.output / f"gdn_{family}_{dtype}.o"
           for family in sorted(families) for dtype in ("fp16", "bf16")]
if not objects:
    raise RuntimeError("No supported FlashInfer GDN targets were exported")
for obj in objects:
    validate_host_object(obj)
ffi = pathlib.Path(tvm_ffi.__file__).parent
runtime = pathlib.Path(cutlass.__file__).resolve().parents[2] / "cu13/lib"
subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", str(vendor / "bridge.cpp"),
                *map(str, objects), *[f"-DINFINITY_GDN_SM{family * 10}=1" for family in sorted(families)], f"-I{ffi / 'include'}", f"-L{ffi / 'lib'}", "-ltvm_ffi",
                f"-L{runtime}", "-lcute_dsl_runtime", "-Wl,-z,defs", "-Wl,-rpath,$ORIGIN",
                "-Wl,-soname,libinfinitygdn.so", "-o", str(a.output / "libinfinitygdn.so")], check=True)
for source in (ffi / "lib/libtvm_ffi.so", runtime / "libcute_dsl_runtime.so"):
    shutil.copy2(source, a.output / source.name)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = dict(abi_version=1, flashinfer=importlib.metadata.version("flashinfer-python"),
                architectures=[json.loads((a.output / f"sm_{family}.json").read_text())
                               for family in sorted(families)],
                libraries={x.name: digest(x) for x in a.output.glob("*.so")})
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
shutil.copy2(vendor / "provenance.json", a.output / "provenance.json")
for name in ("flashinfer-python", "apache-tvm-ffi", "nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-cu13", "nvidia-cutlass-dsl-libs-base"):
    dist = importlib.metadata.distribution(name)
    for file in dist.files or []:
        if ".." not in file.parts and any(part.lower().startswith(("license", "notice", "copying")) for part in file.parts):
            source = pathlib.Path(dist.locate_file(file))
            if source.is_file():
                dest = a.output / "licenses" / name / file
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
