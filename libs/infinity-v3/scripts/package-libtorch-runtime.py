"""Collect the native server, SDK libraries, and notices for Docker's runtime stage."""
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

messages, site_packages, output = map(Path, sys.argv[1:])
native = [
    Path(message["out_dir"])
    for line in messages.read_text().splitlines()
    if (message := json.loads(line)).get("reason") == "build-script-executed"
    and "infinity-backend-libtorch" in message["package_id"]
]
if len(native) != 1:
    raise RuntimeError(f"Expected one native bridge build, got {native}")
bridge = native[0] / "lib/libtei_torch.so"
elf = subprocess.check_output(["cuobjdump", "--list-elf", str(bridge)], text=True)
actual = set(re.findall(r"\.sm_(\d+[a-z]?)\.", elf))
expected = {
    re.sub(r"-(real|virtual)$", "", arch)
    for arch in os.environ["LIBTORCH_CUDA_ARCHITECTURES"].split(";")
    if not arch.endswith("-virtual")
}
if not expected <= actual:
    raise RuntimeError(f"Native bridge lacks CUDA targets: {expected - actual}")

for directory in ["bin", "lib", "licenses"]:
    (output / directory).mkdir(parents=True, exist_ok=True)
shutil.copy2("target/release/infinity_v3", output / "bin/infinity_v3")
libraries = [bridge, *sorted((site_packages / "torch/lib").glob("*.so*"))]
libraries += sorted((site_packages / "nvidia").rglob("*.so*"))
for source in libraries:
    if "stubs" in source.parts or source.name in {"libcuda.so", "libcuda.so.1", "libtorch_python.so"}:
        continue
    destination = output / "lib" / source.name
    if destination.exists():
        with destination.open("rb") as existing, source.open("rb") as incoming:
            if hashlib.file_digest(existing, "sha256").digest() != hashlib.file_digest(incoming, "sha256").digest():
                raise RuntimeError(f"Conflicting runtime libraries: {source.name}")
    else:
        shutil.copy2(source, destination)
shutil.copytree(native[0] / "share/licenses/tei_torch", output / "licenses/native", dirs_exist_ok=True)
for distribution in sorted(site_packages.glob("*.dist-info")):
    notices = distribution / "licenses"
    if notices.is_dir():
        shutil.copytree(notices, output / "licenses/dependencies" / distribution.name, dirs_exist_ok=True)
    for notice in distribution.iterdir():
        if notice.is_file() and notice.name.upper().startswith(("LICENSE", "NOTICE", "COPYING", "AUTHORS")):
            destination = output / "licenses/dependencies" / distribution.name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(notice, destination / notice.name)
(output / "cuda-architectures.txt").write_text("\n".join(sorted(actual)) + "\n")
(output / "sdk-requirements.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
