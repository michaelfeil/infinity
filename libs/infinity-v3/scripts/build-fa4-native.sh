#!/usr/bin/env bash
# Build a pinned architecture-specific bundle without requiring a GPU in the Docker builder.
set -euo pipefail
manifest_path=${1:?Candle Cargo.toml is required}
bundle_path=${2:?Output directory is required}
compute_cap=${3:-90}
case "$compute_cap" in
    90) arch=sm_90a; extra_flags=(--deberta) ;;
    120) arch=sm_120; extra_flags=(--deberta) ;;
    80|86|89) arch="sm_${compute_cap}"; extra_flags=() ;;
    *) echo "Unsupported FA4 compute capability: $compute_cap" >&2; exit 1 ;;
esac
work_path=$(mktemp -d)
trap 'rm -rf "$work_path"' EXIT
revision=$(python3 - "$manifest_path" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
match = re.search(r'^candle-flash-attn-v4\s*=.*?rev\s*=\s*"([0-9a-f]{40})"', text, re.M)
if not match:
    raise SystemExit('FA4 dependency must pin a full commit')
print(match.group(1))
PY
)
git init "$work_path/source"
git -C "$work_path/source" fetch --depth 1 https://github.com/michaelfeil/candle-flash-attn-v4 "$revision"
git -C "$work_path/source" checkout --detach FETCH_HEAD
python3 -m venv "$work_path/venv"
python_bin="$work_path/venv/bin/python"
"$python_bin" -m pip install 'torch==2.13.0'
"$python_bin" -m pip install -r "$work_path/source/requirements-build.txt"
runtime_path=$("$python_bin" - <<'PY'
import importlib.metadata as m
print(m.distribution('nvidia-cutlass-dsl').locate_file('nvidia_cutlass_dsl/cu12/lib'))
PY
)
ffi_path=$("$python_bin" - <<'PY'
import pathlib, tvm_ffi
print(pathlib.Path(tvm_ffi.__file__).parent)
PY
)
"$python_bin" "$work_path/source/scripts/build_aot.py" "$bundle_path" \
    --compile-only --arch "$arch" "${extra_flags[@]}" --runtime-dir "$runtime_path" --ffi-root "$ffi_path"
mkdir -p "$bundle_path/licenses/wrapper"
cp "$work_path/source"/LICENSE* "$bundle_path/licenses/wrapper/"
"$python_bin" - "$bundle_path/licenses" <<'PY'
import importlib.metadata as m, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
for name in ['nvidia-cutlass-dsl', 'apache-tvm-ffi']:
    dist = m.distribution(name)
    for f in dist.files or []:
        if 'license' in f.name.lower() or 'notice' in f.name.lower():
            source = pathlib.Path(dist.locate_file(f))
            if source.is_file():
                target = root / name / str(f)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
PY
# Objects are build artifacts. Serving only needs the shared libraries and provenance.
rm -f "$bundle_path"/*.o
