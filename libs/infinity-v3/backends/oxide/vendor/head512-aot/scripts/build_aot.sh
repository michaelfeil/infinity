#!/usr/bin/env bash
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
# Build the isolated head_dim=512 AOT provider without Torch or a Python runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HEAD512_OUT:-$ROOT/build}"
CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
NVCC="${NVCC:-$CUDA_HOME/bin/nvcc}"
CXX="${CXX:-g++}"
if [[ -n "${CUTLASS_INCLUDE:-}" ]]; then
  CUTLASS_DIR="$CUTLASS_INCLUDE"
elif [[ -n "${CUTLASS_ROOT:-}" ]]; then
  CUTLASS_DIR="$CUTLASS_ROOT/include"
else
  echo "Set CUTLASS_INCLUDE (or CUTLASS_ROOT) to a CUTLASS/CuTe include tree" >&2
  exit 2
fi
for required in "$NVCC" "$CUTLASS_DIR/cutlass/cutlass.h" "$CUTLASS_DIR/cute/tensor.hpp" "$CUDA_HOME/include/cuda_runtime_api.h"; do
  [[ -e "$required" ]] || { echo "missing build input: $required" >&2; exit 2; }
done
mkdir -p "$OUT"

SOURCE_HASHES=()
hash_file() { printf '%s  %s\n' "$(sha256sum "$1" | awk '{print $1}')" "${1#$ROOT/}"; }
for file in \
  native/head512_launch.cu \
  native/head512_bridge.cpp \
  native/include/head512_provider.h \
  native/include/head512_launch.h \
  native/include/head512_cuda_error.h \
  native/torch_flash_attention/flash.h \
  native/torch_flash_attention/flash_fwd_kernel.h \
  native/torch_flash_attention/flash_fwd_launch_template.h \
  native/torch_flash_attention/kernel_traits.h \
  native/torch_flash_attention/utils.h \
  native/torch_flash_attention/softmax.h \
  native/torch_flash_attention/mask.h \
  native/torch_flash_attention/block_info.h \
  native/torch_flash_attention/dropout.h \
  native/torch_flash_attention/rotary.h \
  native/torch_flash_attention/philox.cuh \
  native/torch_flash_attention/philox_unpack.cuh \
  native/torch_flash_attention/static_switch.h \
  native/torch_flash_attention/hardware_info.h \
  native/torch_flash_attention/namespace_config.h; do
  SOURCE_HASHES+=("$(hash_file "$ROOT/$file")")
done
SOURCE_JSON="$(printf '%s\n' "${SOURCE_HASHES[@]}" | sed 's/\\/\\\\/g; s/"/\\"/g' | awk '{printf "%s\"%s\": \"%s\"", (NR==1?"":","), $2, $1}')"
CUTLASS_HASH="$(sha256sum "$CUTLASS_DIR/cutlass/cutlass.h" | awk '{print $1}')"
CUDA_VERSION="$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -n1)"
[[ -n "$CUDA_VERSION" ]] || CUDA_VERSION="unknown"

build_variant() {
  local arch="$1" dir="$2" variant="$3" qualification="$4"
  local cc block_m warps smem variant_define
  case "$variant" in
    native) block_m=64; warps=4; smem=131072; variant_define="-DINFINITY_HEAD512_KERNEL_NATIVE=1" ;;
    compact) block_m=32; warps=2; smem=98304; variant_define="-DINFINITY_HEAD512_KERNEL_COMPACT=1" ;;
    *) echo "unsupported head512 variant: $variant" >&2; exit 2 ;;
  esac
  case "$arch" in
    90) GENCODE="arch=compute_90,code=sm_90"; CC=90 ;;
    100) GENCODE="arch=compute_100,code=sm_100"; CC=100 ;;
    120) GENCODE="arch=compute_120,code=sm_120"; CC=120 ;;
    *) echo "unsupported head512 architecture: $arch" >&2; exit 2 ;;
  esac
  local build="$OUT/$dir"
  mkdir -p "$build"
  local lib="$build/libinfinity_head512_$dir.so"
  local launch_obj="$build/head512_launch.o"
  local bridge_obj="$build/head512_bridge.o"
  local common_defines=(-DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_BACKWARD -DFLASHATTENTION_DISABLE_SOFTCAP -DFLASHATTENTION_DISABLE_ALIBI -DFLASHATTENTION_DISABLE_UNEVEN_K "$variant_define")
  local includes=(-I"$ROOT/native/include" -I"$ROOT/native/torch_flash_attention" -I"$CUTLASS_DIR" -I"$CUDA_HOME/include")
  "$NVCC" -c "$ROOT/native/head512_launch.cu" -o "$launch_obj" \
    -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda \
    -ccbin "$CXX" -Xcompiler -fPIC -lineinfo -gencode "$GENCODE" \
    "${common_defines[@]}" "${includes[@]}"
  "$CXX" -c "$ROOT/native/head512_bridge.cpp" -o "$bridge_obj" \
    -std=c++17 -O3 -fPIC -pthread "${common_defines[@]}" "${includes[@]}"
  "$CXX" -shared -o "$lib" "$launch_obj" "$bridge_obj" \
    -L"$CUDA_HOME/lib64" -Wl,--no-undefined -Wl,-z,defs -lcudart -pthread
  local needed
  needed="$(readelf -d "$lib" | awk '/NEEDED/ {gsub(/[\[\]]/,"",$NF); print $NF}' | sort -u)"
  printf '%s\n' "$needed" > "$build/dt-needed.txt"
  if printf '%s\n' "$needed" | grep -Eiq 'torch|c10|aten|python'; then
    echo "head512 runtime dependency gate failed for $dir" >&2
    printf '%s\n' "$needed" >&2
    exit 3
  fi
  printf '%s\n' "$needed" | grep -q 'libcudart' || { echo "missing exact cudart dependency in $dir" >&2; exit 3; }
  for symbol in infinity_head512_abi_v2 infinity_head512_variant_v2 infinity_head512_info_v2 infinity_head512_error_v2 infinity_head512_prepare_v2 infinity_head512_enqueue_v2; do
    nm -D --defined-only "$lib" | grep -q " $symbol$" || { echo "missing export $symbol in $dir" >&2; exit 3; }
  done
  if nm -D --defined-only "$lib" | grep -Eq ' infinity_head512_[A-Za-z0-9_]*_v1$'; then
    echo "old ABI v1 exports are forbidden in $dir" >&2
    exit 3
  fi
  local lib_hash
  lib_hash="$(sha256sum "$lib" | awk '{print $1}')"
  cat > "$build/manifest.json" <<JSON
{
  "schema": "infinity-head512-aot/v2",
  "abi_version": 2,
  "cold_prepare_required": true,
  "warm_launch_only": true,
  "architecture": "sm_$arch",
  "compute_capability": $CC,
  "library": "$(basename "$lib")",
  "library_sha256": "$lib_hash",
  "head_dim": 512,
  "dtypes": ["fp16", "bf16"],
  "causal": true,
  "noncausal": true,
  "windows": ["global", "local"],
  "ragged_qk": true,
  "runtime_gqa": true,
  "stride_d": 1,
  "kernel_variant": "$variant",
  "qualification": $qualification,
  "block_m": $block_m,
  "block_n": 32,
  "warps": $warps,
  "dynamic_shared_memory_bytes": $smem,
  "min_optin_shared_memory_bytes": $smem,
  "local_experimental": true,
  "torch_free": true,
  "requires_cudart": true,
  "dt_needed": [$(printf '%s\n' "$needed" | awk '{printf "%s\"%s\"", (NR==1?"":","), $0}')],
  "cuda_version": "$CUDA_VERSION",
  "cutlass_include": "$CUTLASS_DIR",
  "cutlass_cutlass_h_sha256": "$CUTLASS_HASH",
  "nvcc": "$NVCC",
  "cxx": "$CXX",
  "compile_definitions": [$(printf '%s\n' "${common_defines[@]}" | awk '{printf "%s\"%s\"", (NR==1?"":","), $0}')],
  "source_sha256": {$SOURCE_JSON}
}
JSON
  echo "built $lib ($variant variant, qualification=$qualification)"
}

# Fixed four-artifact set: Native sm_90/sm_100 and Compact sm_120 plus the
# H100 math-qualification sm_90_compact build. Keep these directory names
# stable for the loader and parent compile packet.
build_variant 90 "sm_90" native false
build_variant 100 "sm_100" native false
build_variant 120 "sm_120" compact false
build_variant 90 "sm_90_compact" compact true
