#!/usr/bin/env bash
# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
# CPU-only stub bundles for Rust cold-prepare/cold-load/C-ABI tests. No CUDA
# headers, device compilation, context or kernel execution is involved.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/build-host-stub}"
CC="${CC:-cc}"
SRC1="$(sha256sum "$ROOT/native/head512_launch.cu" | awk '{print $1}')"
SRC2="$(sha256sum "$ROOT/native/head512_bridge.cpp" | awk '{print $1}')"

write_manifest() {
  local build="$1" arch="$2" library="$3" variant="$4" cc="$5"
  local qualification="$6" smem="$7" block_m="$8" warps="$9"
  local hash
  hash="$(sha256sum "$build/$library" | awk '{print $1}')"
  cat > "$build/manifest.json" <<JSON
{
  "schema": "infinity-head512-aot/v2",
  "abi_version": 2,
  "cold_prepare_required": true,
  "warm_launch_only": true,
  "architecture": "sm_$arch",
  "compute_capability": $cc,
  "library": "$library",
  "library_sha256": "$hash",
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
  "dt_needed": ["libcudart.so.12"],
  "cuda_version": "stub",
  "cutlass_include": "stub",
  "cutlass_cutlass_h_sha256": "stub",
  "nvcc": "stub",
  "cxx": "$CC",
  "compile_definitions": ["stub", "-DINFINITY_HEAD512_KERNEL_${variant^^}=1"],
  "source_sha256": {
    "native/head512_launch.cu": "$SRC1",
    "native/head512_bridge.cpp": "$SRC2"
  }
}
JSON
}

build_stub() {
  local arch="$1" dir="$2" variant="$3" cc="$4" qualification="$5"
  local smem="$6" block_m="$7" warps="$8" extra="$9"
  local build="$OUT/$dir"
  local name="${dir##*/}"
  local lib="libinfinity_head512_$name.so"
  local variant_define=()
  if [[ "$variant" == "compact" ]]; then
    variant_define=(-DINFINITY_HEAD512_KERNEL_COMPACT=1)
  fi
  mkdir -p "$build"
  "$CC" -shared -fPIC -std=c11 -I"$ROOT/native/include" "${variant_define[@]}" \
    ${extra:+$extra} "$ROOT/tests/stub_provider.c" -o "$build/$lib"
  write_manifest "$build" "$arch" "$lib" "$variant" "$cc" "$qualification" \
    "$smem" "$block_m" "$warps"
}

# The same four production artifact directories, backed by CPU stubs.
build_stub 90 sm_90 native 90 false 131072 64 4 ""
build_stub 100 sm_100 native 100 false 131072 64 4 ""
build_stub 120 sm_120 compact 120 false 98304 32 2 ""
build_stub 90 sm_90_compact compact 90 true 98304 32 2 ""

# Admission-only scenario bundles. These are never production candidates.
build_stub 90 old_abi/sm_90 native 90 false 131072 64 4 \
  "-DINFINITY_HEAD512_STUB_ABI_V1=1"
build_stub 90 missing_prepare/sm_90 native 90 false 131072 64 4 \
  "-DINFINITY_HEAD512_STUB_NO_PREPARE=1"
build_stub 90 fail_prepare/sm_90 native 90 false 131072 64 4 \
  "-DINFINITY_HEAD512_STUB_PREPARE_FAIL=1"

echo "$OUT"
