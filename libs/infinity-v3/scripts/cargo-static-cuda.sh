#!/usr/bin/env bash
# Build/test with static CUDA toolkit archives. The NVIDIA driver remains dynamic.
set -euo pipefail

# rustc bundles cudarc's static C++ dependency into its rlib before the final
# system linker runs, so its native search path must include the compiler archive.
cxx_archive="$("${CXX:-g++}" -print-file-name=libstdc++.a)"
if [[ ! -f "$cxx_archive" ]]; then
    echo "Static CUDA builds require libstdc++.a from the host C++ compiler." >&2
    exit 1
fi
cxx_library_dir="$(dirname "$cxx_archive")"

# CUDA 13.1 static NVRTC initialization fails silently with LLVM LLD: the API
# reports zero supported architectures. GNU BFD preserves the initialization.
# Preserve caller flags and the repository's default CPU optimization.
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native} -L native=$cxx_library_dir -C link-arg=-fuse-ld=bfd"
exec cargo "$@"
