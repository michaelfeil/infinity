#!/bin/sh
# Copyright Infinity Core Contributors. Licensed under the scope in ../LICENSE.
# Collect repository license documents for the Linux container builds.
set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p -- "$1"
output_dir=$(CDPATH= cd -- "$1" && pwd)
case "$output_dir" in
    "$repo_root"|"$repo_root/backends"|"$repo_root/backends/"*|"$repo_root/LICENSES"|"$repo_root/LICENSES/"*)
        echo "Output must not overwrite the source license files." >&2
        exit 2
        ;;
esac

cd "$repo_root"
cp -- LICENSE NOTICE LICENSING.md THIRD_PARTY_NOTICES.md "$output_dir/"
cp -R -- LICENSES "$output_dir/"
find backends -type f \( -iname 'LICENSE*' -o -iname 'NOTICE*' \
    -o -iname 'COPYING*' -o -name 'AUTHORS' -o -iname '*provenance*' \) \
    -exec cp --parents -- '{}' "$output_dir/" \;
