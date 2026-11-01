# Standalone MPNet attention source builder

`build.py` is a source-only, reproducible packaging utility for the locally
authored adapter `standalone_mpnet_attention.cu` and the patched
`kernel_forward.h`. It does not vendor the 908 public PyTorch/CUTLASS headers.
It consumes them from a caller-supplied public source tree, verifies every
manifest SHA-256 before compilation, and emits per-architecture shared
libraries plus provenance manifests.

The adapter exports exactly these C symbols:

- `mpnet_attn_prepare_bf16_h12d64`
- `mpnet_attn_validate`
- `mpnet_attn_launch_bf16_h12d64`
- `mpnet_attn_smem_bytes`

The ABI manifest records version 1, BF16, 12 heads, head dimension 64, QKV row
stride 2304, 32 bias buckets, non-causal operation, and supported architectures
`[90, 100, 120]`.

## Inputs

- `public-source-manifest.json`: pinned PyTorch commit
  `5c4886908584029761b579af026dcfb627c84070`, pinned CUTLASS commit
  `e05f953a5b3d38adc240df2ff928e0421c2abba3`, and 908 public file entries.
- `standalone_mpnet_attention.cu`: the adapter source.
- `kernel_forward.h`: the patched header containing `bias_table_ptr` and
  `relative_bucket`.
- `--source-root`: a downloaded public tree shaped as
  `pytorch/...` and `cutlass/...`.

The public tree is read-only input. It is never copied into the package.

## Default check

No CUDA compiler, GPU, or network is used:

```sh
python3 build.py --check-only
```

With a public source tree, every manifest hash is verified:

```sh
python3 build.py --check-only --source-root /path/to/public-tree
```

Explicit, opt-in fetching fills only missing files:

```sh
python3 build.py --check-only --source-root /path/to/public-tree --fetch
```

Fetching is restricted to pinned `raw.githubusercontent.com` URLs. CUTLASS
manifest entries without a URL are derived from
`https://raw.githubusercontent.com/NVIDIA/cutlass/<cutlass_commit>/<path>` after
stripping the leading `cutlass/`. Existing files are hashed first; a corrupt
existing file is an error and is never overwritten. Downloads are limited per
file, hashed before an atomic replace, checked for relative-path and symlink
escapes, and run with at most four workers.

## Build

```sh
python3 build.py --build \
  --source-root /path/to/public-tree \
  --output /path/to/fresh-output \
  --cuda-toolkit /usr/local/cuda-13.1 \
  --arch 90 --arch 100 --arch 120
```

`--output` must be absent or an empty directory. `--arch` accepts `90`, `100`,
and `120`, individually or as comma-separated values. The default toolkit is
`/usr/local/cuda-13.1`.

Compilation is hidden from GPUs with `CUDA_VISIBLE_DEVICES=-1`; no device is
queried. The compile command uses:

```text
-std=c++17 -O3 -shared --cudart=shared
-Xcompiler -fPIC -Xlinker -z -Xlinker defs
-DMPNET_ATTENTION_HAVE_BUCKET_BIAS=1
-DC10_USING_CUSTOM_GENERATED_MACROS
-arch=sm_<XX>
-I <PATCHROOT> -I <output> -I <PUBLIC>/pytorch/aten/src -I <PUBLIC>/pytorch
-I <PUBLIC>/cutlass/include
-lcuda
```

The CUDA stubs library directory is added only when it exists.

The patched header is installed in the output at the public-relative path
`pytorch/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h`.
The first include root is `<output>/pytorch/aten/src` (`PATCHROOT`), which
overrides the public `<ATen/...>` header while preserving native include
spelling.

## Output layout

```text
<output>/
  manifest.json
  NOTICE
  licenses/LICENSE-PyTorch-BSD3
  licenses/LICENSE-CUTLASS-BSD3
  licenses/NOTICE
  src/standalone_mpnet_attention.cu
  pytorch/aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h
  sm_90/libmpnet_attention.so
  sm_90/manifest.json
  sm_100/libmpnet_attention.so
  sm_100/manifest.json
  sm_120/libmpnet_attention.so
  sm_120/manifest.json
```

Each `sm_XX/manifest.json` records the ABI, architecture, PyTorch/CUTLASS pins,
four exported symbols, library SHA-256, exact build command, `nvcc` version,
hash provenance, license paths, and qualification status. The top-level
`manifest.json` indexes all built artifacts.

The build always audits the produced ELF with `readelf -d` and `nm`. It rejects
forbidden `torch`, `c10`, and `ATen` dependencies, undefined `at::`/`torch`/`c10`
symbols (demangled), and any missing ABI export. The result is a cold,
header-only build: it does not import PyTorch, link LibTorch/c10 runtime
libraries, or require a GPU at compile time.

License files are copied from the verified public tree (`pytorch/LICENSE` and
`cutlass/LICENSE.txt`) when package-local copies are absent. `NOTICE` states the
source provenance.

## Tests

```sh
python3 -m unittest -v test_build.py
```

The standard-library tests need no CUDA toolkit, compiler, GPU, or network.
They cover valid, missing, and corrupt manifests; manifest formatting; absolute,
parent-traversal, and symlink paths; hash verification; atomic fetch failures;
refusal to overwrite corrupt files; unpinned URLs; byte limits; default
no-network checking; unsupported architectures; fresh output directories;
forbidden ELF dependencies and symbols; and manifest layout.

## Qualification

This utility only reproduces a build from pinned public sources. It makes no
numerical-accuracy claim, does not compare outputs against a reference, and
makes no performance claim.
