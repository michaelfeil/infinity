# oxide-moe standalone ABI package (V3 exact qualified flags)

A compact build package for the supplied public/original MIT CUDA routed-MoE
operators. It is **not** a model backend: there are no weights, tokenizers,
runtime outputs, serving bindings, FA4 code, CUDA graphs, autotuning, or Python
serving dependency. The package exposes only a small versioned C ABI over the
existing raw BF16 forward/workspace operators.

The raw CUDA math is used unmodified. The supplied
`source/backends/libtorch/cpp/kernels/moe_kernels.cpp` is reference-only and is
neither staged nor compiled; no ATen/c10/Torch part is copied.

## Supported operators

| ABI model kind | Experts | Top-k | Hidden | Intermediate | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `OXIDE_MOE_MODEL_QWEN3_E128_TOP8` | 128 | 8 | caller-defined | caller-defined | positive multiples of 8; optional renormalize |
| `OXIDE_MOE_MODEL_QWEN35_E256_TOP8` | 256 | 8 | caller-defined | caller-defined | positive multiples of 8; optional renormalize |
| `OXIDE_MOE_MODEL_GEMMA4_E128_TOP8` | 128 | 8 | 2816 | 704 | requires per-expert FP32 scales; always renormalizes |

In `oxide_moe_model_info`, a zero hidden/intermediate means the caller supplies
the dimension. Gemma4 reports its fixed dimensions.

## ABI

`include/oxide_moe.h` is C-compatible, fixed-width and has no CUDA, Torch, ATen,
c10 or Python include. Input structs carry `struct_size` and `abi_version`;
`OXIDE_MOE_ABI_VERSION` is `1`.

Exported functions (and nothing else):

| Function | CUDA/GPU use | Purpose |
| --- | --- | --- |
| `oxide_moe_abi_version` | none | Return the fixed ABI version |
| `oxide_moe_query_build` | none | Return build arch, default kernel, kernel/model masks, flags |
| `oxide_moe_query_model` | none | Return fixed model metadata |
| `oxide_moe_query_workspace` | none | Pure-CPU required-workspace query for a shape/kernel |
| `oxide_moe_prepare` | none | Validate shape, resolve auto kernel, emit a fixed plan |
| `oxide_moe_forward_bf16` | raw operator only | Validate and call the selected raw BF16 forward |

Status values are typed integers in `oxide_moe_status_t`; no exception crosses
the C ABI. `oxide_moe_forward_bf16` validates before calling the raw operator and
returns `OXIDE_MOE_STATUS_KERNEL_ERROR` if the raw operator returns non-zero,
preserving the raw integer through the optional `out_kernel_status` argument.

A plan is architecture- and kernel-specific. `oxide_moe_prepare` resolves
`OXIDE_MOE_KERNEL_AUTO` once, at preparation time. Forward has no fallback path:
an unavailable kernel or a plan built for another architecture is rejected.

The forward buffers are wholly external: input, logits, gate-up/down weights,
optional Gemma4 scales, output, workspace and stream are owned by the caller.
The stream is passed as the opaque `void*` value of a `cudaStream_t`; a null
value means the CUDA default stream. The workspace base must be 16-byte aligned;
the raw workspace carver does not realign the caller's base. The shim adds no
allocation, synchronization, copy, graph capture, autotuning or fallback; known
raw-unit behavior is recorded below.

### Workspace query is CPU-only

Both supplied workspace helpers (`routed_workspace_bytes` in the portable and
Hopper units) are host-only integer arithmetic. The shim calls the selected
helper directly for `oxide_moe_query_workspace` and `oxide_moe_prepare`. Those
queries do not call CUDA runtime/device APIs or initialize CUDA, although the
shared library itself links normal CUDA runtime symbols.

## Architecture policy

| Build arch | Raw paths included | Default kernel | Status |
| --- | --- | --- | --- |
| `90` | portable SM80-CUTLASS + SM90-CUTLASS Hopper | Hopper | Buildable and testable |
| `100` | portable SM80-CUTLASS only | Portable | **Compile-only** until Root real-GPU qualification |
| `120` | portable SM80-CUTLASS only | Portable | **Compile-only** until Root real-GPU qualification |

No runtime architecture detection is performed. The architecture is selected by
`build.py --arch`. SM100/SM120 deliberately do not include the Hopper source;
explicit Hopper requests on those builds return
`OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL`.

This package makes **no performance claim** for any architecture. It also makes
no correctness claim for SM100/SM120 beyond a successful compile.

## Build

Requirements for `--build`: a pinned public CUTLASS tree supplied by Root, a
CUDA toolkit whose compiler supports the requested gencode, a host C++ compiler,
and `readelf` for ELF verification. The build does not download anything or
query a GPU.

The package does not vendor the raw CUDA sources. When installed at the public
target `libs/infinity-v3/backends/oxide/moe_sidecar`, the default source root is
`ROOT.parents[2]` = `libs/infinity-v3`. The package reuses the public raw files
in place. If the package is copied outside that checkout, the default source
root is unavailable and the builder returns a typed error requiring
`--source-root <path-to-libs/infinity-v3>`.

```sh
# Installed public checkout: default source root works.
python3 build.py --check-only

# Copied/standalone package: explicit public source root required.
python3 build.py --check-only --source-root /path/to/libs/infinity-v3

# Check with a prospective CUTLASS tree; still no CUDA invocation.
python3 build.py --check-only --source-root /path/to/libs/infinity-v3 \
  --cutlass-root /path/to/cutlass --arch 90

# One standalone shared library.
python3 build.py --build --source-root /path/to/libs/infinity-v3 \
  --cutlass-root /path/to/cutlass \
  --output /tmp/oxide-build-sm90 \
  --arch 90

# All three; output must be fresh/empty.
python3 build.py --build --source-root /path/to/libs/infinity-v3 \
  --cutlass-root /path/to/cutlass \
  --output /tmp/oxide-build-all \
  --arch all
```

`--output` must not exist or must be empty; `--check-only --output` applies the
same freshness rule without creating anything. The script stages the package
source, records SHA-256 hashes and license hashes in `SOURCE_MANIFEST.json`,
builds `liboxide_moe_sm{90,100,120}.so`, and writes `BUILD_MANIFEST.json` with
exact build arguments, gencode, CUTLASS header hashes, output hashes, and ELF
verification. `CUDACXX`, `CXX`, `--nvcc` and `--cxx` can override the tool paths.

After each link, `build.py` runs `readelf -d` and `readelf -W --dyn-syms`. It
rejects Torch/ATen/c10/Python shared-library dependencies, missing required
`oxide_moe_*` exports, and any exported raw symbol. The verification is recorded
in the build manifest. No hardware performance claim is made. If the pinned
CUTLASS tree contains a root license file, it is copied to
`LICENSES/CUTLASS-UPSTREAM-LICENSE.txt` and hashed in the source manifest.

Fixed gencode choices are:

- `90`: `compute_90a/sm_90a` (the Hopper source requires the `a` target)
- `100`: `compute_100/sm_100` (portable path)
- `120`: `compute_120/sm_120` (portable path)

The raw symbols are prefixed through the supplied `moe_symbols.h` (passed with
`-include`) and hidden with `-fvisibility=hidden` plus a linker version script.
Only the six `oxide_moe_*` ABI symbols are exported, so multiple architecture
libraries can be loaded in one process without raw-symbol ODR collisions.

## CPU tests

The tests do not include CUDA headers, execute kernels, query a GPU or use the
network. They compile as C11, `dlopen` the built libraries and exercise the pure
CPU ABI surface, including:

- ABI/version and struct-size checks;
- model metadata and architecture/kernel masks;
- workspace numerical bounds, monotonic growth, overflow and shape rejection;
- invalid model kind, invalid kernel kind, and per-architecture SM policy;
- auto/explicit kernel selection;
- plan preparation and stale/tampered plan rejection;
- forward argument validation before any kernel call;
- hidden raw symbols and same-process loading of multiple architecture libraries.

The separate build-policy test checks the explicit check-only default, fresh
output refusal, missing/invalid CUTLASS refusal, and the check-only JSON claims
(`network_access=false`, `gpu_query=false`, `cuda_invoked=false`). It never
invokes CUDA.

```sh
# Installed public checkout:
python3 tests/build_policy_tests.py

# Copied/standalone package:
OXIDE_MOE_SOURCE_ROOT=/path/to/libs/infinity-v3 python3 tests/build_policy_tests.py

python3 tests/run_cpu_tests.py --build-dir /tmp/oxide-build-sm90
python3 tests/run_cpu_tests.py --build-dir /tmp/oxide-build-all --arch 90,100,120
```

The test executable can also be compiled manually:

```sh
cc -std=c11 -O2 -Wall -Wextra -Iinclude tests/cpu_tests.c -ldl -o /tmp/cpu_tests
/tmp/cpu_tests /tmp/oxide-build-sm90/liboxide_moe_sm90.so
```

Loading a built `.so` may resolve its normal CUDA runtime dependency, but the
tests never initialize or query a device.

## Existing hot device queries (Root follow-up)

The raw math was not edited. These existing queries are called from the hot
forward path and are recorded rather than silently changed:

- `source/.../moe/grouped_gemm_hopper.cu` calls `cudaGetDevice` and
  `cutlass::KernelHardwareInfo::make_kernel_hardware_info` for every Hopper
  forward (around lines 89-93). That hardware-info helper can read device
  properties.
- The portable path constructs and initializes `cutlass::gemm::device::GemmGrouped`
  per forward (around lines 66-70); CUTLASS may perform its own device
  introspection internally. This was not audited or modified.
- Both raw forward paths call `cudaGetLastError` before returning. This is an
  error check, not a metadata query, and the shim adds no synchronization.
- Reference-only `moe_kernels.cpp` uses ATen device properties but is not
  compiled.
- The shim itself makes no allocation. The portable raw unit passes `nullptr`
  as the CUTLASS `GemmGrouped::initialize` workspace; CUTLASS may allocate an
  internal GEMM workspace if its reported workspace size is non-zero. This is
  existing raw behavior, not a shim allocation, and is a Root follow-up if a
  strictly zero-allocation portable forward is required.

These are outside the shim. Removing or caching them would require a separate
math/policy change and real-GPU qualification.

## Layout

```text
include/oxide_moe.h                 fixed versioned C ABI
src/oxide_moe_shim.cpp              original validation/forward shim
build.py                            check-only default, staging, hashing, builds
tests/cpu_tests.c                   CPU ABI tests
tests/run_cpu_tests.py              test compiler/runner
tests/build_policy_tests.py         check-only/fresh-output policy tests
THIRD_PARTY_NOTICES.md              source MIT/CUTLASS notices
LICENSES/MIT.txt                    MIT text
LICENSES/CUTLASS-BSD-3-Clause.txt   CUTLASS license text
source-root/...                     public raw sources resolved in place, not vendored
```
