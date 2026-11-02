# Building the Qwen3.5 runtime resources

Run these commands from the repository root in a dedicated build environment.
Use the CUDA toolkit and pinned cargo-oxide compiler described in the Oxide
[device build instructions](../../docs/oxide-backend.md). Builds use no GPU; execution and
quality on each GPU architecture are separate checks.

```sh
# Existing normal Rust device module, including the Qwen3.5 exports.
CUDA_VISIBLE_DEVICES= CARGO_OXIDE=/path/to/pinned/cargo-oxide \
  CUDA_TOOLKIT_PATH=/path/to/cuda \
  libs/infinity-v3/backends/oxide/device/build.sh /path/to/resources/oxide

# New raw recurrence and auxiliary pack/transpose cubins.
CUDA_VISIBLE_DEVICES= python3 libs/infinity-v3/backends/oxide/device/build_gdn.py \
  --raw-output /path/to/resources/gdn-raw \
  --aux-output /path/to/resources/gdn-aux \
  --nvcc /path/to/cuda/bin/nvcc
```

The raw/aux tool uses only Python's standard library and `nvcc`. It compiles
SM90, SM100 and SM120, copies source and license snapshots into each bundle,
and hashes the actual snapshots and cubins in the loader's ABI1 manifests.
The flags preserve FMA, non-FTZ and precise divide/square-root semantics. It
checks CUDA ELF entrypoint symbols, including missing/extra exports and Torch
symbols. Both destination directories must be absent. Completed bundles are
published by separate atomic renames; if the second publish fails the first
is rolled back. If filesystem permissions prevent cleanup, the error names the
residual directory alongside the primary failure. The pair is not a single
filesystem transaction.

The provider and attention libraries already have public source builders. In
a build environment with the project's CUDA Torch SDK installed, install the
pinned build dependencies:

```sh
python3 -m pip install -r libs/infinity-v3/backends/libtorch/cpp/vendor/infinity_fa4/requirements-build.txt
python3 -m pip install --no-deps -r libs/infinity-v3/backends/libtorch/cpp/vendor/infinity_fa4/requirements-sources.txt
CUDA_VISIBLE_DEVICES= python3 libs/infinity-v3/backends/libtorch/cpp/vendor/infinity_fa4/build_all.py \
  /path/to/resources/fa4 --architectures '90;100;120'
CUDA_VISIBLE_DEVICES= python3 libs/infinity-v3/backends/libtorch/cpp/vendor/infinity_gdn/build.py \
  /path/to/resources/gdn-provider --architectures '90;100;120'
```

See the [FA4 build guide](../libtorch/cpp/vendor/infinity_fa4/README.md) for
portable host wrappers, selected-architecture development builds and deployment
linking. The provider builder exports the FlashInfer0.7.2 fresh/initial-state
kernels using the source revision in its `provenance.json`; it writes an ABI2
manifest and copies `libinfinitygdn.so`, TVM-FFI/CuTe runtime dependencies and
licenses. It does not link LibTorch. Python/Torch/CuTe are needed to build those
providers; the Oxide serving process uses precompiled native libraries, with
no runtime Python, LibTorch or JIT.

Configure the resulting directories:

```sh
export INFINITY_OXIDE_KERNEL_MODULE=/path/to/resources/oxide
export INFINITY_FA4_ROOT=/path/to/resources/fa4
export INFINITY_OXIDE_GDN_RAW_KERNELS=/path/to/resources/gdn-raw
export INFINITY_OXIDE_GDN_PROVIDER=/path/to/resources/gdn-provider
export INFINITY_OXIDE_GDN_AUX_KERNELS=/path/to/resources/gdn-aux
```

The normal Qwen3.5 constructor validates these resources cold. The Rust device
module, FA4, raw recurrence, ABI2 provider and auxiliary pack bundles have
different contracts and are not interchangeable. Keep their source revisions
paired with the serving binary; a compiled bundle alone does not qualify model
features, numerical quality or performance.

Retain the unchanged notices and licenses. Both bundles retain a full `LICENSE`;
the raw ABI1 manifest records its unchanged three-source set and names this
extra file under `additional_licenses`. The auxiliary bundle's `LICENSE`
is specifically `libs/infinity-v3/LICENSE` (Infinity Restricted MIT1.0), not
the older MIT license at the repository root. `LICENSE.candle-MIT` retains the
inherited Candle terms. Provider/FA4 dependency licenses remain separate.
