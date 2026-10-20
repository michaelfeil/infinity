# InfinityFA4

InfinityFA4 adapts the retained MIT native bridge from
`michaelfeil/candle-flash-attn-v4` to LibTorch. The attention kernels are
Dao-AILab FlashAttention-4, BSD-3-Clause. `provenance.json` records the pinned
revisions; the original licenses remain alongside the adapter.

Python, CuTe DSL and PyTorch are build dependencies. Serving loads only
`libinfinityfa4.so`, `libcute_dsl_runtime.so` and `libtvm_ffi.so`.
There is no serving-time Python import or kernel compilation.

## Build

Install the project's pinned CUDA Torch SDK, then:

```sh
python -m pip install -r requirements-build.txt
python -m pip install --no-deps -r requirements-sources.txt
python build_all.py /path/to/infinityfa4 --architectures '80;86;89;90;100;120'
export INFINITY_FA4_ROOT=/path/to/infinityfa4
```

The exporter uses fake tensors and requires no GPU. Every architecture gets
its own native bundle, including FP16 and BF16 DeBERTa exports. SM90 and SM100
use their architecture-specific `a` targets. The bridge selects the correct
export ABI for SM100, including runtime maximum query length. Export and
link failures stop the build.

The separate source install deliberately retains CuTe 4.7.1: Quack 0.5.3's
metadata pins an older development release. This exporter validates its helper
imports with 4.7.1 and does not use Quack GEMM kernels. Do not let dependency
resolution replace the validated export/runtime toolchain.

Native CMake/Cargo builds opt in through `INFINITY_FA4_ROOT`. The NVIDIA
Dockerfile builds the bundles for its requested architectures automatically.
The installer retains runtime dependency notices and manifests alongside the
normal native license bundle. Manifests record kernel, object and library
hashes and the exact tool versions.

## Dispatch and arithmetic

The current standard exports cover global/windowed d64 MHA, causal d128
GQA4, and global d128 GQA2. Other geometries keep the existing attention path.
Head counts and packed sequence lengths are runtime values. Layout checks
preserve real packed rows and aligned strided projection views.

DeBERTa uses a score hook with packed relative-position tables and a signed
distance lookup. Its auxiliary storage grows linearly with token count for
a fixed relative-position span. It does not construct a quadratic attention
bias tensor.

`fa4_source.py` applies checked patches only to a private copy of the pinned
upstream sources: the retained softmax reduction/tile choices, and enabling
the DeBERTa forward score hook in the existing SM8x implementation. Persistent
schedulers honor the exporter's target SM count without querying a build GPU.
That count tunes grid occupancy; persistent loops still cover all runtime tiles.
Backward
and arbitrary SM8x score hooks remain outside this adapter.

Set `INFINITY_TORCH_FA4=0` before startup to retain the original attention
selection. Builds containing the bundles otherwise try InfinityFA4 first for
eligible shapes. Explicit decoder/BERT cuDNN selection retains precedence.
Kernel launch failures propagate; unsupported shapes alone use the original
attention path.

## Verification

`infinity_fa4_test` checks independent attention references in both dtypes,
strided views, unequal query/key lengths, masks, non-default streams, CUDA graph
replay with changed tensors and boundaries, sequence isolation and concurrent
initialization on two devices. Full DeBERTa fixtures check shared/separate position
projections, logarithmic buckets and the auxiliary memory budget against a CPU
reference. A separate test checks the startup disable switch.
Register it with `INFINITY_BUILD_NATIVE_TESTS=ON` and
`INFINITY_RUN_CUDA_TESTS=ON`.

Offline compilation checks code generation and linkage for each target.
Runtime accuracy and performance require native and trained-checkpoint
qualification on that device.
