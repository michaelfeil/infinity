# InfinityFA4

Native LibTorch adapter for Dao-AILab FlashAttention-4 (BSD-3-Clause), adapted
from `michaelfeil/candle-flash-attn-v4` (MIT). Pinned revisions and changes are
recorded in `provenance.json`; retain the accompanying licenses.

With the project's pinned CUDA Torch SDK installed, run from this directory:

```sh
python -m pip install -r requirements-build.txt
python -m pip install --no-deps -r requirements-sources.txt
python build_all.py /path/to/infinityfa4 --architectures '80;86;89;90;100;120'
export INFINITY_FA4_ROOT=/path/to/infinityfa4
```

The separate source install preserves tested CuTe 4.7.1 despite Quack's older
metadata pin. SM90/SM100 use their `a` targets. Compile-only export uses fake
tensors and `no_jit_engine=True`; dependency imports can still initialize a
build GPU. Export qualification must independently verify that no attention
kernel launches occurred. The external `CUTE_DSL_PTXAS_PATH` loader hook is
unset before imports in this mode to prevent foreign-target library loading.
The NVIDIA Dockerfile builds these bundles automatically. CMake installs native
libraries, manifests and license notices; serving needs no Python or JIT.

See [backend documentation](../../../../../docs/libtorch-backend.md#infinityfa4)
for supported shapes and `INFINITY_TORCH_FA4=0`. Native regression tests are
registered with `INFINITY_BUILD_NATIVE_TESTS=ON` and `INFINITY_RUN_CUDA_TESTS=ON`.

Host wrappers are exported for baseline x86-64 (or generic AArch64), independently
of the GPU SM target. Linking rejects exports without matching host provenance;
x86-64 builds also reject AVX/VEX/EVEX instructions in the wrapper objects and
library. Re-export old bundles instead of reusing build-host-specialized objects.
The LibTorch runtime falls back when a bundle lacks the portable-host marker.

All supported architecture bundles export noncausal head-width-256 GQA4 attention
for Qwen3.5 decisions across packed lengths. SM8x/SM120 use a 64x64 tile to fit
their shared-memory limits. SM100 uses its dedicated head256 ABI for self-attention;
independent K/V boundaries retain the existing provider. Older bundles fall back
through the existing capability mask.

For the Oxide BF16 DeBERTa fast path, add `--deberta-exact-mul` to `build.py
--deberta` or `build_all.py`. This appends two exports without moving ordinary
profile indexes or replacing the normalized DeBERTa exports. Runtime deployment
still needs only the native libraries and manifest.

The hooks accept raw BF16 relative matmul outputs. They explicitly perform
`mul.rn.f32`, round the result to BF16 with ties-to-even, then widen it before
retaining the existing BF16 relative-sum and QK-sum grouping. For the two BF16
materialized divisors 13.875=111/8 and 11.3125=181/16, the FP32 reciprocal bits are
`0x3d939a86` and `0x3db509e7`. This substitution is valid after that BF16 boundary
for BF16 inputs; it is not a general FP32 division replacement.

For finite inputs, the odd denominators bound the exact quotient's distance
from a BF16 midpoint below by a BF16 unit divided by twice the denominator.
The FP32 reciprocal/multiplication rounding error is smaller than this gap,
including the subnormal result range. Signed zero and infinities preserve their
classes. The exhaustive CPU regression in Oxide's `deberta_reciprocal` test
checks all 65,536 BF16 patterns for both constants, including NaN conversion.
GPU instruction and model equivalence remain separate qualification steps;
compilation alone establishes neither runtime quality nor performance.
