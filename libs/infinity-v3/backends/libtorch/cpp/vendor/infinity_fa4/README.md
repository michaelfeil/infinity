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
metadata pin. Exports require no GPU; SM90/SM100 use their `a` targets.
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

CPU portability regression (no GPU required; CUDA attribute queries are stubbed):

```sh
cc -march=x86-64 -O2 -rdynamic ../../tests/fa4_host_portability.c -ldl -o /tmp/fa4-host-test
qemu-x86_64 -cpu Nehalem /tmp/fa4-host-test /path/to/sm_90/libinfinityfa4.so
```

The alpha.16 host-specialized bundle SIGILLs in this test; portable exports must
initialize all ten kernels successfully. This does not replace GPU parity tests.
