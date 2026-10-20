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
