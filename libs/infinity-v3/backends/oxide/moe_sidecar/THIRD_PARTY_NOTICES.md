# Third-party notices

## Public routed-MoE CUDA sources (MIT)

The following public files are reused in place through `--source-root` (or the
installed default `ROOT.parents[2]`); they are not vendored in this package:

- `libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/moe/grouped_gemm.cu`
- `libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/moe/grouped_gemm_hopper.cu`
- `libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/kernels/gemma4_moe.cu`
- `libs/infinity-v3/backends/libtorch/cpp/vendor/candle_kernels/kernels/gemma4_moe_kernels.cuh`
- `libs/infinity-v3/backends/libtorch/cpp/kernels/moe_symbols.h`

They are used unmodified. Their SPDX identifier is `MIT`; the full MIT text is
in `LICENSES/MIT.txt`.

The new ABI header, shim, build helper, and CPU tests in this package also carry
`SPDX-License-Identifier: MIT`.

## NVIDIA CUTLASS (BSD-3-Clause)

CUTLASS is not vendored in this package. A pinned public CUTLASS tree is supplied
by the caller through `build.py --cutlass-root`. CUTLASS is distributed under
the BSD-3-Clause license; its notice is reproduced in
`LICENSES/CUTLASS-BSD-3-Clause.txt`. The build manifest records hashes of the
CUTLASS version header and top-level header used for a build. If the supplied
tree has a root license file, the build copies it to
`LICENSES/CUTLASS-UPSTREAM-LICENSE.txt` and records its hash.

## Reference-only file, not compiled

`libs/infinity-v3/backends/libtorch/cpp/kernels/moe_kernels.cpp` is a public
reference for the original Native dispatch policy. It is not staged, copied into
the build, or compiled. No ATen/c10/Torch code from it is used by this package.

The top-level `LICENSE` and `LICENSING.md` are retained unchanged for the
originating package. The SPDX notices in individual source files remain
authoritative for those files.
