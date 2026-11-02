# Qwen3.5 GDN AOT packing module

Status: source, compile-only producer, pure-CPU index contract tests, and all
three SM90/SM100/SM120 cubins are complete. No CUDA context was created, no
device was queried or launched, and no commit was made. No model weight, gold,
runtime, benchmark, or SDK artifact was read.

- Worktree: `/tmp/infinity-oxide-qwen35-gdn-aot-worktree`
- New source directory: `libs/infinity-v3/backends/oxide/device/native_gdn_aot/`
- Bundle output: `/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels/`
- Task artifacts: this file, `test_aot_index.py`, `kernels/`
- CPU commands used `taskset -c 112-119`, `CUDA_VISIBLE_DEVICES=''`
- No existing host, executor, factory, admission, raw-module, `build.sh`, or
  `gdn_prepared.rs` file was edited.

## Owned files

Added in the worktree (only these three):

| File | SHA-256 |
| --- | --- |
| `qwen35_gdn_aot.cu` | `98828570666a5ab763e22472142136d7933628707fb5a3fb2d609523502d601f` |
| `build.py` | `cd5614d2adcaff198585306be0f630af73e9e5f00ec8863848337124b3369019` |
| `NOTICE` | `4d89b6bb6912e5bfa90a2acc4ab9031a5ba512006e40155efb3818ab70096f37` |

Task artifacts outside the worktree:

| File | SHA-256 |
| --- | --- |
| `/tmp/oxide-qwen35-gdn-aot-pack-v1/test_aot_index.py` | `2050a3c550861fc2609d4100d0ac8cffea41359672f622775c01b060b74ac0e6` |
| `/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels/manifest.json` | `d1cd7aab5324ad3f1cc8e961ee33f016caa5a344e440661ec332d089ba904be8` |

## Preserved raw module and host module

The eight-export raw module and the prepared host module were not modified.
Their hashes before and after this task are identical:

| File | SHA-256 |
| --- | --- |
| `device/native_gdn/qwen35_gdn_native.cu` | `0678b7bba3fc3c80f887d0d58f934a08be033c2613804c538d54cb0f98e6ff4b` |
| `device/native_gdn/build.py` | `6a2a2fe5fb247d5b16198d5e3892f56450f00649af2dbc8100b9232ca9296735` |
| `device/native_gdn/NOTICE` | `55cf47f0510214a52b69f39a00dad492a6961fc12cd3a12764bc79379fdca58c` |
| `device/native_gdn/LICENSE.candle-MIT` | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| `src/gdn_prepared.rs` | `2c25800c2d50b7e763fdc8bff9e5e64f9930a049c95f4dbcaf78b81bc55c7f35` |

The new cubin contains exactly three additive exports, disjoint from the eight
old exports: `infinity_qwen35_gdn_aot_pack_bf16`,
`infinity_qwen35_gdn_aot_state_kv_to_vk`, and
`infinity_qwen35_gdn_aot_state_vk_to_kv`.

## Verified source layout

The actual Native source was read, not guessed:

- `kernels/qwen35_kernels.cu:1288-1367`, `pack_gdn_inputs`:
  - `qk` is F32 `[tokens*key_heads, 256]`; per row Q is `[0:128]` and K is
    `[128:256]`; Q and K are cast to BF16 with the source `T(float)`
    conversion (round-to-nearest-even).
  - `values`/conv is BF16 `[tokens, 2*key_heads*128 + value_heads*128]`; V is
    copied from the tail beginning at `2*key_heads*128`.
  - `gates`/prepared A/B is F32 `[tokens, 2*value_heads]` with alpha in
    `[0, value_heads)` and beta in `[value_heads, 2*value_heads)`. The logical
    `(alpha, beta)` pair for a token/head is therefore **not adjacent** in
    memory. A literal interleaved `[tokens*value_heads, 2]` interpretation is
    wrong. This module uses the actual two-half source layout:
    `token*ab_stride + gate*value_heads + head`.
- `models/qwen35_models.cpp` and `kernels/flashinfer_gdn.cpp`: native resident
  FP32 state is `[batch, value_heads, key=128, value=128]` (K outer); provider
  FP32 state is `[batch, value_heads, value=128, key=128]` (V outer). The
  initial native state is transposed and optionally gathered with the int64
  `block_cache_sources`-style map. The final provider state is transposed back
  by sequence position with no source gather.

## Exact exported ABI

All extents are 64-bit. The host must overflow-check every product, stride, and
grid extent before launch. No kernel allocates, copies host→device, or
synchronizes.

### `infinity_qwen35_gdn_aot_pack_bf16`

```text
grid: (tokens, ceil(value_heads*128 / threads), 1), block: threads
arguments, in order:
  const float*          qk            F32 [tokens*key_heads, 256]
  const __nv_bfloat16*  conv          BF16 [tokens, 2*key_heads*128 + value_heads*128]
  const float*          prepared_ab   F32 [tokens, 2*value_heads]
  __nv_bfloat16*        q             BF16 [tokens, key_heads, 128]
  __nv_bfloat16*        k             BF16 [tokens, key_heads, 128]
  __nv_bfloat16*        v             BF16 [tokens, value_heads, 128]
  float*                alpha         F32 [tokens, value_heads]
  float*                beta          F32 [tokens, value_heads]
  int                   key_heads     positive
  int                   value_heads   positive, >= key_heads
  int64_t               qk_stride     F32 elements per qk row, normally 256
  int64_t               conv_stride   BF16 elements per conv row, normally
                                      (2*key_heads + value_heads)*128
  int64_t               ab_stride     F32 elements per prepared_ab token row,
                                      normally 2*value_heads
```

Mapping: `q[row, head, c] = bf16(qk[(token*key_heads+head)*qk_stride + 0*128 + c])`,
`k[...] = bf16(qk[... + 128 + c])`,
`v[token, head, c] = conv[token*conv_stride + 2*key_heads*128 + head*128 + c]`,
`alpha[token, head] = prepared_ab[token*ab_stride + head]`,
`beta[token, head] = prepared_ab[token*ab_stride + value_heads + head]`.
Per-thread column guards handle ragged widths and block tails. Only the source
BF16 conversions/copies are performed; alpha/beta are F32 copies.

### `infinity_qwen35_gdn_aot_state_kv_to_vk`

```text
grid: state_tile_count(batch, value_heads, key_dim, value_dim) x (32,32,1)
arguments, in order:
  const uint32_t* source          native FP32 KV payload
                                  [source_count, value_heads, key_dim, value_dim]
  uint32_t*       destination     provider FP32 VK payload
                                  [batch, value_heads, value_dim, key_dim]
  const int64_t*  source_indices  optional [batch] gather; null means identity
  int64_t         batch           output row count
  int64_t         source_count    history row count; indices must be in
                                  [0, source_count)
  int64_t         value_heads     positive
  int64_t         key_dim         positive, normally 128
  int64_t         value_dim       positive, normally 128
```

For output row `b` and source `s = source_indices ? source_indices[b] : b`:
`destination[b,h,v,k] = source[s,h,k,v]`, copied as uint32 bits. Duplicate and
permuted sources are valid. Out-of-range sources are skipped. `batch == 0`
yields `state_tile_count == 0` and no work.

### `infinity_qwen35_gdn_aot_state_vk_to_kv`

```text
grid: state_tile_count(batch, value_heads, key_dim, value_dim) x (32,32,1)
arguments, in order:
  const uint32_t* source        provider FP32 VK payload
                                [batch, value_heads, value_dim, key_dim]
  uint32_t*       destination   native FP32 KV payload
                                [batch, value_heads, key_dim, value_dim]
  int64_t         batch         sequence-indexed row count
  int64_t         value_heads   positive
  int64_t         key_dim       positive, normally 128
  int64_t         value_dim     positive, normally 128
```

`destination[b,h,k,v] = source[b,h,v,k]`, copied as uint32 bits. This is the
final-state direction; it never applies a source gather.

`state_tile_count(batch, value_heads, key_dim, value_dim)` is
`batch * value_heads * ceil(value_dim/32) * ceil(key_dim/32)` and is zero when
any extent is non-positive. Each 32x32 tile has guarded load/store edges, so
dimensions that are not multiples of 32 are handled without padding.

## Build

Compile-only NVCC 13.1.80 (`/usr/local/cuda-13.1/bin/nvcc`), no CUDA context,
`CUDA_VISIBLE_DEVICES=''`, `taskset -c 112-119`:

```sh
CUDA_VISIBLE_DEVICES='' NVCC=/usr/local/cuda/bin/nvcc \
CUOBJDUMP=/usr/local/cuda/bin/cuobjdump \
python3 build.py /tmp/oxide-qwen35-gdn-aot-pack-v1/kernels
```

Flags, identical to the retained raw module:

```text
-std=c++17 -O3 --cubin --fmad=true --ftz=false --prec-div=true \
--prec-sqrt=true -Xptxas -v -arch=sm_90|sm_100|sm_120
```

For each architecture the producer records the exact `nvcc` argv, verifies the
cubin is nonempty, parses `cuobjdump -symbols`, requires exactly the three
exports, and rejects `at::`, `c10`, `libtorch`, `torch::`, or `_zn2at` markers.

## Output manifest

Bundle root: `/tmp/oxide-qwen35-gdn-aot-pack-v1/kernels/`

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `sm_90/kernels.cubin` | 23000 | `dd1288955ad1fb4ac470696ccccff43f8ee44276a5b63724ffb1f29a3c94746a` |
| `sm_100/kernels.cubin` | 30168 | `1a833c281c92b20241b09f973641ae2e23588e8f2692c60aefbd857ed71379f1` |
| `sm_120/kernels.cubin` | 30728 | `689714260ecd59835456b40d5580d7f2dfee83a7ead9a0ea17f89f06d6b52b13` |
| `manifest.json` | 12148 | `d1cd7aab5324ad3f1cc8e961ee33f016caa5a344e440661ec332d089ba904be8` |
| `NOTICE` | 1877 | `4d89b6bb6912e5bfa90a2acc4ab9031a5ba512006e40155efb3818ab70096f37` |
| `LICENSE` | 18907 | `2bd9e20e4b48d6c2c7b551590fadf762fa4077dc6529fb531f66db9587d24ba7` |
| `LICENSE.candle-MIT` | 1023 | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| `index-contract/qwen35_gdn_aot_index_test.cpp` | 8534 | generated CPU harness |
| `index-contract/qwen35_gdn_aot_index_test` | 16864 | generated CPU executable |

`manifest.json` records:

- `abi_version: 1`, `operator: qwen35_gdn_aot_pack`, `compile_only: true`,
  `device_calls: false`
- `architectures: ["sm_90", "sm_100", "sm_120"]`
- `exports`: exactly the three names above
- `flags`: the exact flag vector above
- `toolkit`: resolved nvcc path and full `nvcc --version`
- `sources`: SHA-256 for `qwen35_gdn_aot.cu`, `NOTICE`, `LICENSE`,
  `LICENSE.candle-MIT`
- `upstream`: the three read-only Infinity MIT source paths at revision
  `c4c90b9`
- `abi`: the same exact argument lists and grid formulas shown above
- `index_contract`: pure-CPU harness compiler/command/output
- `raw_module`: the eight preserved exports and the four sibling-file SHA-256
  values, with `untouched_by_this_build: true`
- `compilations`: per architecture: cubin path, SHA-256, byte count, exports,
  empty Torch marker list, exact compile and `cuobjdump` commands, and last
  eight ptxas log lines
- `torch_symbols: []`

## Pure-CPU tests

Run from `/tmp/oxide-qwen35-gdn-aot-pack-v1`:

```sh
CUDA_VISIBLE_DEVICES='' python3 -m unittest -v test_aot_index.py
CUDA_VISIBLE_DEVICES='' ruff check --select F821 \
  /tmp/infinity-oxide-qwen35-gdn-aot-worktree/libs/infinity-v3/backends/oxide/device/native_gdn_aot/build.py \
  test_aot_index.py
```

Result: 7 tests passed; F821 passed. The full ruff check also passes.

The build helper compiles `qwen35_gdn_aot.cu` in
`INFINITY_GDN_AOT_INDEX_ONLY` mode with host `g++ -std=c++17 -O2 -Wall -Wextra
-Werror` and runs a harness that tests the production layout helpers against
independent reference loops:

- pack Q/K/V row and half indexing, conv V tail, and prepared A/B two-half
  layout (asserts `beta - alpha == value_heads`, not an adjacent pair)
- state identity transpose in both directions
- duplicate source gather rows (`{2,0,2,1}`) are bit-identical
- permuted source gathers
- `B == 0` and zero-extent `state_tile_count`
- 64-bit offsets at and beyond `2**32`, with explicit uint32-truncation
  counterexamples for native KV, provider VK, qk, prepared A/B, and conv

Additional Python tests validate the generated manifest, per-architecture
cubin SHA-256/byte counts, exact export and flag lists, required license
records, and that the sibling raw module still has its distinct eight exports
and unchanged hashes.

Limitation: the CUDA kernels themselves were not executed. The tests compile
the shared host/device index helpers, not the kernel bodies. No GPU arithmetic,
bitwise output equality, race freedom, or performance is proven here.

## Out of scope

This task intentionally adds no host wiring and no Rust, so no Cargo target
directory was needed. The exported kernels own no CUDA context, stream, module,
or buffer; they are stateless launch entry points. Context/stream/module/buffer
ownership through GPU completion, typed error propagation, and driver loading
remain in the existing host and the separate loader task. Production Rust was
not edited; the existing panic-free/typed-error host code is unchanged. No
executor, factory, admission policy, model route, AOT provider dispatch,
FlashInfer runtime, or numerical-parity claim is added.
