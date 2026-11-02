# Qwen3-VL e582 device bundle / old90 preservation

Author did not run NVCC, cargo-oxide, a GPU, or a device compile. This packet is
source-only.

## Symbol sets

- Pre-existing public e582 exports: **90**
- Additive Qwen3-VL exports: **6**
  - `infinity_decoder_rows_bf16`
  - `infinity_qwen3vl_add_bf16`
  - `infinity_qwen3vl_pos_interp`
  - `infinity_qwen3vl_rope2d_bf16`
  - `infinity_qwen3vl_gelu_tanh_bf16`
  - `infinity_qwen3vl_silu_bf16`
- Total: **96**

`make_packet.py` verifies the actual `build.sh` + `encoder_activation_exports.txt`
set equals `expected-new96.txt`, that `expected-old90.txt` is a strict subset,
and that canonical26 FA4 is untouched.

## Fast Root probe

```sh
CARGO_OXIDE=<cargo-oxide> \
  libs/infinity-v3/backends/oxide/device/build.sh /tmp/qwen-vl-sm90 \
  --architectures sm_90 --only-symbol-probes
```

This is a development probe, not an all-hardware qualification.

## Full all3 + old90 preservation

```sh
OLD90_BUNDLE=<published-90-export-kernels> \
CARGO_OXIDE=<cargo-oxide> \
  bash aot/build_all3.sh /tmp/qwen-vl-all3
```

`aot/preserve_old90.py` compares every published old export's `.text` and
driver resources on `sm_90`, `sm_100` and `sm_120`. Any byte/resource drift
fails. If the published bundle is absent, the step fails; it must not be
fabricated. `PACKET.json` intentionally records
`old90_preservation_executed=false`.

## Exact-FP16 follow-up

The cached production Qwen3-VL / Qwen3.5 checkpoint is BF16, so the `ExactFp16`
temporal split is not part of this gate. If a later F16 checkpoint requires it,
produce a separate minimal SM90 source packet with its own hashes and old90
preservation before claiming support.
