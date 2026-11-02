# Central device-owner request for Mistral3 v7

Packet: `/tmp/oxide-all-model-feature-matrix-v1/mistral3-v7`
Owner: `/root/parity_catalog` (current union is held at 130; this request is
for the next additive union, likely 131).

No device/AOT compile, GPU, Torch, Docker, or network execution was performed
by this packet. The following source exports are added under
`libs/infinity-v3/backends/oxide/device` and are listed in
`native/root_device_abi.json`:

1. `infinity_mistral3_model_dtype_mean_bf16` (BLOCKED from compile until
   `ATEN-MEAN-PLAN.md` Root controls close)
   - source: `device/src/mistral3_model_mean.rs`
   - args: `u16* input, u32* cu_seqlens, u32* selected, f32* out, u32 count, u32 width`
   - grid: `(count, ceil(width/256), 1)`, block: `(256,1,1)`
   - math: candidate masked sum in F32 from BF16 values, one BF16 round,
     BF16 divide, widen to F32. ATen 5c488 reduction order is unqualified for
     warp-split sequence lengths; do not claim bit-exact yet.
2. `infinity_mistral3_gelu_mul_bf16`
   - source: `device/src/mistral3.rs`
   - args: `u16* gate, u16* up, u16* out, u32 elements, u32 mode`
   - grid: `blocks(elements)`, block: `256`
   - mode: `0` exact GELU, `1` tanh GELU, `2` SiLU; activation is rounded to
     BF16 before multiplying `up`, matching the existing Native SiLU path.

Required next steps by the central owner: add both symbols to the next AOT
bundle, publish its symbol/manifest hashes, and run the existing device tests
(including zero/overflow mean counts, zero-merge image rejection, and
GELU/SiLU gated activation). Until that bundle exists, the Rust loader keeps
the GELU kernel optional for SiLU checkpoints and fails GELU activation closed
with a typed error; Mistral3 mean fails closed if its export is absent.
