# Optional checkpoint comparison tools

These tools compare native LibTorch and Candle using real local checkpoints.
They are qualification utilities, not part of the inference library or router.
Cargo builds them only when explicitly requested with `benchmark-cuda`.
The existing example names and command lines remain available:

```sh
cargo run --release -p infinity-backend-libtorch \
  --features benchmark-cuda --example compare_bert -- MODEL 200 2 3 mean
```

| Tool | Purpose |
| --- | --- |
| `compare_bert` | Packed encoder checkpoints with explicit pool selection and encoder position IDs |
| `compare_decoders` | Decoder/encoder comparisons, dtype selection, optional long-context and FP8 workloads |
| `compare_gemma` | Gemma text and image/audio embedding comparisons |
| `compare_decisions` | Typed decision heads and field/action accuracy |
| `compare_heads` | Scalar classifiers and SPLADE vocabulary outputs |
| `compare_radix` | Prefix reuse, changed inputs and folded/ragged execution |

Each prints JSON measurements. Reports under `docs/benchmarks/` are kept locally
and ignored by Git; publishing them is deferred to a follow-up PR. An individual
run is not qualification by itself. These tools require checkpoint files and CUDA and
are not run by the normal unit tests. `benchmark-fp8` additionally enables the
Candle FP8 reference where supported.
