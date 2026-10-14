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

`laya_performance.py` measures a native Laya library through the synchronous C
ABI using the pinned trained checkpoint. Its default matrix covers sequence
lengths 100, 200, 300, 400 and 500 with batch sizes 1, 2, 4, 8, 16 and 32, plus
ragged batches and the supplied original request fixtures. Controlled inputs
reuse real token IDs but are shape workloads, not semantic accuracy examples.
The tool requires Python, NumPy and the same Torch version used by the library.

```sh
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python laya_performance.py --library /path/to/libtei_torch.so \
  --model /path/to/laya --fixture /path/to/decision-inputs.json \
  --output /path/to/report --iterations 100 --repeats 2 --profile
```

Use `--graphs` for the native encoder graph mode. Compare baseline and candidate
on the same idle GPU and CPU affinity. Reports retain exact inputs, raw latency
samples, outputs, peak allocator memory, library/checkpoint identity and optional
operator traces. Timings include input upload, encoder, decision head and host
outputs; they exclude HTTP and tokenization. `--case length300-batch8` selects a
single matrix workload for focused profiling.
