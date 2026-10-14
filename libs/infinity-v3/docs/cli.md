# Running Infinity v3

From the repository root:

```sh
cd libs/infinity-v3
cargo build --locked
cargo run --locked --bin infinity_v3 -- --help
cargo run --locked --bin infinity_v3 -- --model-id /path/to/local/model
```

The project pins its Rust toolchain. Run commands from `libs/infinity-v3` so
Cargo loads the project build configuration. The default build uses the Candle
CPU backend; CUDA and native LibTorch require their additional build features
and SDKs. See the [contributor guide](../CONTRIBUTING.md) and
[LibTorch build instructions](libtorch-backend.md) for those configurations.

`--help` lists the flags and `INFINITY_*` environment variables supported by the
current binary. This avoids maintaining a stale copy of generated CLI help.
Use a local model directory or a compatible Hugging Face model ID. Model
availability, access credentials, pooling and supported inputs depend on the
checkpoint and selected backend.

Open `/docs` on the running server for interactive API documentation. The API
schema is available at `/api-doc/openapi.json`.

## UltraBERT embeddings

`dmis-lab/UltraBERT-Embed-150M` uses the Candle backend and produces
768-dimensional CLS embeddings with a 32,768-token context:

```sh
infinity_v3 --backend candle --model-id dmis-lab/UltraBERT-Embed-150M \
  --max-batch-tokens 32768
```

Use `prompt_name: "query"` for retrieval queries; documents use the empty
`document` prompt. Mean and last-token pooling overrides and raw token
embeddings are supported. CPU supports FP32/FP16; CUDA SM80+ supports FP16/BF16.
Its attention geometry uses FA2 even with `ATTN_BACKEND=fa4`. Native LibTorch,
Metal, and classification heads are unsupported for this checkpoint.
