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
