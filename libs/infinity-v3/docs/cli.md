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

The backend is normally determined when the binary or container image is built,
so most deployments should omit `--backend`. A single-backend build selects its
compiled backend automatically. Builds containing both prefer LibTorch; use
`--backend candle` (or `INFINITY_BACKEND=candle`) as an advanced override when
Candle is needed. The flag cannot enable a backend missing from the
image; startup errors list the compiled choices.

Open `/docs` on the running server for interactive API documentation. The API
schema is available at `/api-doc/openapi.json`.

## UltraBERT embeddings

`dmis-lab/UltraBERT-Embed-150M` uses the Candle backend and produces
768-dimensional CLS embeddings with a 32,768-token context. The override below
selects Candle in dual-backend builds; Candle-only builds can omit it:

```sh
infinity_v3 --backend candle --model-id dmis-lab/UltraBERT-Embed-150M \
  --max-batch-tokens 32768
```

Use `prompt_name: "query"` for retrieval queries; documents use the empty
`document` prompt. Mean and last-token pooling overrides and raw token
embeddings are supported. CPU supports FP32/FP16; CUDA SM80+ supports FP16/BF16.
Its attention geometry uses FA2 even with `ATTN_BACKEND=fa4`. Native LibTorch
and Metal are unsupported for this checkpoint.

UltraBERT sequence classification checkpoints with a saved trained classifier
head use `/predict`; trained single-label rerankers also use `/rerank`.
`UltrabertForTokenClassification` checkpoints use `/predict_tokens`. These
heads share the embedding encoder's CPU and CUDA support. Base masked-LM
weights require a fine-tuned classifier head before serving classification.

## Structured decision inputs

`/v1/systemone` accepts text, JSON objects, or arrays in `state`. Laya question
`instructions` accepts the same forms. Ordinary structured content is rendered
as JSON before tokenization. Objects containing `messages` use the native
conversation format and retain its validation; Laya rejects native messages.
D3 retains its separate JSON-state and media contract.

`model` is optional with one loaded model and required with multiple models.
Model routing and backend response extensions retain their existing behavior.
