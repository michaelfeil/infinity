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

CUDA inference requires SM80 (Ampere) or newer in both backends. SM75/Turing
is unsupported; use a CPU build on hosts without a supported GPU.

## Request service tiers

`--priority-queue-bonus-ms` sets the virtual queue age bonus for requests carrying
`"service_tier": "priority"`. It defaults to `1000` ms; `0` disables priority scheduling.
Only inputs waiting for model batching are reordered, without adding a delay.

```sh
infinity_v3 --model-id /path/to/model --priority-queue-bonus-ms 250
```

`INFINITY_PRIORITY_QUEUE_BONUS_MS` (or `PRIORITY_QUEUE_BONUS_MS`) is the environment
alternative. One value applies to all models; semicolon-separated values align
with `--model-id`. Negative and fractional millisecond values are rejected.

`--flex-queue-penalty-ms` / `INFINITY_FLEX_QUEUE_PENALTY_MS` adds a virtual queue
penalty for `"service_tier": "flex"` (default `10000` ms). Flex yields to recent
work but ages, so later arrivals cannot starve it indefinitely. `0` disables flex
scheduling; set both adjustments to `0` for FIFO. Values broadcast or align with
`--model-id`, just like the priority bonus. Billing is unchanged for flex.

## Sequence length and batch capacity

Use `--max-sequence-length` to cap tokens per sequence independently of the
batch's total token budget:

```sh
infinity_v3 --model-id /path/to/model \
  --max-sequence-length 8192 --max-batch-tokens 32768
```

The cap includes formatted prompts and image tokens, and cannot exceed the
model's supported input length or the batch token budget. When omitted, existing
model and batch limits apply. Inputs follow the endpoint's existing truncation
policy; OneJev rejects oversized prompts. `/v1/systemone` request `max_len` can
lower its limit further.
Set `INFINITY_MAX_SEQUENCE_LENGTH` (or `MAX_SEQUENCE_LENGTH`) instead of the flag.
Like other model settings, one value applies to all models; semicolon-separated
values align with `--model-id`, and `none` leaves a model's limit unchanged.

## UltraBERT embeddings

`dmis-lab/UltraBERT-Embed-150M` supports native LibTorch and Candle and produces
768-dimensional CLS embeddings with a 32,768-token context. For the native
LibTorch GPU image:

```sh
infinity_v3 --model-id dmis-lab/UltraBERT-Embed-150M \
  --torch-device cuda --dtype bfloat16 --max-batch-tokens 32768
```

Use `prompt_name: "query"` for retrieval queries; documents use the empty
`document` prompt. Mean and last-token pooling overrides and raw token
embeddings are supported. Native LibTorch uses FP32 on CPU and FP16/BF16 on
CUDA SM80+. See [native model coverage](libtorch-model-coverage.md) for backend
limitations and the [model catalog](supported-models.md) for other checkpoints.

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
