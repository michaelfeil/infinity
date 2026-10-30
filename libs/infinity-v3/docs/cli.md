# Running Infinity v3

From the repository root:

```sh
cd libs/infinity-v3
cargo build --locked
cargo run --locked --bin infinity_v3 -- --help
cargo run --locked --bin infinity_v3 -- --model-id /path/to/local/model
```

The project pins its Rust toolchain. Run commands from `libs/infinity-v3` so
Cargo loads the project build configuration. The default build uses LibTorch
and requires its SDK through `LIBTORCH`; use a CPU or CUDA distribution for the
desired accelerator. See the [contributor guide](../CONTRIBUTING.md) and
[LibTorch build instructions](libtorch-backend.md) for those configurations.

`--help` lists the flags and `INFINITY_*` environment variables supported by the
current binary. This avoids maintaining a stale copy of generated CLI help.
Use a local model directory or a compatible Hugging Face model ID. Model
availability, access credentials, pooling and supported inputs depend on the
checkpoint and selected backend.

LibTorch is the only inference backend. Most deployments should omit
`--backend`; `--backend libtorch` remains accepted for existing configurations.

Open `/docs` on the running server for interactive API documentation. The API
schema is available at `/api-doc/openapi.json`.

CUDA inference requires SM80 (Ampere) or newer. SM75/Turing
is unsupported; use a CPU build on hosts without a supported GPU.

## Request service tiers

`--priority-queue-bonus-ms` sets the virtual queue age bonus for requests carrying
`"service_tier": "priority"`. It defaults to `1000` ms; `0` disables priority scheduling.
Only inputs waiting for model batching are reordered, without adding a delay.

```sh
infinity_v3 --model-id /path/to/model \
  --priority-queue-bonus-ms 1000 --flex-queue-offset-ms 10000
```

`INFINITY_PRIORITY_QUEUE_BONUS_MS` (or `PRIORITY_QUEUE_BONUS_MS`) is the environment
alternative. One value applies to all models; semicolon-separated values align
with `--model-id`. Negative and fractional millisecond values are rejected.

`--flex-queue-offset-ms` / `INFINITY_FLEX_QUEUE_OFFSET_MS` adds a virtual enqueue-time
offset for `"service_tier": "flex"` (default `10000` ms). Flex yields to recent
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

### OneJev block-attention fanout

Block-attention questions from one request execute in one batch, preserving one
prefix prefill per request to match input-token billing. Requests exceeding the
compute, expanded-token, or request-count limit return HTTP 422 before inference.
They are never split into repeated prefills. Other protocols retain their batching.

Several complete requests can share a batch when their combined compute,
expanded-token and question counts fit. Each request is charged independently;
additional prefix sharing across requests does not enlarge the admission budget.

`--max-batch-expanded-tokens` defaults to `--max-batch-tokens`. Raising it lets
`--max-batch-tokens` count matching prefixes once plus every question suffix,
while bounding the sum of full sequence lengths separately:

```sh
infinity_v3 --model-id /path/to/elephant --decision-protocol onejev \
  --max-batch-tokens 32768 --max-batch-expanded-tokens 700000
```

A 13,000-token prefix plus fifty 150-token suffixes costs 20,500 compute tokens
and 657,500 expanded tokens. Template and image tokens also count. Prefixes must
match IDs, types, positions and prepared image identity; independent image
allocations are conservatively charged separately. Existing deployments may
need a larger expanded limit for requests that previously spanned batches.
Measure peak GPU memory before raising it: backend K/V storage still expands
across questions, and startup warmup does not validate this fanout shape.

For roughly 13K prefix tokens and 256 suffixes of 185 tokens, the budgets grow
to about 60.4K compute tokens and 3.39M expanded tokens. With paged attention
enabled (`INFINITY_TORCH_BLOCK_PAGED=1`), a validation configuration is
`--max-batch-tokens 65536 --max-batch-expanded-tokens 3500000`, with the
per-sequence limit kept at `--max-sequence-length 16000`. Set
`--max-decision-questions` and any `--max-batch-requests` to at least 256.
OneJev's separate 512-option limit caps all-yes/no requests at 256 questions;
questions with more answer options reach that limit earlier.

Suffix attention assembles K/V in whole-question chunks targeting 500,000
expanded tokens, reusing the prefix without another prefill. Set
`INFINITY_TORCH_BLOCK_KV_CHUNK_TOKENS` to a positive integer to tune this target;
a single question may exceed it. This bounds temporary K/V copies, not weights,
hidden states, or total GPU memory.

Last-token decisions select their final hidden rows directly from shared
execution storage, including during CUDA graph replay. They do not reconstruct
the prefix's hidden states for every question. Full-token outputs still require
the expanded layout.

`INFINITY_TORCH_BLOCK_PAGED=1` opts into shared prefix pages for CUDA suffix
attention. Complete prefix pages are stored once; each question owns only its
suffix, the incomplete prefix tail, and page padding. The attention operator
reads those pages with one softmax over prefix plus suffix. Small, unshared or
unsupported layouts retain the chunked path. This is disabled by default while
hardware and workload coverage is expanded.

For one shared prefix of length P and suffix lengths S, row-local work scales
with P + sum(S), but full attention still evaluates P² + sum(S × (P + S)) pairs.
Unique-token counts therefore do not measure all attention work or GPU memory.
For the verified Qwen tokenizer used by Elephant and Mammoth, multi-question
preparation tokenizes the first full prompt and reuses its prefix for subsequent
questions with identical rendered prefix text. Each template is still rendered
in full; other tokenizer configurations retain full-prompt tokenization.

`INFINITY_MAX_BATCH_EXPANDED_TOKENS` is the environment alternative. Values
broadcast or align with models; `none` uses the compute limit. An explicit limit
must be at least `--max-batch-tokens` and fit signed 32-bit offsets.
`infinity_batch_next_tokens` reports expanded tokens; the new
`infinity_batch_next_compute_tokens` reports the conservative compute charge.

## UltraBERT embeddings

`dmis-lab/UltraBERT-Embed-150M` supports native LibTorch and produces
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

Native message content also accepts OpenAI-style aliases:

- `{"type":"input_text","text":"Inspect this photo."}` normalizes to `text`.
- `{"type":"input_image","image_url":"data:image/png;base64,..."}` normalizes
  to `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}`.

An optional `detail` on `input_image` is preserved inside `image_url` and checked
by the model adapter (OneJev accepts only omitted detail or `auto`). URL loading,
image limits and supported roles remain model-specific. Unsupported fields such
as `file_id` are rejected. These aliases apply to the shared message content
parser, including `state.messages[].content`; they do not add a top-level `input`
alias, image interpretation of plain `state` arrays, or OpenAI Decisions question
and response formats. Existing content parts retain their canonical serialization.
