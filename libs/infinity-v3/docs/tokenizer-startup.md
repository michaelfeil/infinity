# Tokenizer and frontend startup

Infinity temporarily pins Basetenkenizer through Cargo's git dependency to
commit `8e95fd5e983c0bd0556331bd3352384af4dfb825`, with default features disabled.
That commit contains the constructor improvements in
[Basetenkenizer #16](https://github.com/basetenlabs/basetenkenizer/pull/16).
The pinned revision retains independent Fancy VM programs and the BPE prefix
automaton to preserve throughput; see the upstream
[validation](https://github.com/basetenlabs/basetenkenizer/blob/8e95fd5e983c0bd0556331bd3352384af4dfb825/docs/startup-throughput.md).
Cargo fetches the source; no copy is checked into Infinity. Replace the git pin
with a published crate version once a release contains these changes.

## Optional tokenizer initialization

Embedding and classification workers serve requests with Hugging Face while
their shared Basetenkenizer builds on a background thread. The first worker
starts after one seed clone; remaining worker copies are created progressively.
Once the accelerator is published through a shared `OnceLock`, each worker
retains its reference for eligible ID-only requests. Paired reranking inputs,
source-aligned outputs, metadata operations, and unsupported tokenizers keep
their Hugging Face behavior. Accelerator failure also retains HF serving.

Qwen embedding/conversation processors use the same background initialization.
Their configured HF tokenizer is shared immutably with the builder, while
per-request truncation still creates an owned tokenizer. WordPiece and other
non-BPE tokenizers start no accelerator thread.

Decision models skip the generic pooled-text accelerator because SystemOne
owns their tokenization. OneJev also uses an image-only Qwen processor, avoiding
a second text tokenizer that its image path never uses.

The generic `/tokenize` and `/decode` pool starts on its first metadata request
for Decision models, Mistral3 embeddings, and processor-owned Qwen embeddings.
Raw-text Qwen retrieval variants, EmbeddingGemma2, generic embeddings, and
classifiers retain eager generic workers because inference needs them. Worker
limits and bounded queue capacity still apply.

Shutdown closes queues and joins workers and builders, including initialization
already in progress. Shutdown before a lazy pool's first use creates no workers.
An in-progress blocking constructor can delay final resource destruction.

## Frontend and backend overlap

Tokenizer loading starts on a blocking task alongside device discovery. Backend
checkpoint loading remains gated on startup configuration validation. Readiness
waits for frontend completion, backend health, and the full configured warmup.
Failed startup drains its peer and shuts down any successfully constructed side.

Required tokenizer files, queue limits, and applicable multimodal numeric limits
and host patterns are checked before checkpoint loading. Corrupt tokenizer or
processor content can fail during concurrent loading; cleanup waits for the
backend before returning the error.

The redundant LibTorch arithmetic preflight is removed. Device syntax/dtype and
checkpoint validation retain their ordering; CUDA availability is checked before
native weight allocation. CUDA initialization still occurs when the device is
first used. A separate B200 probe benchmark found no material net saving from
removing the arithmetic operation alone: median initialization was 659.7 ms
with the probe and 651.7 ms without it, with most work moving to the first copy.

Optional model-artifact probes run concurrently while preserving configuration
priority and required companion-file errors. Lower-priority probes are cancelled
once the selected configuration is known.

## Measurements and validation

The production trace recorded frontend completion at 2.351 seconds and backend
startup immediately afterward. The 1.665-second gap after the token-limit log
preceded backend loading; OneJev's unused Qwen text tokenizer was constructed
synchronously there. All artifacts were already local.

An earlier five-pair Joey-1 campaign on a 256-CPU host with one B200 measured
frontend construction at a median 1.603 seconds on main versus 0.415 seconds
with the startup changes. Process-to-health timing was noisy; the paired median
improvement was 1.098 seconds and four of five runs improved. The two binaries
used the same existing native library and full warmup. This historical build
also contained the upstream constructor candidate, although OneJev skipped those
accelerators; this is not a freshly built Docker comparison of this revision.

Three-process metadata-pool measurements with sixteen workers and a local Joey
tokenizer found median idle resident memory of 981 MiB for the eager pool and
122 MiB for the lazy pool. Thread counts were 18 and 1. The first metadata
request paid the deferred cost: 41.6 ms versus 0.127 ms. These measurements do
not depend on the upstream constructor candidate.

Earlier offline release smoke runs against published version 0.2.9 passed single
and paired HF token-ID parity and joined shutdown for embedding and
classification. The example's first request took 27.9 ms and 28.8 ms respectively;
these are individual CPU-only samples after pool construction, excluding HF JSON
loading, weights, HTTP readiness, and inference.

Deterministic tests hold the builder behind a gate, serve embedding,
classification, paired reranking, prompt/truncation, and metadata requests, then
compare outputs after publication. Additional tests cover concurrent first use,
shutdown, unsupported tokenizers, Qwen text/image behavior, artifact-probe
priority, startup failure cleanup, and validation ordering.

## Reproduce locally

Both examples use a local tokenizer JSON and download no artifacts or weights:

```sh
cd libs/infinity-v3
cargo build --locked --release -p infinity_v3-core --no-default-features \
  --example tokenizer_startup --example metadata_tokenizer_startup
target/release/examples/tokenizer_startup /path/to/tokenizer.json embedding steady
target/release/examples/tokenizer_startup /path/to/tokenizer.json classification steady
target/release/examples/metadata_tokenizer_startup /path/to/tokenizer.json eager
target/release/examples/metadata_tokenizer_startup /path/to/tokenizer.json lazy
```

The worker example checks short multilingual single/paired token IDs, reports
first-request and optional steady sequential timing, and joins shutdown. The
Linux metadata example compares idle resources and first-use cost. Startup logs
also report elapsed time for HF loading and background accelerator construction.
