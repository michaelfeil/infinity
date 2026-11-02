# Mistral3 v5 source-closure review (read-only)

Scope: `/tmp/oxide-all-model-feature-matrix-v1/mistral3-v5` only, plus prior
review `/tmp/oxide-mistral3-serving-independent-review-v1/REVIEW.md` /
`SOURCE-HASHES.sha256`. No product/source/GPU/Torch/Docker/network/CUDA-device
build or optimized build was performed. CPU-only focused validation used the
existing debug test binary; no all-tests run.

Preserved prior outputs: v4 and review-v1 files were not modified. New outputs:
this file and `SOURCE-HASHES.sha256` in
`/tmp/oxide-mistral3-serving-independent-review-v2/`.

## v1 findings now closed at source level

H1 context bind is present in `Mistral3Engine::enqueue` (`mistral3_exec.rs:1221-
1223`) and in qualification `run` (`:1479`). H2 params-only detection runs
before config.json-only probes (`lib.rs:4369-4378`, `:5076-5081`). H3-H5
vocab/selector/merge/image-grid checks are now pre-upload
(`lib.rs:6371-6517`; `multimodal.rs:26-43`). H6 selection capacity is
`max_tokens + max_sequences` (`mistral3_exec.rs:1109-1121`) and is checked before
upload (`:1244-1256`). H7 uses real `Inference::Deferred` with staged device work
(`lib.rs:4896-4968`, `:4970-5068`). H9 CLS is admitted (`lib.rs:4810-4814`).
H10 score labels come from the score tensor shape (`mistral3.rs:979-991`,
`mistral3_exec.rs:1134-1168`, `lib.rs:4995-5068`). M1 result-memory reservation,
M2 image-grid bound, M3 marker-block bound, M4 GELU execution, M5 failed-fence
poisoning, and L1 empty fast path are present at source level.

`Engine::complete_work` now marks the engine failed and closes the I/O pool on
an unhealthy completion (`lib.rs:2679-2687`); `Mistral3Pending` waits/packs in
the Deferred closure, poisons the lease on errors, and releases the engine work
(`lib.rs:4601-4755`).

## Remaining P1 findings

### P1-1. The new Mistral3 model-dtype mean leaf is algebraically, not ordinally, ATen mean

Native Mistral3 `model_dtype_mean_pool()` returns true
(`libs/infinity-v3/backends/libtorch/cpp/models/mistral3_models.cpp:401-403`);
the Native bridge therefore bypasses the Candle packed mean
(`.../bridge.cpp:1478-1491`) and calls `Mistral3::mean_pool` at `:1504`. Native
mean is `(value.sum(0, true) / value.size(0)).to(at::kFloat)` (`:405-408`).

The new Oxide leaf is `device/src/mistral3_model_mean.rs:1-52`. It assigns one
thread per feature and accumulates the sequence rows **sequentially** in F32
(`:29-35`), rounds the sum once to BF16, divides in BF16, then widens to F32
(`:36-40`). The routed leaf is selected for every Mistral3 `Pool::Mean` request
(`decoder.rs:888-899`, `:2141-2205`).

`at::sum` on CUDA BF16 is an ATen reduction: its accumulation order depends on
the ATen reduce configuration (parallel/block/warp/shared-memory tree), not a
single-thread row-order F32 sum. The packet contains no ATen reduction-order
reference, and `vendor/candle_kernels/pooling/mean_pool.cu` /
`kernels/packed_pool.cu` are the Candle tree, not ATen. The leaf's comment
"ATen BF16 sum rounds once" establishes the rounding boundary only, not the
required reduction order. Therefore Mistral3 mean embeddings and classifier
scores cannot be treated as Native-exact from this source. This is a P1
readiness blocker for any consumer that compares mean/score vectors; Root needs
either an exact ATen-order implementation or a Native-gold qualification of this
leaf before claiming parity.

### P1-2. The optimized `f065` consumer is an old artifact, not a v5 build

v5 metadata repeats the old optimized artifact claim
(`HANDOFF.json:384-397`, `BUILD.json:366-378`): binary
`.work/host-consumer-opt3-v4/release/mistral3-host-consumer`, sha256
`f0653ddec06f0667f3c36dedf65b8026f85154c3629e224af4d766d4340c3067`.

The actual binary timestamp is `2026-11-02 07:28:18`, while the v5 source files
are timestamped `mistral3_exec.rs` 08:09:11, `mistral3.rs` 08:10:42, `lib.rs`
08:10:42, and the new device mean leaf 07:57:35. The debug consumer binary is
also older (07:00:22). `native/host-consumer/src/main.rs` is byte-identical to
v4 (`cbe0968…`) and does not exercise CLS, dynamic C-label output, two
in-flight Deferred requests, or failed-fence/quarantine behavior.

This artifact may be preserved as old host-opt evidence, but it is **not** v5
source evidence. A new optimized unified normal-factory/owning-life consumer
must be built from the v5 source before any readiness claim.

## Remaining P2 findings

### P2-1. Raw duplicate selectors are rejected although Native and the ordinary Oxide path accept them

`validate_mistral3` rejects repeated raw selectors
(`lib.rs:6416-6423`). The Native bridge iterates every `raw_count` entry and
emits each requested span, with no duplicate rejection
(`libs/infinity-v3/backends/libtorch/cpp/bridge.cpp:1513-1521`). The ordinary
Oxide output layout explicitly preserves raw duplicates/last-wins precedence
(`output.rs:320-350` tests; `lib.rs:8455`).

The v5 fixed arena was sized as `max_tokens + max_sequences`; rejecting raw
duplicates avoids the capacity problem by narrowing Native-valid request
semantics. Either support duplicates within an explicit capacity or document
that Mistral3 differs from Native/the ordinary path.

### P2-2. The empty-selector fast path bypasses image marker validation

`submit_embed_mistral3` returns `Inference::Ready(Embeddings::default())`
immediately after `validate_mistral3` when both selector lists are empty
(`lib.rs:4900-4903`). `validate_mistral3` checks grid/merge/pixel/marker-block
geometry but not the actual placeholder ids (`lib.rs:6418-6505`). Marker-id
validation is only in `Mistral3Prepared::enqueue_vision`
(`mistral3_exec.rs:381-382`), which this fast path never reaches.

Native `Mistral3::forward` validates image placeholders before returning states
(`mistral3_models.cpp:421-440`). Thus an empty-output Mistral3 request with
malformed image placeholders is accepted as empty by Oxide while Native
rejects it. Validate markers/media before the empty short-circuit, or define
and test that empty requests intentionally skip media execution.

### P2-3. Mandatory Mistral3 mean/GELU exports are still optional at cold load

`DecoderKernels::load` resolves `infinity_mistral3_model_dtype_mean_bf16` with
`.ok()` (`decoder.rs:433-439`), and `Kernels::load` resolves
`infinity_mistral3_gelu_mul_bf16` with `.ok()` (`mistral3_exec.rs:47-54`).

The actual Mistral3 wrapper will use the model-dtype mean for every Mean request
(`decoder.rs:896`), and a GELU vision config will use `gelu_mul`. If the AOT
bundle lacks either export, cold open succeeds and the first affected request
fails only after the decoder/vision body has been enqueued
(`decoder.rs:2192-2199`; `mistral3_exec.rs:906-910`), then poisons the lease and
engine. Admission should require these exports at cold load when the parsed
config selects the corresponding path, rather than deferring a fatal error to
request execution. This is not a wrong-output path; it is an avoidable runtime
availability/readiness gap.

### P2-4. Vision activation admission is broader than Native Mistral3

v5 parses and executes `vision_config.hidden_act = gelu_pytorch_tanh` as tanh
GELU (`mistral3.rs:316-322`, `mistral3_exec.rs:882-892`). Native Mistral3
`ready` admits only `silu` or `gelu` for the vision tower
(`mistral3_models.cpp:167`) and maps the non-SiLU case to exact-erf activation
(`:347-352`). The GELU/SiLU leaf itself matches Native's rounded gated
activation for Native-valid modes, but the extra tanh-vision mode is a
Native-invalid admission, not a Native parity extension.

## Focused CPU validation performed

Existing debug test binary `target/debug/deps/infinity_backend_oxide-7bef44295a29f9d8`
(sha256 `59febe5e0b3b1c8bb71187a793ef8f77013990b18630f234d05ad0c3a88be133`,
timestamp 08:10:50) was run with `CUDA_VISIBLE_DEVICES=` and CPU affinity
144-151, filters only:

- `mistral3::tests::validation_rejects_vocab_and_selector_oob_before_media` and
  `...validation_rejects_native_invalid_image_grid_and_short_marker_block`:
  2 passed.
- `decoder::tests::mistral3_model_dtype_mean_route_is_count_independent_and_mean_only`:
  1 passed (route selection only; no device math).
- `mistral3::tests::params_only_probe_admits_official_short_key_without_config_json`:
  1 passed.

No GPU, device-leaf numeric comparison, Torch, Docker, network, optimized build,
or all-tests run was performed.

## Conditional readiness for one new optimized unified consumer

Not unconditionally ready. Source-level normal-factory/Deferred/owning-life
closure is substantially present, but a single optimized consumer should be
authorized only conditionally on:

1. The Root device union containing both
   `infinity_mistral3_model_dtype_mean_bf16` and
   `infinity_mistral3_gelu_mul_bf16` (v5 requests 131 exports vs current 130);
2. resolving P1-1 with an actual ATen-order mean or a Native-gold numeric
   qualification, because the sequential leaf is not exact by source;
3. rebuilding the unified normal-factory/owning-life consumer from v5 source;
   `f065` and the pre-existing debug binary are preserved old artifacts, not v5
   evidence; and
4. including at least CLS, dynamic-C score layout, two in-flight Deferred
   requests, early-drop lease lifetime, and the new mean/GELU paths in that
   consumer, then running it under the Root GPU guard.

No all-model parity claim is made.

## Source hashes

See `SOURCE-HASHES.sha256` in this directory. Reviewed v5 hashes match
`SOURCE-PINS.txt` for the listed produced sources; the old `f065` binary hash is
recorded separately and is explicitly not a v5 source artifact.
