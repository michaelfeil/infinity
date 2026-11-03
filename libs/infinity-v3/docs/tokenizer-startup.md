# Tokenizer startup on many-core hosts

Basetenkenizer 0.2.9 compiled both a fancy-regex expression and a PCRE2 JIT
expression once per visible CPU for each `Split` pre-tokenizer. Its fused
Split/ByteLevel path also cloned the Split, recompiling the PCRE2 expressions.
For Joey-1 on 256 CPUs, one Split therefore compiled 256 fancy-regex expressions
and 512 PCRE2 expressions before serving a request.

The vendored patch compiles each expression once and clones handles. PCRE2
clones share immutable compiled code and own separate matching scratch pools.
The number of parallel handles, chunking, and runtime thread counts are retained.
This applies to Joey-1's marks-aware regex, which is not recognized by the
existing hand-written scanners. See [patch provenance](../vendor/basetenkenizer/PATCHES.md).

Decision models also skip the generic pooled-text Basetenkenizer because
SystemOne owns their tokenization. Image processing and HF `/tokenize` and
`/decode` workers remain available. Mistral3 embeddings and Qwen embeddings that
always use their model processor also defer the generic pool until metadata
requests need it. Other embedding and classifier acceleration initializes in
the background; their first worker can serve with Hugging Face before
Basetenkenizer is ready.

For OneJev specifically, `prepare_images` never uses the Qwen processor's text
tokenizer: OneJev formats and tokenizes its own prompts. OneJev now uses the
image-only constructor, which skips that second, also unused Basetenkenizer.
The shared loader retains vision-token, processor, and budget validation.
Ordinary Qwen embedding/conversation processors initialize acceleration in the
background and use Hugging Face while it is pending.

## Local OneJev frontend

The reported production trace uses a local model directory and records:

| Event | Seconds from frontend start |
| --- | ---: |
| Startup configuration validated | 0.366 |
| Maximum input tokens logged | 0.686 |
| Basetenkenizer constructed; frontend complete | 2.351 |
| Backend initialization starts | 2.352 |

The 1.665-second interval after the token-limit log precedes backend loading.
There are no Hub download probes on this local-directory path. OneJev's unused
Qwen text tokenizer was built synchronously in that interval.

A CPU-only release benchmark isolates this OneJev frontend with the same pinned
Joey tokenizer/config/template/processor files used below. Five fresh processes
per variant, alternating order and reversing it each round, produced:

| Phase | Eager Qwen text tokenizer | Image-only processor |
| --- | ---: | ---: |
| HF tokenizer load | 359.1 ms | 350.2 ms |
| OneJev label/template setup | 1.34 ms | 1.35 ms |
| Image processor setup | 590.0 ms | 34.6 ms |
| Total frontend setup | **955.4 ms** | **385.6 ms** |

This comparison uses the workspace's optimized release test build, 256 visible
CPUs, and a 32,768-token limit. Both variants include the other tokenizer startup
improvements: it isolates the image-only change rather than recreating the
original deployed binary. It excludes startup configuration/device validation,
metadata worker creation, backend initialization, and weights. It is not a
measurement of production container readiness. Images, token counts, patches,
and validation failures match in the synthetic regression test.

Run the ignored benchmark against local artifacts, without weights:

```sh
INFINITY_FRONTEND_PROFILE_MODEL=/path/to/local/model \
cargo test --locked --release -p infinity_v3-router --no-default-features \
  --features http --lib profile_frontend_startup -- --ignored --nocapture
```

The table above was captured before general background acceleration was added.
`INFINITY_FRONTEND_PROFILE_LEGACY=1` now selects the ordinary Qwen constructor
with background text acceleration, so it no longer recreates the historical
eager-construction measurement.

## Concurrent loading and optional workers

Required tokenizer parsing and processor construction now run on a blocking
worker alongside global CUDA discovery, backend loading, and configured warmup.
Per-model device discovery runs on a separate blocking worker after global
configuration validation, so it cannot serialize frontend setup on a CUDA
driver call. Checkpoint loading is gated on startup configuration validation.
HTTP readiness waits for both sides.
A failed peer is fully drained and any successful frontend/backend is shut down.
Cancellation cannot interrupt an already-running native or blocking initializer;
its owned result is dropped when that work completes.

Missing tokenizer files, invalid queue limits, and applicable multimodal numeric
limits/host patterns are checked before checkpoint loading. Corrupt tokenizer or
processor content can still fail during concurrent loading; the backend is then
drained and shut down before startup returns the error.

Decision models now create their auxiliary `/tokenize` and `/decode` worker pool
only on its first request. OneJev's required tokenizer remains ready before HTTP
starts. Initialization is shared across concurrent requests, and shutdown before
first use creates no workers. Eager embedding/classifier workers avoid the lazy
initialization mutex in the request path. Main already created workers
incrementally in a background supervisor; deferring that entire supervisor is
the new behavior for Decision models. Once started, the first worker still waits
for one seed clone when multiple workers are configured, then the supervisor
builds remaining copies in the background. A single unaccelerated worker does
not clone. An accelerated BPE worker also needs a temporary builder template,
even with a one-worker limit.

Three fresh optimized processes per variant, with a local Joey tokenizer and
sixteen metadata workers, measured the following after one idle second:

| Measurement | Eager metadata pool | Lazy metadata pool |
| --- | ---: | ---: |
| Process resident memory, median | 981 MiB | 122 MiB |
| Process threads | 18 | 1 |
| First metadata request, median | 0.127 ms | 41.6 ms |

The memory saving was about 859 MiB before the optional endpoints were used.
The first metadata request pays one seed-clone/thread-start cost; subsequent
workers are created in the background. Requested tokenization worker limits and
bounded queues still apply. A pure OneJev decision does not start this pool.

Reproduce this Linux-only resource measurement with:

```sh
cargo build --locked --release -p infinity_v3-core --no-default-features \
  --example metadata_tokenizer_startup
target/release/examples/metadata_tokenizer_startup /path/to/tokenizer.json eager
target/release/examples/metadata_tokenizer_startup /path/to/tokenizer.json lazy
```

The unconditional LibTorch arithmetic preflight has also been removed. Device
and index validation remain before weight access, and actual model probes and
full configured warmup remain readiness requirements. Five fresh-process pairs
on a B200 with Torch 2.14.1, an 8 MiB BF16 host-to-device copy and a small BF16
linear forward gave median total GPU initialization of 659.7 ms with the probe
and 651.7 ms without it. Most of its roughly 593 ms moved into the first copy
(587 ms without preflight); the noisy results do not establish a material net
saving from removing the arithmetic operation alone.

## Embedding and reranking startup

The generic worker pool now starts serving Hugging Face requests while a shared
Basetenkenizer builds on a separate thread. Remaining HF worker copies are also
created progressively. Workers check for a published accelerator without waiting
or taking an initialization mutex, then retain the reference so subsequent
eligible ID-only requests avoid shared initialization checks. Paired reranking
inputs, source-aligned outputs, metadata operations, and unsupported tokenizers
retain their HF behavior.

Qwen embedding/conversation processors use the same background initialization.
Their configured HF tokenizer is shared immutably with the builder, while
per-request truncation still creates an owned tokenizer. OneJev's image-only
processor starts no accelerator. WordPiece and other non-BPE tokenizers also
start no accelerator thread. Decision metadata pools remain lazy as described
above. Mistral3 embeddings and Qwen variants that do not accept raw text also
keep their unused generic pool lazy. Raw-text Qwen retrieval variants,
EmbeddingGemma2, generic embeddings, and classifiers/rerankers retain eager
generic workers because their inference paths need them.

Only optional acceleration is deferred. Required tokenizer files, model/media
validation, backend health, and configured model warmup remain startup work.
Initialization failure retains HF serving. Shutdown joins builders and workers;
an in-progress CPU build can delay final resource destruction, including an
early Qwen processor drop.

Deterministic tests hold the builder behind a gate, serve embedding, classifier,
paired reranking, prompt/truncation, and metadata requests, release the builder,
then compare outputs after publication. Separate tests cover pending Qwen text
processing, image parity, unsupported tokenizers, and shutdown while building.

CPU-only release measurements compared the state before this staging extension
with the final implementation, using the cached Joey tokenizer, 256 visible
CPUs, and 16 tokenization workers. Five fresh-process pairs per mode alternated
baseline/patched order. Timings begin after HF JSON parsing and pool construction;
they include the wait for worker 1 and the first encode, and exclude model
weights, HTTP, and inference.

| First tokenizer request | Before staging extension | Final staged startup |
| --- | ---: | ---: |
| Embedding | 588.0 ms | 43.7 ms |
| Single-input sequence classification | 592.5 ms | 46.5 ms |

Every patched first request completed before the Basetenkenizer construction
log. After a two-second settling interval, the accelerator was ready in every
run. Fifty subsequent sequential requests took roughly 0.07 ms each on both
variants. An additional burst of 8,192 concurrent requests per process exercised
the bounded 1,024-job queue. Across both modes, aggregate burst throughput was
152k requests/s before and 155k after, with mixed paired deltas; no throughput
speedup is claimed. All first, sequential, and concurrent token IDs matched HF.

Additional checks retained HF behavior for paired inputs and WordPiece. A single
Qwen processor pair using the same local model assets measured constructor time
of 552.2 ms before and 0.6 ms after; HF parsing was excluded, and text IDs matched
before and after accelerator publication. This is a constructor component check,
not an embedding-model or container-readiness benchmark.

## Cached Joey-1 server on a B200

Five fresh-process pairs compared pristine main commit `47ac07479d884a512297f938e44d379840e220a9`
with the complete patch, alternating main/patched and patched/main order.
These native-server measurements preceded the general embedding/reranking
staging extension; OneJev already skipped its unused accelerators in this build.
The host exposed 256 CPUs. Both binaries used Rust 1.93.1, the workspace release
profile (fat LTO, native CPU target), Torch 2.14.1+cu130, one B200, and the local
5,443,677,680-byte Joey checkpoint at revision
`9efbe485efbed9598dfca5294cbbe8da00e21908`. No artifacts were downloaded.

Both variants linked the same existing native library, SHA-256
`f5a7f190f4a58730604963c072ea22d068d992e02ce7a89388f4732bb38dfbe3`.
The main source came from `git archive`; all 433 tracked files other than the
documented link-only replacement for `backends/libtorch/build.rs` matched main's
Git blob hashes. This measures Rust/frontend changes against a fixed native
backend. It is not a freshly built Docker image comparison.

| Measurement | Main median | Patched median |
| --- | ---: | ---: |
| Frontend construction | 1.603 s | 0.415 s |
| Process launch to HTTP health 200 | 3.357 s | 1.579 s |
| First startup log to readiness log | 3.145 s | 1.308 s |
| Global configuration/device validation | 1.389 s | 0.531 s |

Frontend construction was stable: main ranged from 1.587–1.620 s, patched from
0.401–0.429 s. Required tokenizer setup overlapped CUDA discovery; in the patched
smoke run the frontend finished at +407 ms and global validation at +559 ms.
Backend health and full configured warmup remained readiness requirements.

End-to-end timing was noisy. Process-to-health ranged from 2.423–6.517 s on main
and 1.171–6.613 s patched; health polling had 100 ms resolution. The median paired
improvement was **1.098 s**, with four of five patched runs faster. Every sample,
including the patched run with 1.895 s validation and 4.196 s checkpoint loading,
is retained below. Differing CUDA discovery/loading delays affect the overall
medians, so their difference is not an isolated tokenizer speedup or a tail
latency guarantee.

| Pair | Main launch to health | Patched launch to health | Patched minus main |
| --- | ---: | ---: | ---: |
| 1 | 2.474 s | 1.376 s | -1.098 s |
| 2 | 5.360 s | 2.427 s | -2.934 s |
| 3 | 3.357 s | 6.613 s | +3.256 s |
| 4 | 6.517 s | 1.171 s | -5.347 s |
| 5 | 2.423 s | 1.579 s | -0.844 s |

A separate five-pair comparison against the intermediate version that already
skipped OneJev's unused tokenizers did not establish an additional readiness
gain: the patched paired median was 102 ms slower in that noisy campaign.
The additional lazy-worker change saved about 923 MiB of resident memory at the
post-decision sampling point in that comparison. Main's original metadata BPE
can still be initializing after readiness, so immediate post-decision memory
and thread counts must not be treated as steady-state measurements.

First-decision inference took about 6–7 ms longer with the complete patch.
Four additional fresh processes, ordered main/patched/patched/main, each served
20 identical decisions. Requests 2–20 reported **4 ms inference time on both
variants**, with 0 ms rounded tokenization and queue time. OneJev does not start
the optional metadata worker pool. The first-request difference did not persist;
its cause was not isolated. All 80 responses were HTTP 200 and byte-identical.
All startup comparisons also passed health, decision-response parity, and
graceful exit. Some immediate GPU-memory samples retained 1–4 MiB; final delayed
checks showed zero memory, no server processes, and released ports.

The common server arguments were:

```sh
CUDA_VISIBLE_DEVICES=0 infinity_v3 --model-id /path/to/local/joey-1 \
  --backend libtorch --torch-device cuda --dtype bfloat16 \
  --decision-protocol onejev --max-batch-tokens 32768 \
  --max-sequence-length 16000 --max-batch-expanded-tokens 700000 \
  --auto-truncate --hostname 127.0.0.1 --port 18080 \
  --prometheus-port 19000 --json-output
```

## Compile-once regex measurement

CPU-only component measurements with the public `baselabs/joey-1` tokenizer at
revision `9efbe485efbed9598dfca5294cbbe8da00e21908`, 248,044 vocabulary entries,
33 added tokens, and a 19,989,339-byte tokenizer JSON. Baseline: published
Basetenkenizer 0.2.9, as used by Infinity commit `47ac0747`. Patched: the same
dependency versions with the vendored regex construction changes.

Five fresh processes per variant on a host exposing 256 CPUs, alternating
baseline/patched order and reversing that order every round. Both benchmark
binaries used Rust 1.93.1, release optimization level 3 without LTO. Files were
already local and the storage cache was uncontrolled. These timings exclude
weights, downloads, LibTorch initialization, and HTTP readiness.

| Component | Baseline median | Patched median |
| --- | ---: | ---: |
| HF tokenizer load | 309.5 ms | 308.4 ms |
| Serialize configured HF tokenizer | 139.2 ms | 141.8 ms |
| Basetenkenizer construction | 1,055.8 ms | 810.0 ms |

Basetenkenizer construction improved by 23%. Individual construction samples:

| Round | Baseline | Patched |
| --- | ---: | ---: |
| 1 | 1,159.7 ms | 835.5 ms |
| 2 | 1,057.0 ms | 788.8 ms |
| 3 | 1,041.7 ms | 801.9 ms |
| 4 | 1,055.8 ms | 822.3 ms |
| 5 | 1,050.4 ms | 810.0 ms |

An initial three-process comparison pinned to eight CPUs gave 803.9 ms median
baseline construction; the 256-CPU baseline was 1,062.8 ms. The patch removes
this repeated-compilation penalty without restricting runtime CPU parallelism.
Skipping the unused generic tokenizer for Decision models removes a separate
construction, but its effect on readiness depends on overlap with backend work;
these component timings cannot establish total Docker startup improvement.

The subsequent BPE constructor patch removes an automaton used only for exact
whole-token lookup and an unused prefix table, shares vocabulary strings, and
reduces construction-only lookup/allocation work. The measured construction
median falls further to 387.7 ms; see the vendored [stage measurements and
regression coverage](../vendor/basetenkenizer/PATCHES.md#bpe-constructor).
Short cached-encode microbenchmarks cost approximately 20 ns more per input
(3–5%); long inputs were faster and unique inputs were within measurement noise.
OneJev's image-only path avoids this text tokenizer altogether.

## Reproduce locally

Use a local tokenizer JSON; the example downloads no artifacts or weights:

```sh
cd libs/infinity-v3
cargo build --locked --release -p infinity_v3-core --no-default-features \
  --example tokenizer_startup
taskset -c 0 target/release/examples/tokenizer_startup /path/to/tokenizer.json
taskset -c 0-255 target/release/examples/tokenizer_startup /path/to/tokenizer.json
```

Choose CPU IDs available to the container. The example reports each phase,
checks token IDs against Hugging Face for short and long multilingual inputs,
and measures sixteen HF clones separately. The workspace release profile uses
LTO, so its absolute timings can differ from the comparison above. Fresh-process
checks of this example passed with one and 256 CPUs visible.

Startup logs now include `elapsed_ms` for HF loading and Basetenkenizer
construction. Optional model artifact probes overlap, preserving the legacy
configuration priority and required companion-file errors. Lower-priority
legacy probes are cancelled once the selected configuration is known.

Native Unix safetensors validation already reads only headers and file sizes;
it does not hash or scan tensor payloads before loading. Header validation and
the native reader were left intact. The native server was exercised with the
cached Joey checkpoint on a B200, reusing a pinned native library. A fresh Docker
image build was not measured.
