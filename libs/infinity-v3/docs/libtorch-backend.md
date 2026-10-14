# Experimental native LibTorch backend

Candle is deprecated. Use `--backend libtorch` or `INFINITY_BACKEND=libtorch`
for models supported here, with a build that includes the `libtorch` feature.

LibTorch 2.14.1 runs alongside Candle as an optional inference runtime. Rust owns
model configuration, checkpoint loading, batch validation and infinity_v3 integration;
a small C ABI bridge runs the model with the PyTorch C++ tensor API. Python is
not required at build time or inference time when using a standalone LibTorch
distribution. There is no TorchScript export step.

## Packed inference

Token embeddings, projections, attention outputs and MLP activations use
`[total_tokens, hidden_size]`, with Q/K/V shaped
`[total_tokens, heads, head_dim]`. Sequence boundaries come directly from infinity_v3's
cumulative lengths. No token padding is allocated. Relative position bias paths use attention bias at each sequence's actual length.

On CUDA, attention calls `aten::_flash_attention_forward` with device-resident
INT32 cumulative lengths and maximum sequence length. This is the Flash
Attention operator behind
[PyTorch's `varlen_attn`](https://docs.pytorch.org/docs/stable/nn.attention.varlen.html).
The C++ call uses zero dropout and full bidirectional attention for BERT. Kernel
errors propagate to Rust; there is no padded fallback.

ModernBERT, Gemma3, Nomic MoE and validated Gemma2 paths additionally use
privately compiled forward kernels from
the Flash source pinned by Torch 2.14.1. The upstream headers remain unchanged;
compiling without Torch's `UNFUSE_FMA` definition preserves fused softmax
arithmetic needed for Candle parity. This path uses actual packed token counts,
strides and cumulative offsets, with no Python or Candle attention runtime.
The same arithmetic is selected for the validated Qwen3 MoE BF16 configuration,
alongside its exact residual normalization. The vendor directory records the
Torch and Flash revisions and source hashes; installation retains the complete
BSD license, authors and provenance under `share/licenses/tei_torch`.

CPU has an unpadded reference implementation: all projections/MLPs stay packed,
and attention uses SDPA on each sequence at its actual length. This is useful for
correctness testing; it is not a fused CPU varlen kernel. MPS and XPU are rejected
in this prototype because this Torch varlen operator is exposed through CUDA.

Native routed MoE CUDA kernels reuse Candle’s retained MIT sources and the same
pinned CUTLASS revision `e406c186f510a15091cce01f782020ceb7ba8eb5`.
Set `CUTLASS_ROOT` for an offline checkout; otherwise CMake reuses the Candle
cache or downloads that pinned revision. Installation includes the retained
CUTLASS BSD and Candle kernel MIT notices under `share/licenses/tei_torch`. Kernel-level exact routing
tests do not establish trained checkpoint accuracy.

## Model support

Native implementations cover the encoder, decoder and multimodal families in
[model coverage](libtorch-model-coverage.md). Classifier, SPLADE and typed decision
heads are architecture-specific; arbitrary custom checkpoint modules are rejected.
Single and sharded safetensors are supported, with a pure Rust PyTorch checkpoint
fallback. Loading and inference require no Python runtime.

CUDA packed attention requires FP16 or BF16. Automatic dtype selection respects
checkpoint configuration and Gemma's BF16 requirements; CPU uses FP32. Explicit
dtype selection remains available. Relative-bias and ALiBi attention use actual
sequence lengths without padded activations.

Supported causal text models can reuse radix prefixes. Bidirectional, media and
typed decision requests reject folded inputs; the backend reports this capability
to the router. Optional MLP FP8 kernels preserve attention, normalization and head
dtypes and reject unsupported geometries rather than padding them.

Gemma image features use a bounded per-model cache (32 entries, 64 MiB). CUDA
events protect cached values across streams, and graph replay uses independent
storage. Cache hits, misses and eviction have different serving costs.

## Build and run

### NVIDIA Docker image

The LibTorch NVIDIA image targets SM80, SM86, SM89, SM90, SM100, and SM120 in
one image and selects device kernels at runtime. It defaults to the LibTorch
backend. SM75 is unsupported. The image uses CUDA 13.0 and requires a compatible
NVIDIA driver and NVIDIA Container Toolkit on the host.

Build from the Infinity repository root; a GPU is not required for compilation:

```bash
docker build -f libs/infinity-v3/Dockerfile-libtorch \
  -t infinity-v3:nvidia-local libs/infinity-v3
docker run --rm infinity-v3:nvidia-local --help
docker run --rm --gpus all -p 7997:80 \
  -v "$PWD/model:/models/model:ro" \
  infinity-v3:nvidia-local --model-id /models/model \
  --torch-device cuda --dtype bfloat16
```

The build pins Torch 2.14.1 and CUDA 13.0.2. Python installs the C++ SDK in the
builder; the runtime contains native shared libraries and their notices.
Cargo dependency notices are included under `/usr/share/licenses/infinity_v3/cargo/`.
Use `--build-arg BUILD_JOBS=16` on a host with sufficient RAM to speed up compilation.
The default is four jobs. Supported SM targets and resolved SDK dependencies
are recorded under `/usr/share/infinity_v3/` inside the image.

GitHub Actions publishes NVIDIA images as
`ghcr.io/michaelfeil/infinity_v3:nvidia-<version>` when a `v3.*` tag is pushed.
Prerelease versions retain the full suffix, for example
`nvidia-3.0.0-alpha.2`, without replacing the rolling `latest` tags. Main and tag
builds publish the NVIDIA and CPU variants; manual workflow runs can select `nvidia`
alone or `all` variants.

### Native build

Install the [LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html)
for **2.14.1**, choosing CPU or a CUDA build compatible with your GPU driver.
Set `LIBTORCH` to its root (containing `include/`, `lib/` and `share/`). Building
requires CMake 3.25 or newer and a C++20 compiler. CUDA distributions require a CUDA toolkit to compile the native packed operators.

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

# Compile both backends; Candle remains the default.
cargo build --release -p infinity_v3-router --features libtorch

target/release/infinity_v3 \
  --backend libtorch --torch-device cuda --dtype float16 \
  --model-id /path/to/bert-model --pooling mean \
  --backend-device-ids 0,1
```

An A10 uses the CUDA path. Each selected GPU loads a complete model and uses
infinity_v3's existing replica queue; this is data parallel serving. Use `--device-id 0`
for one GPU or omit device selection to use all visible CUDA devices.
`CUDA_VISIBLE_DEVICES` controls visible ordinals. Auto chooses CUDA when a device
is available, otherwise CPU. Other accelerator families are not implemented yet.

For CPU, use `--torch-device cpu --dtype float32` and omit replica selection.
To build without Candle, replace build features with
`--no-default-features --features libtorch,http`, and select `--backend libtorch`
when starting the server. Existing Candle builds do not require LibTorch.

CUDA builds default to a multi-architecture native library targeting SM80,
SM86, SM89, SM90, SM100, and SM120. SM75 is outside the supported baseline.
Each target includes native code and PTX.
The separate SM90a MoE optimization is selected only on Hopper; other GPUs use
the portable path. The bundled LibTorch distribution must also support the target
GPU, and individual operations and dtypes retain their hardware requirements.
The full target list requires a CUDA toolkit supporting SM100 and SM120;
older toolkits require a narrower explicit architecture list.

`LIBTORCH_CUDA_ARCHITECTURES` accepts a semicolon-separated CMake architecture
list. Candle's `CUDA_COMPUTE_CAP` setting no longer narrows LibTorch builds.
Distribution builds can pin the full target list explicitly:

```bash
LIBTORCH_CUDA_ARCHITECTURES='80;86;89;90;100;120' \
  cargo build --release -p infinity_v3-router \
  --no-default-features --features libtorch,http
```

Use `LIBTORCH_CUDA_ARCHITECTURES=90` for a faster H100-only development build.
Multi-architecture compilation takes more time and produces a larger library;
compilation coverage does not replace runtime qualification on each GPU family.

CMake handles LibTorch's compiler/ABI flags. The bridge is a small shared library;
the server's build rpath points to its Cargo output directory. When moving the
server to another machine, ship `libtei_torch.so` from that build alongside the
executable **and the complete `share/licenses/tei_torch` directory from the same
Cargo build output**. Keep that license directory in the deployed package.
It contains this project's license and notices. CUDA builds also include the
retained Flash Attention authors/license/provenance, CUTLASS, Candle kernel,
cuBLASLt and layer-norm notices incorporated into the bridge.
Provide LibTorch's shared libraries via `LD_LIBRARY_PATH` and retain their
licenses/notices as well. The executable and bridge also search their own directory
on Linux. No `libtorch_python` library is linked by the bridge.

## Validation and performance

`INFINITY_TORCH_INPUT_UPLOAD_MAX_BYTES` limits each native request's estimated input
upload and preprocessing storage, defaulting to 536870912 bytes (512 MiB).
It includes actual packed token buffers, media position axes, media payloads and
conversion temporaries, radix maps, and native descriptor copies. Repeated media
descriptors are charged independently. Requests over this limit return a backend
error before input transfers or deferred output allocation. This limit does not
cover model activations, workspaces, output storage, or allocator-reserved memory.

`INFINITY_TORCH_HOST_OUTPUT_MAX_BYTES` limits estimated Rust host output storage per
submission, defaulting to 536870912 bytes (512 MiB). The estimate includes result
payloads, token-row vector metadata, output map storage, and the temporary flat
buffer used by CPU inference. Typed decisions include both flat logits and copied
result logits. Requests that exceed the limit return a backend error before model
inference. Large serving buffers use fallible reservations so reservation failure
returns an error; subsequent requests can still run. This output limit does not
cover model activations, native staging buffers, upstream request buffers or
allocator-reserved memory.

`INFINITY_HOST_RESULTS_MAX_BYTES` adds one process-wide budget for admitted LibTorch
host results, shared across replicas and requests, defaulting to 536870912 bytes
(512 MiB). It must be a positive, representable byte count. The existing
per-submission limit also applies. Reservations precede output packing and cover
estimated float payloads, row/map storage and CPU packing overlap. Core carries
these reservations with results; HTTP retains them through serialization and the
last transport byte clone. A shared batch remains charged until its last response
owner releases it. Cancellation releases completed results, while deferred native
work keeps its reservation until completion or destruction drains the work.

Core token-result tuples and HTTP sparse, base64 and classification conversions
reserve their additional storage before allocation. Budget rejection returns 429
and preserves backend health so a later admitted request can succeed. Native
memory-admission errors use bridge error kind 2; recognized native allocator
failures use kind 3. Kind 0 clears a previous error after success, and kind 1
retains other failures as fatal. The native output regression exercises an actual
CUDA allocator rejection followed by a valid response. This establishes that
recovery path, not recovery from every device fault or avoidance of all OOMs.

NER label rows, grouping vectors, score scratch and joined words also reserve
requested storage before fallible allocation. Decision probability buffers and
owned response trees use the same pool; JSON and MsgPack still expose objects
with the same fields and last-value behavior for duplicate keys. Derived credits
accumulate without a new reservation allocation for every label or field. They
remain charged through the last response owner, including intermediate storage
already released during conversion, so this policy can reject conservatively.

This pool covers LibTorch numeric results and these HTTP derived results when the
request carries a native result reservation. It does not bound Candle results,
native pinned allocator reservations, parsed request objects, async scheduling
bookkeeping, model memory, socket buffers or allocator overhead. The separate HTTP encoded-response budget
still applies. This is an allocation-admission limit, not a process RSS ceiling.
Direct Core consumers must keep response metadata alive with extracted results.

HTTP POST requests also reserve a request slot before reading or parsing the body.
`INFINITY_HTTP_INPUT_MAX_BYTES` limits aggregate reserved request-body capacity and
`INFINITY_HTTP_OUTPUT_MAX_BYTES` limits aggregate encoded-response allocation capacity;
each defaults to 536870912 bytes (512 MiB). The output budget must be between
65536 bytes (64 KiB) and 4294967295 bytes (`u32::MAX`) on 64-bit platforms;
on 32-bit platforms its maximum is Tokio's smaller `Semaphore::MAX_PERMITS`.
Values outside this range are rejected at startup. Each admitted request reserves the
full configured `--payload-limit`, and remains admitted until its response bytes
are consumed or dropped. Output capacity is reserved before JSON or MessagePack
serialization allocates its buffer, including both old and new allocations during
buffer growth, and follows cloned response bytes through
transport handoff. Exhausted capacity returns 429 without waiting; the existing
per-body size limit still returns 413. These HTTP budgets do not include parsed
object overhead, Core result vectors, model memory or socket buffers. Untyped
framework/document format conversion is limited to 2 MiB.

```bash
cargo test -p infinity-backend-libtorch --lib

# Requires a working NVIDIA GPU and the CUDA LibTorch distribution.
cargo test -p infinity-backend-libtorch \
  cuda_varlen_matches_cpu_for_ragged_outputs -- --ignored
```

CPU parity checks cover ragged pooled/raw output, all three pooling modes,
sequence isolation, sharded checkpoints and weight prefixes. Error tests cover
malformed batches and missing native weights. The GPU parity test exercises the
actual packed CUDA varlen operator and moving a model onto the inference worker.

Native model and kernel fixtures can also be built independently:

```bash
cmake -S backends/libtorch/cpp -B /tmp/tei-native-tests \
  -DCMAKE_PREFIX_PATH="$LIBTORCH" -DCMAKE_BUILD_TYPE=Release \
  -DINFINITY_BUILD_NATIVE_TESTS=ON -DINFINITY_RUN_CUDA_TESTS=ON
cmake --build /tmp/tei-native-tests --parallel
CUDA_VISIBLE_DEVICES=0 ctest --test-dir /tmp/tei-native-tests --output-on-failure
```

Omit `INFINITY_RUN_CUDA_TESTS` to register only CPU tests. Shared CUDA libraries must
be discoverable by the linker and runtime when using a CUDA distribution.

CUDA varlen parity has been validated on an H100. A10 throughput still needs
hardware validation. Native C++
avoids Python dispatch, but eager tensor operations still launch GPU kernels.
Set `INFINITY_TORCH_CUDA_GRAPHS=1` to enable an experimental four-shape replay cache
for eligible dense models and the validated Hopper BF16 Qwen3 MoE geometry.
Other routed configurations run eagerly. Cache keys include exact sequence boundaries; no
tokens are padded. Capture cost and cache misses matter for serving latency.
Shapes need three consecutive requests before capture. Alternating misses run
eagerly. `INFINITY_TORCH_CUDA_GRAPH_ENTRIES` defaults to four and
`INFINITY_TORCH_CUDA_GRAPH_REUSE` to three. A lifetime budget of eight capture attempts
(`INFINITY_TORCH_CUDA_GRAPH_MAX_CAPTURES`) allows a bounded working-set turnover;
afterward existing graphs still replay and other shapes run eagerly.
`INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES` defaults to 536870912 bytes. Admission charges
private-pool reserved segments plus static CUDA input buffers, cumulatively even
after eviction. An oversized trial capture stops further captures, so the bound
can overshoot by one trial pool. These budgets bound graph admission, not total
model/allocator memory or CUDA graph executable metadata. Eviction reclaims only
the retired private pool; live output tensors keep their storage. Pools retained
by live outputs are retried before later requests allocate. Capture streams
are owned by the cache and reused only after their previous graph is destroyed, with a separate stream
for each live graph and at most one additional trial stream per device. No global
allocator cleanup runs per request. Set entries or attempts to zero to disable capture.
Cache teardown drains pending replay before releasing graph workspaces and streams.
The default capture limit is 4096 tokens; `INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS` accepts
a positive override. Unsupported capture operations run eagerly. Media capture
requires the additional experimental flag described below.

`INFINITY_TORCH_CUDNN_VARLEN=1` enables experimental packed cuDNN attention for eligible
BERT and eligible decoder requests on supported GPUs and cuDNN versions.
Restrictive sliding windows retain Flash attention. This option remains opt-in.
`INFINITY_TORCH_FULL_PRECISION_GEMM=1` disables reduced-precision GEMM accumulation
and TF32 for this process; experiments have not established a general accuracy
or latency benefit.
ModernBERT CUDA uses the Candle normalization kernel adapter by default, with
matching half-precision rotary arithmetic and private Torch-pinned attention.
Nomic v2 MoE uses model-dtype GELU and beta LayerNorm with unrounded residual
statistics.
CUDA mean pooling uses Candle's exact model-dtype reduction tree in one packed
launch over selected sequence spans; raw token outputs retain their original widths.
`INFINITY_TORCH_EXACT_ENCODER_NORM=0` selects ATen normalization for diagnostics.
`INFINITY_TORCH_MEDIA_CUDA_GRAPHS=1`, together with `INFINITY_TORCH_CUDA_GRAPHS=1`, enables
experimental image/audio graph capture. The cache keys include exact media shapes
and descriptors and copy media values and prepared audio indices on replay.
Unsupported captures fall back to eager inference. Trained media accuracy and
performance remain under evaluation.

Detailed benchmark measurements and qualification reports are deferred to a
follow-up PR. Local reports under `docs/benchmarks/` are ignored by Git.

## Comparing against Candle

The `compare_bert` example loads the same checkpoint into both runtimes on CUDA
devices 0 (LibTorch) and 1 (Candle), concurrently, with FP16 and CLS pooling by
default. The optional final argument selects `cls`, `mean`, or `last_token`
pooling. It covers single requests, uniform batches and ragged batches. Inputs
contain deterministic synthetic IDs bounded to the checkpoint vocabulary, with
configured special tokens and the router position offset; tokenization and HTTP
are excluded.
Both runtimes return host embeddings before the timer stops, including input
transfer, model execution, pooling and output transfer. Input cloning is excluded
equally. It checks embedding cosine agreement before measuring each workload.

Use an optimized build for both backends:

```bash
CUDA_COMPUTE_CAP=90 LIBTORCH_CUDA_ARCHITECTURES=90 CARGO_PROFILE_RELEASE_LTO=false \
  CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo build --release -p infinity-backend-libtorch \
    --features benchmark-cuda --example compare_bert

target/release/examples/compare_bert /path/to/bert-model 100 0 1 > bert-results.json
```

Set both architecture overrides for your GPU (90 for H100, 86 for A10). Each workload uses
20 warmups per runtime and a synchronized start for parallel measurements. GPU
ordinals are configurable; swap them to check hardware bias. The JSON includes P50/P95 latency, token throughput at P50 and output
agreement. These are synchronous backend measurements, not HTTP throughput or
multi-GPU scaling measurements. GPU clocks are not locked by the harness.

For real 1K/2K-token inputs, build the `compare_decoders` example with the
same `benchmark-cuda` feature and select a checkpoint declaring at least
2048 positions. The benchmark never changes the checkpoint's position limit.

```bash
INFINITY_BENCHMARK_LONG_CONTEXT=only INFINITY_TORCH_CUDA_GRAPHS=0 \
  target/release/examples/compare_decoders /path/to/qwen3-model 200 0 1 float16 last_token
```

Repeat with GPU ordinals `1 0`, then with `INFINITY_TORCH_CUDA_GRAPHS=1` and
`INFINITY_TORCH_CUDA_GRAPH_MAX_TOKENS=16384`. Use `mean` for ModernBERT embedding
checkpoints. The five workloads include actual 1024/2048-token sequences,
eight-sequence batches, and ragged 1024/1536/2048/1280 inputs. Report the
observed graph count alongside timings; graph requests can fall back to eager
execution. `INFINITY_BENCHMARK_LONG_CONTEXT=1` appends these to the short suite.

CUDA embedding and classifier outputs use one persistent CPU completion thread per
replica. Submission returns either a ready result or an owned completion handle,
after enqueuing D2H into reusable pinned storage and handing work to the completion
thread. Completion waits for the CUDA event and allocates
the final Rust result vectors directly from that storage, avoiding an intermediate
flat host copy. CPU inference completes inline.

Two output leases alternate A/B per replica; the next submission waits until its
slot finishes transfer and result packing. Serial traffic warms both slots.
Buffer capacities grow to the largest observed batch and are reused.
Pinned storage, FP32 device staging, and tensor-retention vector capacity are
reused after warmup. Each lease owns its pinned
input staging buffers, output buffer, and GPU transfer tensors through completion
and result construction. Real token arrays, offsets, radix maps, and pooling spans
are uploaded without padding. Upload, graph replay, output conversion, and D2H
remain ordered on the same CUDA stream; the next replay cannot overwrite buffers
before the preceding copy consumes them. No separate copy stream is introduced.

Request metadata stays owned until host results are ready, including after client
cancellation. Only successful deferred submission releases the exclusive model
lease early. Completion failures close the replica pool, and shutdown drains and
joins the completion worker before destroying the model and its CUDA graphs.

`INFINITY_TORCH_OUTPUT_STAGING_MAX_BYTES` limits the combined retained pinned inputs,
pinned outputs and GPU output buffers across both staging slots per replica.
It defaults to 536870912 bytes (512 MiB). Idle slot buffers are released before
growing a buffer when needed to fit the budget; pending results stay owned until
their completion lease is released. A submission that cannot fit returns a backend
error before allocating additional staging storage. Increase the limit for larger
raw outputs only when the deployment has sufficient host and device memory.
This limit covers staging tensor storage, not model weights, activations or memory
reserved by the host/CUDA caching allocators.

`INFINITY_TORCH_PINNED_STAGING_MAX_BYTES` (default 512 MiB, positive) limits native
pinned input/output storage across all replicas in the process. Native pins use
direct CUDA host allocations instead of Torch's pinned size-class cache. Their
platform-host-page-rounded allocation credit remains owned by the underlying tensor storage,
including idle slots and surviving views; it is returned only after successful
physical release. Cancellation and partial submission drain the lease's CUDA
stream before releasing its storage. A failed release keeps its credit charged.
Admission supplements this native bound with an estimate from Torch's observed
`allocated_bytes` and retained block count, including cached blocks. It allows
both host-page boundaries for unaligned registered blocks and reclaims completed
unused cache blocks before rejecting for pressure. These statistics do not expose
all foreign backing storage, such as unused pinned reserve segments. The guarantee
covers directly owned native staging; total foreign Torch pins, concurrent foreign
allocations, other host allocations, and driver bookkeeping remain outside it.

`INFINITY_TORCH_ATTENTION_BIAS_MAX_BYTES` (default 512 MiB, positive) bounds each
packed attention bias construction. ALiBi admission includes its backing storage
and a conservative sum of construction temporaries. DeBERTa also checks bucket
table construction against this limit. Requests that exceed the limit fail before
these buffers are allocated; their token lengths and model math are unchanged.
This is a limit for these attention buffers, not a total activation, workspace or
CUDA allocator reservation budget.

The native bridge also applies `INFINITY_TORCH_TRANSIENT_MAX_BYTES` (8 GiB by default) before request uploads and deferred output allocation. It caches a profile of the ready model’s actual matrix dimensions, model dtype, attention geometry, routed experts and media towers. Its conservative tensor estimate includes residual/QKV/FFN intermediates, F32 conversion buffers, dense or packed attention bias and relative buckets, recurrent states, retained shared KV tensors, image/audio processing and requested outputs. SPLADE charges its full token-by-vocabulary tensor. Unknown model or requested media profiles are rejected. Inputs are not padded or truncated.

CUDA requests additionally reserve that estimate against current driver free
memory, shared across native models using the same device in this process.
`INFINITY_TORCH_DEVICE_MEMORY_HEADROOM_BYTES` reserves a positive allowance, 1 GiB by
default, beyond outstanding request estimates. This allowance is a policy for
unestimated allocations, not a proven upper bound on vendor workspaces.
Admission uses the smaller of driver free memory and the configured Torch
allocator quota minus all allocator reserved bytes. Allocator reserved-but-inactive
memory is not counted as available memory. Nondefault quotas require native
allocator accounting; unsupported accounting rejects before request allocation.
Reservations remain live until GPU completion, including cancellation and
error cleanup. Synchronous calls drain their stream before releasing admission;
deferred calls use their existing output completion event and lease.

This does not control foreign processes or other allocators racing after the
free-memory query, or establish
an upper bound on private graph pools, retained vendor workspaces and every
model's actual tensor peak. Those limits require separate accounting.

`INFINITY_TORCH_RETAINED_WORKSPACE_MAX_BYTES` separately bounds the retained custom
OrdinaryLinearExecutor, Gemma and Qwen3.5 radix-projection CUDA workspaces per device
within this process, at 512 MiB by default. Models sharing a device use the
most restrictive configured allowance. Admission happens before allocation;
when the allowance is occupied, these optimizations use their existing ATen
fallback and retry the same supported plan after workspace credit is released.
Cached plans retain the reservation with their exact workspace
allocation, including old generations. This cap does not include ATen-owned
cuBLAS/cuDNN workspaces or temporary routed-MoE scratch.

This estimate is a per-request tensor admission limit. It does not account for vendor workspace internals, allocator rounding or reserved memory, model weights, graph pools retained from earlier requests, or simultaneous requests on the same device. Those require separate available-memory and concurrency admission; this setting alone is not a guarantee against OOM.
