# Native LibTorch backend

Infinity v3 runs inference through LibTorch, the C++ interface of PyTorch,
with a Rust HTTP server, tokenizer workers and batch scheduler. Serving requires
no Python runtime or TorchScript export. Images containing LibTorch select it
automatically; no `--backend` flag is needed.

- **Deploy**: use the NVIDIA image below, or follow the [quickstart](../README.md#getting-started).
- **Models**: see [supported architectures and tasks](libtorch-model-coverage.md).
- **Clients**: see [embedding formats and client usage](embedding-tensors.md).
- **Configuration**: see [CLI and request options](cli.md).

## Build and run

### NVIDIA Docker image

Use the prebuilt image with an SM80+ GPU, a compatible NVIDIA driver and
NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0-alpha.16 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cuda --dtype float16 --port 8000
```

Choose a version from the [container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3).
Use `--device-id 0` for one GPU; otherwise each visible GPU runs a complete replica.
For CPU builds, use `--torch-device cpu --dtype float32`.

<details>
  <summary>Optional dynamic FP8</summary>

### Optional dynamic FP8

For dense Llama, Mistral, Qwen2, Qwen3 and Qwen3.5 decoders, `--enable-fp8-dynamic`
quantizes MLP, Q/K/V and attention output projections by default (`all` scope).
Set `INFINITY_TORCH_FP8_SCOPE=mlp` to quantize only MLP projections.
Supported BERT-family and ModernBERT encoders retain MLP-only dynamic FP8;
the `all` scope is only supported for the dense decoders listed above.
Dense Gemma4 also supports MLP-only FP8; attention, per-layer embeddings and
media encoders keep their original precision. Gemma4 MoE is not yet supported.
Attention, normalization, embeddings and heads retain the selected FP16/BF16
dtype. Weights are quantized once at startup and activations each forward pass.
Qwen3.5 requires BF16; `all` also quantizes DeltaNet QKV, Z and output
projections. Its small decay/update gates, convolution, recurrent state, vision
tower and output heads retain their original precision. Qwen3.5 MoE is unsupported.
Dynamic FP8 changes model outputs and remains opt-in. Enabling it now uses
more FP8 projections for dense decoders; select `mlp` to retain the previous scope.

```bash
target/release/infinity_v3 --model-id /path/to/model \
  --torch-device cuda --dtype bfloat16 --enable-fp8-dynamic
```

To use MLP-only FP8:

```bash
INFINITY_TORCH_FP8_SCOPE=mlp target/release/infinity_v3 --model-id /path/to/model \
  --torch-device cuda --dtype bfloat16 --enable-fp8-dynamic
```

Dynamic FP8 requires an unquantized FP16/BF16 checkpoint, native decoder
arithmetic, projection widths divisible by 16, and SM89, SM90, SM100 or SM120.
It uses the existing native build. The new kernels compile for all four
architectures; runtime validation currently covers H100.

`INFINITY_TORCH_CUDNN_VARLEN=1` optionally selects cuDNN attention for eligible
devices, sequence lengths and attention windows. This also applies to Llama
with all-linear FP8 or BF16 hidden width 4096; its normalization arithmetic is
preserved, while cuDNN may change output rounding. SM89/SM120 retain
Flash attention. Omitting this option preserves existing attention selection.

BF16 Qwen3/Llama prefill also uses vectorized normalization and SwiGLU kernels.
Hopper uses a cached gate/up GEMM choice for eligible large batches with the
validated cuBLASLt 13.1.1; other versions, shapes and devices retain ATen.
These optimizations use the native build and require no startup quantization.
The BF16 kernels compile for SM80, SM89, SM90, SM100 and SM120; runtime
validation currently covers H100.

</details>

<details>
  <summary>Build from source</summary>

### Native build

#### Container build

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
Packaging follows ELF dependencies and retains runtime-loaded cuDNN/NVRTC
modules; identical FA4 runtime libraries are shared between GPU bundles.
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

#### LibTorch source build

Install the [LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html)
for **2.14.1**, choosing CPU or a CUDA build compatible with your GPU driver.
Set `LIBTORCH` to its root (containing `include/`, `lib/` and `share/`). Building
requires CMake 3.25 or newer and a C++20 compiler. CUDA distributions require a CUDA toolkit to compile the native packed operators.

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"

# Compile a LibTorch-only server; the backend is selected automatically.
cargo build --release -p infinity_v3-router --no-default-features --features libtorch,http

target/release/infinity_v3 \
  --torch-device cuda --dtype float16 \
  --model-id /path/to/bert-model --pooling mean \
  --backend-device-ids 0,1
```

An A10 uses the CUDA path. Each selected GPU loads a complete model and uses
infinity_v3's existing replica queue; this is data parallel serving. Use `--device-id 0`
for one GPU or omit device selection to use all visible CUDA devices.
`CUDA_VISIBLE_DEVICES` controls visible ordinals. Auto chooses CUDA when a device
is available, otherwise CPU. Other accelerator families are not implemented yet.

For CPU, use `--torch-device cpu --dtype float32` and omit replica selection.
To compile both backends instead, use `--features libtorch` without
`--no-default-features`. Those builds also default to LibTorch; use
`--backend candle` only when a model needs Candle. Existing Candle-only builds
do not require LibTorch.

CUDA builds default to a multi-architecture native library targeting SM80,
SM86, SM89, SM90, SM100, and SM120. Both Candle and LibTorch require SM80 or newer
for supported CUDA builds. SM75/Turing is unsupported.
Each target includes native code and PTX.
The separate SM90a MoE optimization is selected only on Hopper; other GPUs use
the portable path. The bundled LibTorch distribution must also support the target
GPU, and individual operations and dtypes retain their hardware requirements.
The full target list requires a CUDA toolkit supporting SM100 and SM120;
older toolkits require a narrower explicit architecture list.

`LIBTORCH_CUDA_ARCHITECTURES` accepts a semicolon-separated CMake architecture
list of explicit targets >=80 (optionally suffixed with `-real` or `-virtual`).
Special targets such as `all` and `native` are rejected to prevent implicit
pre-Ampere builds. Candle's `CUDA_COMPUTE_CAP` setting no longer narrows
LibTorch builds.
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

</details>


## Advanced reference

<details>
  <summary>Packed inference and FlashAttention-4</summary>

## Packed inference

Token embeddings, projections, attention outputs and MLP activations use
`[total_tokens, hidden_size]`, with Q/K/V shaped
`[total_tokens, heads, head_dim]`. Sequence boundaries come directly from infinity_v3's
cumulative lengths. No token padding is allocated. Relative position bias paths use attention bias at each sequence's actual length.

On CUDA, eligible shapes use InfinityFA4 when its bundles are installed (see
below). The existing attention paths call `aten::_flash_attention_forward` with device-resident
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

### InfinityFA4

The NVIDIA image includes native FP16/BF16 FlashAttention-4 bundles for SM80,
SM86, SM89, SM90, SM100 and SM120. Supported shapes are global/windowed d64
MHA, causal d128 GQA4, global d128 GQA2/GQA6/GQA8, and d64 DeBERTa relative attention.
DeBERTa avoids quadratic relative-position bucket and attention-bias storage.
FP8 projections retain their existing attention dtype.

FA4 is enabled when built. Set `INFINITY_TORCH_FA4=0` before startup to use
previous attention paths. Unsupported shapes retain their existing kernels;
explicit BERT/decoder cuDNN selection takes precedence. Launch errors propagate.

Source builds opt in through `INFINITY_FA4_ROOT`; see the vendor
[build instructions](../backends/libtorch/cpp/vendor/infinity_fa4/README.md).
Export requires Python and no GPU; serving requires neither Python nor JIT
compilation. H100 has runtime coverage; other targets have compile/link coverage.
Performance and numerical differences depend on the checkpoint and precision.

</details>

<details>
  <summary>Model execution and CUDA initialization</summary>

## Model support

Dense feed-forward blocks use persistent `DenseMlp` objects from
`helpers/dense_mlp.h`. Each `MlpProjection` owns its matrix, bias, cast policy and
input/output clipping bounds. A prepared activation recipe records plain,
packed-gated or separate gate/up arithmetic. Text, vision, merger/projector,
audio and decision-head blocks bind these objects during loading; pooled Dense
chains use the same projection contract.

`prepare(MlpPrecision::Fp8)` quantizes the owned matrices once. Calls use the same
`forward(input, context)` interface in model precision and FP8. Bias and
projection transforms apply in either mode. `MlpContext` selects ordinary GEMM
plans or qualified fused epilogues, using the supplied projection metadata;
it does not own model mathematics. Residuals and normalization remain outside
the MLP. Prequantized input can carry row scales explicitly.

Dense MLP projections own an immutable `MlpLinearMethod`. The ordinary method
owns checkpoint matrices and uses request-local GEMM plans; the FP8 method owns
quantized matrices and scales. A loader selects methods through
`DenseMlp::prepare`, either by precision or by a projection-aware factory.
Preparation replaces all projection methods together only after success.
Clipping, casts, separate bias rounding, and activation recipes stay in the MLP
layers, outside the execution method. Methods advertise GEMM epilogue and
prequantized FP8 input capabilities; the forward interface stays the same.

The executor has no model-name eligibility checks. Server FP8 eligibility and
defaults remain explicit: sharing execution does not qualify additional
checkpoints for FP8 accuracy.

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
to the router. Optional FP8 projections preserve attention, normalization and head
dtypes and reject unsupported geometries rather than padding them. See the
FP8 configuration below.

Gemma image features use a bounded per-model cache (32 entries, 64 MiB). CUDA
events protect cached values across streams, and graph replay uses independent
storage. Cache hits, misses and eviction have different serving costs.

After configuration validation and before loading checkpoint weights, LibTorch
checks the selected CUDA device by allocating a small FP32 tensor, computing
`1 + 1`, synchronizing its stream and verifying the result on the CPU. INFO events identify the check's
start and success in the replica context; CUDA exceptions fail initialization
through the normal startup-error path. CPU devices skip GPU arithmetic. This
preflight does not catch a process-fatal SIGSEGV or validate every model kernel.

</details>

<details>
  <summary>Checkpoint loading and tuning</summary>

## Disk reader

On Unix, LibTorch loads safetensors with the bundled fastsafetensors non-GDS
reader by default. It reads validated tensor ranges concurrently, using two
aligned host buffers per worker and overlapping CUDA uploads. Model weights own
their allocations; bounded staging avoids retaining an entire shard through a
small tensor or an FP8-converted layer. Large dtype conversions also use bounded
pieces. The same path loads single files, shards, dense modules and decision heads.

The reader uses ordinary `pread` workers. It requires no Python, GDS, io_uring,
libaio, NUMA configuration or additional container capabilities. Direct I/O is
requested for suitable local filesystems; RAM/network/FUSE filesystems and
filesystems that reject direct reads use buffered reads. PyTorch `.bin` loading
and the Candle backend are unchanged.

| Environment variable | Default | Behavior |
| --- | --- | --- |
| `INFINITY_TORCH_WEIGHT_READER_THREADS` | `16` | Maximum workers per file, from `1` to `64`; small files use fewer workers. |
| `INFINITY_TORCH_WEIGHT_READER_BUFFER_BYTES` | `33554432` up to 32 GiB total checkpoint size; `268435456` above | Budget for the host buffer pool and, separately, each transient tensor stage, from `65536` to `1073741824` bytes. Host buffers add alignment padding. Tiny files are capped by their data size. An explicit setting overrides the default. |
| `INFINITY_TORCH_WEIGHT_READER_DIRECT_IO` | `true` | Set to `false` or `0` to use the filesystem page cache, for example for repeated loads or a managed model-cache mount. |

Header validation remains in Rust, including offset, shape and complete-file
length checks. Read errors and truncated files fail initialization. Native code
drains reads and CUDA copies before returning or releasing their buffers.

At INFO level, each replica logs `Starting checkpoint load` before reading its
weight files and `Finished checkpoint load` with elapsed seconds after all weights
are loaded. Existing safetensors completion events retain per-file timing and
size details. The `model_load` span identifies the model path, device and process
ID; the router's `backend_startup` parent adds the replica index. Context survives
the blocking loader thread, so concurrent loads remain distinguishable in text
and JSON logs. Initialization failures report the error and elapsed startup time
with the replica context. JSON `--disable-spans` omits these span fields.

</details>

<details>
  <summary>Developer tests and performance measurements</summary>

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

# Manual startup/inference smoke test with a local supported checkpoint.
INFINITY_TEST_CHECKPOINT_PATH=/path/to/model cargo test -p infinity-backend-libtorch \
  cuda_real_checkpoint_startup -- --ignored --nocapture
```

CPU parity checks cover ragged pooled/raw output, all three pooling modes,
sequence isolation, sharded checkpoints and weight prefixes. Error tests cover
malformed batches and missing native weights. The GPU parity test exercises the
actual packed CUDA varlen operator and moving a model onto the inference worker.
Reader tests additionally cover malformed headers, exact dtype bytes, unaligned
tensor offsets, independent tensor storage, bounded dtype conversion, direct and
buffered reads, truncated files, asynchronous buffer reuse and cleanup on errors.

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

</details>

<details>
  <summary>Historical Candle comparisons</summary>

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

Forward execution has no fixed transient-tensor cap or heuristic admission
against free GPU memory. Batch size is controlled by the router's token/request
limits; actual tensor allocations are handled by LibTorch. Allocation failures
continue to propagate through the existing native error mapping. Configure batch
limits to fit the deployed models and hardware; warmup exercises those limits.

`INFINITY_TORCH_TRANSIENT_MAX_BYTES` and
`INFINITY_TORCH_DEVICE_MEMORY_HEADROOM_BYTES` have been removed and are no longer
read. Remove these settings from deployment manifests. The input upload, output
staging, pinned storage, attention-bias and retained-workspace limits described
here account for their specific allocations independently of forward activations.
Synchronous calls still drain queued CUDA work on errors; deferred calls retain
their completion events and output leases through result construction.

`INFINITY_TORCH_RETAINED_WORKSPACE_MAX_BYTES` separately bounds the retained custom
OrdinaryLinearExecutor, Gemma and Qwen3.5 radix-projection CUDA workspaces per device
within this process, at 512 MiB by default. Models sharing a device use the
most restrictive configured allowance. Admission happens before allocation;
when the allowance is occupied, these optimizations use their existing ATen
fallback and retry the same supported plan after workspace credit is released.
Cached plans retain the reservation with their exact workspace
allocation, including old generations. This cap does not include ATen-owned
cuBLAS/cuDNN workspaces or temporary routed-MoE scratch.

Builds can change the application environment-variable base using
[`ENV_VAR_BASE`](build-configuration.md), including Docker build arguments.

See [Ministral 3 multimodal embeddings](ministral-3-multimodal.md) for the public
Mistral3/Pixtral checkpoint and independent reference validation.

</details>

<details>
  <summary>UltraBERT implementation and parity tests</summary>

### UltraBERT

UltraBERT uses packed bidirectional GQA with a separate RoPE base and symmetric
attention window for each layer. A `null` window selects full attention; positive
windows include both endpoints. The backend accepts base or `ultrabert.` weight
prefixes and defaults to a 32K position limit when the checkpoint omits it.
CLS, mean and last-token pooling, sequence classification/reranking, and token
classification are supported. Bias-enabled projections and SPLADE are rejected.
CUDA graph capture is available for text requests; bidirectional prefix folding
is disabled.

Generate independent reference fixtures from the public model's Hugging Face
implementation (requires its compatible Transformers version and custom code):

```sh
python scripts/ultrabert-reference.py --model /path/to/UltraBERT-Embed-150M \
  --device cuda --output /tmp/ultrabert-reference.json
ULTRABERT_MODEL=/path/to/UltraBERT-Embed-150M \
ULTRABERT_REFERENCE=/tmp/ultrabert-reference.json \
ULTRABERT_DEVICE=cuda:0 ULTRABERT_DTYPE=bfloat16 \
  cargo test -p infinity-backend-libtorch --test ultrabert_parity \
  ultrabert_matches_hf -- --ignored --nocapture
```

Use `--head sequence` or `--head token` to generate classification references.
Combine their JSON arrays into one file and set
`ULTRABERT_CLASSIFICATION_REFERENCE` for `ultrabert_classifiers_match_hf`.
Set `INFINITY_TORCH_CUDA_GRAPHS=1` when running `ultrabert_cuda_graphs_capture`.
Native CPU tests additionally cover local-window boundaries, per-layer rotary
bases, independent attention width, ragged isolation and malformed configurations.

</details>

<details>
  <summary>OneJev block attention, configuration and validation</summary>

### OneJev block attention

Dense Qwen3.5 decision checkpoints declare their attention and readout in
`config.json`:

```json
"modeling_overrides": {
  "attention": {"full_attention": "block_bidirectional", "linear_attention": "causal"},
  "readout": "slot_lm"
}
```

This enables bidirectional attention inside the complete shared prefix and inside
each question suffix. Prefix tokens cannot attend to a suffix. Gated DeltaNet
layers remain causal.

Serve a local checkpoint in a LibTorch build/image with `--dtype bfloat16
--decision-protocol onejev --model-id /path/to/checkpoint`. The `/v1/systemone`
request and answer-letter scoring format are unchanged. Attention and readout
metadata are loaded from `modeling_overrides` or the legacy
`baseten_modeling_overrides` key. If both are present, their values must be
identical; conflicting declarations are rejected.

For OneJev block attention, `usage.input_tokens`, `x-compute-tokens`, and the
configured billing input-token header count each distinct shared prefix once
per request plus every question suffix, independently of batching or cache hits.

Within each backend batch, identical complete prefixes share full-attention KV,
FP32 recurrent states and convolution history. Prefix tokens, token types,
positions, all mRoPE axes, image pixels/layout/placement and the block boundary
must match. Caller-provided image hashes alone cannot authorize reuse. Suffixes
execute independently from immutable prefix snapshots. Caches are released after
the group finishes; batches split by token limits or assigned to different
replicas recompute their prefixes. Partial-prefix reuse, persistent cross-request
caches, CUDA graph capture for this mode, audio inputs and learned pointer
readout heads are not supported.

Block-attention checkpoints with a Qwen3.5 vision tower and the checkpoint's
image processor configuration also accept OneJev image inputs. This requires a
build containing the multimodal block implementation; released images through
`v3.0.0-alpha.4` reject images in block mode. Supply up to 16 images in user
messages inside `state`; question instructions remain text/structured content:

```json
{
  "state": {
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Inspect this picture."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<PNG_BYTES>"}}
      ]
    }]
  },
  "questions": {"red": {"type": "noul", "instructions": "Is the object red?"}}
}
```

Replace `<PNG_BYTES>` with base64-encoded PNG content. Inline data URLs work
without a hostname allowlist. Supported cloud object-storage HTTPS media URLs
are allowed by default; `--image-allowed-hosts` replaces the list (empty disables
remote media). Each image may use up to 4,096 vision tokens. The complete prompt
for each question, including state, image tokens, question and options, must fit
the configured/requested token limit; use `--max-batch-tokens 32768` for a
32,768-token budget, subject to the model's context limit. State has no separate
character limit. OneJev rejects oversized prompts rather than truncating them.
Image spans expand before the block boundary is
finalized, and token limits apply to the expanded sequence. Spans cannot cross a
block boundary. Both prefix and suffix execution retain the original mRoPE axes.
Direct native decision calls with images must supply `media_positions`; block
image requests without these coordinates are rejected before GPU uploads.
Image uploads and vision features are reused within a native call when their
borrowed pixel storage and layout match, including across prefix and suffix
passes, and a shared visual prefix is prefilled
once. Packed FlashAttention, dense MLPs, and the existing recurrent/chunked GDN
selection remain active, with no sequence padding or per-question full-forward
fallback.

Run the independent full-sequence Transformers comparison (Torch 2.14.1 and
Transformers 5.19.0):

```sh
python scripts/onejev-block-image-reference.py --library /path/to/libtei_torch.so \
  --device cuda:0 --dtype bfloat16 --output /tmp/onejev-images.json
```

The default uses tiny random hybrid/vision weights and checks mixed text/image
batches, shared/distinct pixels, multiple images, changed rotary coordinates,
ragged suffixes and prefix-only rows, in both batch orders. It bounds FP32 logits,
BF16 centered option logits, probabilities and well-separated decisions against
full-sequence block masks. Centering removes a common offset, which does not
change OneJev scores; the report also records uncentered logit drift. BF16
full-sequence, split-prefix and chunked GDN execution are not bitwise equivalent.
Add `--model /path/to/checkpoint --image-grid 64` for a local trained-model
control with 1024 merged tokens per image; native batches with several distinct
long prefixes also exercise the existing H100 chunked GDN gate. The report includes
warmed packed-versus-serial timing for eight questions with one shared image.
These timings cover native inference only, excluding startup, image decoding,
HTTP and tokenization; they do not qualify every model, image size or GPU.
The comparison defaults to FP32 on CPU and BF16 on CUDA; native CUDA DeltaNet
requires BF16. Add `--profile` to check that distinct text prefixes sharing pixel
storage reuse one vision projection, including a separate control sharing images
across prefix and suffix passes. CUDA checks one upload/conversion pair; the report also
indicates whether the chunked GDN kernel ran. Input-budget checks ensure shared
uploads are accounted for once and independent pixel buffers remain charged.

Independent full-sequence reference fixtures can be generated with Transformers
5.19.0 using `scripts/onejev-block-reference.py --output /tmp/onejev-reference`.
This creates a small random hybrid model without downloading weights. Add
`--model /path/to/checkpoint --device cuda --dtype bfloat16` to validate a trained
checkpoint. Run `onejev_block_parity` with `ONEJEV_BLOCK_REFERENCE` pointing to
`reference.json`; set `ONEJEV_DEVICE=cuda:0 ONEJEV_DTYPE=bfloat16` for GPU checks.

Startup overlaps configuration validation (including CUDA discovery) with model frontend initialization. All models must pass validation before checkpoint loading begins. The CUDA preflight, health check, and full configured warmup remain required before readiness. Startup logs report frontend and configuration-validation durations; these phases overlap and must not be added together.

See [ModernBERT startup measurements](modernbert-startup.md) for the draft baseline, candidate results, and remaining cold-start limitations.

The default is selected once using the total on-disk size of all checkpoint shards and selected sidecars, independent of shard layout. Explicit buffer settings take precedence. The 32 GiB cutoff is a simple heuristic based on the measured small/dense and large-MoE workloads.

Reader environment settings are read and validated once per process on first safetensors use. Explicit settings stay fixed for all models and shards; when the buffer override is absent, each model resolves its adaptive default independently. Restart the process to change these settings.

</details>
