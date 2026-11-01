<!-- Copyright Infinity Core Contributors. Licensed under the repository LICENSE. -->

# Oxide CUDA backend

The experimental `oxide` backend executes BERT using GPU kernels authored in
Rust and compiled by NVIDIA's cuda-oxide compiler, with cuBLAS BF16 matrix
multiplication and an optional FA4 AOT attention provider. Its serving process
uses NVIDIA's `cuda-core` runtime and has
no LibTorch, PyTorch, Python, or CUDA C++ model execution dependency.

This is an experimental first slice of the Infinity model catalog. MiniLM has
completed repeated H100 performance and small retrieval-quality comparisons
against LibTorch, described below. Broader model and production qualification
remain necessary.

## Current behavior

- Encoder BERT, absolute position embeddings, LibTorch-compatible tanh GELU, and BF16 compute with
  FP32 accumulation for projections and FP32 embedding outputs.
- Single-file or standard sharded safetensors; F32/F16/BF16 checkpoint weights.
  Required shapes, finite values, shard paths, and model options are validated.
  Standard Hugging Face cache snapshot symlinks to the same repository's flat
  `blobs/<SHA1-or-SHA256>` directory are supported. Other external weight paths
  and lexical shard traversal are rejected.
- Packed bidirectional attention and CLS, mean, or last-token pooled embeddings.
- Existing HTTP endpoints, tokenizer workers, batching interface, response
  encodings, and request-controlled normalization. Unsupported model families,
  trained Dense projections, module pipelines, dtypes, and tasks fail explicitly.

The Rust kernels fuse embedding normalization and QKV preparation; register-cached
384-wide normalization specializations and a parallel mean-pool kernel cover the
MiniLM geometry. The model executor retains at most eight workspaces: one active
plus up to seven inactive device and pinned-host arenas. Each capacity dimension
is the next power of two at or above a minimum: 32 tokens, two packed boundary
entries, and one pooled selection. Arenas therefore grow geometrically, and an
in-budget active arena is reused whenever all three capacities suffice. Otherwise
the executor promotes the smallest cached arena that fits by its actual allocated
bytes, not by a request-size estimate. Capacity reuse never changes logical
tensor extents: actual token counts, packed sequence boundaries, and pooled
selections drive the kernels. Inputs and metadata use a compact H2D copy, and
only actual pooled rows are copied D2H. There is no maximum-length allocation or
cache dimension. Unused capacity is neither transferred nor treated as a sequence.
Inactive workspaces are evicted LRU under the count limit and a 256 MiB budget
covering active plus cached arena buffers. Model weights and driver/provider
metadata sit outside that budget. One active request may exceed the budget; doing
so empties the inactive cache, and that oversized active arena is reused only for
an identical capacity key and otherwise released rather than cached.

Arena event counters start when the serving metrics recorder is installed. The
live-byte gauge also includes arenas allocated during earlier model warmup.
Metrics use these exact names. `oxide_arena_requests_total` counts forward
requests that reach arena planning. `oxide_arena_allocations_total` counts newly
allocated arenas. `oxide_arena_cached_reuses_total` counts cached arenas promoted
to active. `oxide_arena_evictions_total` counts inactive cache removals and
rejected cache inserts. `oxide_arena_live_bytes` is a gauge of the
physically retained arena bytes: allocation increases it, and only a successful
drop after stream completion decreases it. `oxide_arena_retained_bytes_total` is a
monotonic counter of arena bytes whose failed-drain path deliberately retained
allocations, covering both an active workspace forgotten after a failed
error-path synchronization and arenas dropped while synchronization fails.
Retained exceptions stay included in `oxide_arena_live_bytes`; the counter
reports their cumulative byte total rather than a current size. These failed-drain
retentions can exceed the normal active-plus-cache budget.

Execution uses ordinary ordered CUDA streams. CUDA graph capture, replay, and
the `INFINITY_OXIDE_CUDA_GRAPHS` setting have been removed from the oxide backend
only; LibTorch CUDA graph support is unchanged. Kernel modules, allocations, the
primary context, and streams remain owned until outstanding work drains, including
on errors. Arena storage is prepared on allocation; exact-row matrix plans are
prepared when the logical token count changes, outside the layer loop.
Without an explicitly selected AOT provider, attention uses the Rust SIMT kernel.
PDL is not implemented yet.

PDL is the intended optional scheduling extension. Enable it only for qualified
producer/consumer pairs with explicit dependency synchronization, checked native
artifacts and measured gains; opaque provider calls keep ordinary stream
ordering until their launch contract is verified. Correctness must hold without
overlap. PDL does not eliminate host launch calls, so fusion and prepared dispatch
remain separate optimizations. It introduces no capture/replay or fixed-shape
requirement. See [NVIDIA's PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Tensor and matrix foundations

`tensor` provides checked fixed-rank layouts and borrowed `TensorRef`/`TensorMut`
views. Shape, stride, span, offset, and storage arithmetic are checked; mutable
views reject overlapping or broadcast writes. Views allocate nothing, clone no
owners, and make no CUDA calls. They describe a borrow rather than ownership of
unfinished GPU work; enqueue boundaries remain unsafe and the engine retains
all buffers until completion. There is no dynamic operator dispatcher.

`math::LinearPlan` prepares row-major `X W^T` geometry and cuBLAS parameters once
per workspace, including optional row padding. BF16, FP16 and FP32 matrix methods
accumulate in FP32; the BERT model still runs only in BF16. The backend binds its
context once per forward, rather than once per matrix operation. FP8/FP4 enum
variants describe storage sizes only and do not enable arithmetic or quantized
checkpoint loading. Packed scales and quantization schemes need separate methods.

The `dispatch_bench` example measures raw argument assembly versus the actual
checked tensor binding on the host, with allocated GPU buffers but no CUDA calls
inside timed loops. It verifies zero heap allocations during dispatch. These
numbers exclude driver enqueue and GPU execution time.

## Weight loading

Loading reads safetensors headers first and keeps validated opened shard files.
Payload reads use at most 64 KiB of conversion scratch. Two reusable pinned
buffers total 1 MiB; completion events prevent their reuse before H2D finishes.
Weights go into final device allocations, including Q/K/V ranges within one
fused destination. The loader retains neither whole shards nor an FP32 copy of
the model. BF16 bytes are preserved; F16/F32 conversion keeps the original BF16
rounding and nonfinite-value checks. Startup waits for every upload before the
backend becomes ready. Model metadata, device weights, and request workspaces
are separate from the transient staging budget.

Checkpoint path, shard-index, expected shape, missing/duplicate tensor and exact
file-length checks remain required. Read/copy failures prevent readiness and
drain outstanding work before resources are released. Native packed-weight
caching and general tied-weight or quantized-model loading are not implemented.

## Separate compiler and serving toolchains

NVIDIA CUDA Rust is pinned at
[`999094fe568fc3080f80d49e9c44e3680ad1c968`](https://github.com/NVIDIA/cuda-rust/tree/999094fe568fc3080f80d49e9c44e3680ad1c968).
The Infinity host builds with the repository's Rust **1.93.1** toolchain;
`cuda-core` requires Rust 1.89 or newer. The separate device package pins
**1.99.1**, `rustc-dev`, Rust sources, and LLVM tools for cuda-oxide and the
native kernel bundle. Serving does not load the Rust compiler or compile kernels on
a request.

The compiler bootstrap was exercised with CUDA **13.1**, Rust's bundled
LLVM **23**, and Clang **18** development headers. Its environment checker and
a real Rust vector-add kernel completed on H100. Follow upstream
[installation requirements](https://github.com/NVIDIA/cuda-rust/blob/999094fe568fc3080f80d49e9c44e3680ad1c968/cuda-oxide/cuda-oxide-book/getting-started/installation.md)
for other build environments. The CUDA toolkit and NVIDIA driver are separate;
new PTX can require a newer driver even when native cubins work through CUDA
minor-version compatibility.

## Build and run without LibTorch

Run from `libs/infinity-v3`. Install the pinned cuda-oxide producer with its own
Rust toolchain; do not replace the host's toolchain pin. The producer needs a
CUDA toolkit with `ptxas` and `cuobjdump`, libclang development headers, and the
upstream compiler components. See the
[upstream compiler workflow](https://github.com/NVIDIA/cuda-rust/blob/999094fe568fc3080f80d49e9c44e3680ad1c968/cuda-oxide/crates/cargo-oxide/README.md).

```bash
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.1
export PATH="$CUDA_TOOLKIT_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# This script invokes the separately installed compiler and assembles native
# cubins. Python is a build-time manifest/driver tool, not a serving dependency.
CARGO_OXIDE=/path/to/cargo-oxide \
  backends/oxide/device/build.sh /opt/infinity-oxide-kernels

# Host build uses the repository toolchain and does not build LibTorch.
cargo build --locked --release -p infinity_v3-router \
  --no-default-features --features oxide,http

INFINITY_OXIDE_KERNEL_MODULE=/opt/infinity-oxide-kernels \
  INFINITY_FA4_ROOT=/path/to/infinityfa4 \
  INFINITY_OXIDE_ATTENTION=fa4 \
  ./target/release/infinity_v3 \
  --backend oxide --torch-device cuda --device-id 0 --dtype bfloat16 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 --port 8000
```

The existing `--torch-device` CLI name selects CUDA here without introducing a
Torch dependency. Runtime requirements are a compatible NVIDIA driver,
`libcublas.so.13` and its dependencies, and the prepared kernel bundle. A build
containing only `oxide` selects it by default; `--backend oxide` is explicit
when the binary contains multiple backends.

## Native kernel bundle

`device/build.sh` compiles all 36 Rust kernels separately for `sm_90`,
`sm_100`, and `sm_120`, assembles each PTX artifact to a cubin, verifies entry
symbols, and writes a manifest recording source/artifact hashes, target,
compiler revision, and commands. Every architecture is natively compiled from
Rust for all 36 exports. The original nine BERT kernels remain byte-identical
on all three targets. Additional exports provide rounded embedding/residual
LayerNorm, native width-768/1024 normalization arithmetic, ordinary or rounded RMSNorm with an optional residual output,
NeoX/interleaved RoPE with separate-product or fused arithmetic, staged or FP32
SiLU multiplication, plain or scaled word gathers, paired CLS/last-row gathers,
native width-2048 decoder RMSNorm, selected BF16-to-FP32 output extraction,
aligned vector width-2048 RMSNorm, and paired BF16 SiLU multiplication with
retained round-to-nearest and native BF16 division recipes, plus an eight-value
version with 16-byte input/output transactions.
The packed-operation exports preserve their qualified BF16 arithmetic. The
eight-value SiLU export leaves all previous 35 kernel machine
bodies unchanged on all three targets; an exhaustive finite-BF16 gate sweep
with unit up-values matches the scalar control and native operator bit for bit.
Vector RMSNorm uses 16-byte transactions on aligned buffers and a
scalar fallback otherwise. Paired SiLU handles odd widths and unaligned buffers
without reading outside a row. These are standalone primitives; they do not
expand the serving model catalog.
The row-gather and output-extraction kernels copy storage bits
or widen them exactly, support independent tensor widths and repeated output
selections, and need no additional intermediate allocation. These primitives do not
enable decoder model serving: checkpoint loading, model execution, and full
quality/performance qualification remain separate work. Generic RMS reductions
also need qualification against width-specific native reduction recipes.

The original BERT executor still needs its nine entrypoints; the older six-export
bundle fails startup. These are independent target builds;
changing a label in a manifest is not an architecture build.

| Target | Deployment family | Qualification |
| --- | --- | --- |
| `sm_90` | H100/H200 | Native compilation and H100 execution exercised; full quality/performance checks remain separate |
| `sm_100` | B200 | Native compilation/assembly; no B200 runtime qualification reported |
| `sm_120` | Blackwell RTX | Native compilation/assembly; no RTX runtime qualification reported |

`INFINITY_OXIDE_KERNEL_MODULE` accepts the bundle directory, selecting
`sm_<device capability>/kernels.cubin`, or one explicit PTX/cubin/fatbin file.
Native code avoids driver compilation at module load; PTX is compiled by the
driver and must use a supported PTX version. There is no automatic C++ kernel
substitution when an oxide module is absent.

The added arithmetic recipes pass independent numerical/guard tests on H100.
The width-2048 RMS kernels also match the unchanged native library bit-for-bit
on 16 same-input fixtures, including residual halfways, subnormals and partial
blocks. Their presence does not enable the Llama executor.
Run their GPU tests explicitly against the rebuilt bundle:

```bash
INFINITY_OXIDE_KERNEL_MODULE=/opt/infinity-oxide-kernels \
  cargo test --locked -p infinity-backend-oxide --test rounded_norms \
  -- --ignored --test-threads=1
INFINITY_OXIDE_KERNEL_MODULE=/opt/infinity-oxide-kernels \
  cargo test --locked -p infinity-backend-oxide --test decoder_primitives \
  -- --ignored --test-threads=1
```

Module entry signatures and launch contracts are defined by
[the device source](../backends/oxide/device/src/main.rs) and
[the host executor](../backends/oxide/src/lib.rs). Tensor dtype, argument order,
strides, pooling mode, launch geometry, and scratch space must match. A module
with matching names alone is not a compatible implementation.

## cuTile and existing AOT attention

The same [shared CUDA runtime](https://github.com/NVIDIA/cuda-rust/tree/999094fe568fc3080f80d49e9c44e3680ad1c968/cuda-core)
can launch compatible native modules produced by cuda-oxide, CUDA, or Rust
cuTile on the same stream. Rust cuTile normally specializes and compiles Tile IR
on first use. Shipping its cubin ahead of time requires fixed specializations
and a typed-descriptor/entry ABI adapter; arbitrary cuTile cubins are not drop-in
replacements for these entry signatures.

At the pinned revision, [cuTile's compatibility table](https://github.com/NVIDIA/cuda-rust/blob/999094fe568fc3080f80d49e9c44e3680ad1c968/cutile-rs/README.md)
requires Tile IR **13.3 for Hopper**. The current CUDA 13.1 compiler environment
does not provide that producer, so cuTile is optional and does not determine
this backend's runtime baseline.

The repository's FA4 provider has 20 compile profiles in both F16 and BF16,
using native ABI 1/export ABI 7. The optional adapter uses its raw C ABI, passing
device pointers, tensor strides, packed sequence boundaries, and the same CUDA
stream without ATen or a LibTorch model call. It selects a profile only from the
GQA ratio, head width, causality, and exact window. It validates the manifest's dtype, symbol, geometry, window, and export order at
load. Absolute head counts are runtime dimensions: the manifest counts are
compile exemplars. The pinned provider specializes on the GQA ratio and dynamic
tensor layouts, as does the existing native selector. Family names alone do not
establish compatibility. See the pinned [FA4 specialization key](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/interface.py#L1125)
and [dynamic tensor layouts](https://github.com/Dao-AILab/flash-attention/blob/e9cf2c1651d2303191eb40a739a3c135fda00999/flash_attn/cute/cute_dsl_utils.py#L131).

Existing 19-profile bundles remain valid for their original modes. The appended
causal, global d64/GQA8 profile requires its own manifest entry and capability
bit; it does not change earlier mode indices or enable a model executor.

| Export | Example Q heads | Example KV heads | Width | Causal | Window |
| --- | ---: | ---: | ---: | --- | --- |
| `bert` | 16 | 16 | 64 | No | Global |
| `modern_local` | 12 | 12 | 64 | No | Local 64/64 |
| `qwen` | 32 | 8 | 128 | Yes | Global |
| `voyage` | 16 | 8 | 128 | No | Global |
| `gqa6` | 12 | 2 | 128 | No | Global |
| `gqa8` | 32 | 4 | 128 | No | Global |
| `qwen35` | 16 | 4 | 256 | No | Global |
| `mha32` | 12 | 12 | 32 | No | Global |
| `media72` | 16 | 16 | 72 | No | Global |
| `mha128` | 6 | 6 | 128 | No | Global |
| `local128` | 6 | 6 | 128 | No | Local 64/64 |
| `causal_gqa2` | 16 | 8 | 128 | Yes | Global |
| `gqa3` | 24 | 8 | 128 | No | Global |
| `gqa4` | 32 | 8 | 128 | No | Global |
| `gqa7` | 28 | 4 | 128 | No | Global |
| `causal_local128` | 32 | 8 | 128 | Yes | Local 64/0 |
| `wide_gqa6` | 24 | 4 | 256 | No | Global |
| `causal_wide6` | 24 | 4 | 256 | Yes | Global |
| `causal_wide4` | 16 | 4 | 256 | Yes | Global |
| `causal_gqa8_d64` | 32 | 4 | 64 | Yes | Global |

Score scaling is either `1 / sqrt(head_dim)`, resolved at load, or an explicit
finite positive scalar. Score softcap, relative bias, and DeBERTa modifiers are
unsupported by the adapter. A generalized packed launch accepts separate
Q and K cumulative int32 boundaries, distinct Q/K token counts, element strides,
and the longest query sequence; the BERT wrapper uses shared boundaries. On SM100
(`capability / 10 == 10`), width-256 profiles reject separate Q/K boundaries and
require shared boundaries; other exported profiles accept them.

The pooled BF16 BERT executor supplies query heads equal to KV heads, head width
`hidden_size / num_attention_heads`, noncausal global attention, and
`1 / sqrt(head_dim)` scaling. It can use FA4 only when its ratio, width, and mask
match one of the 19 profiles (`mha32` for MiniLM, `bert` for standard width-64 MHA). For example,
12/12 and 24/24 heads can use the width-64 global MHA export compiled with
16/16 heads. Coverage of every export is operator coverage, not
new model-family execution: the pooled BF16 BERT executor remains the only model
executor, and end-to-end FP16 serving is not implemented.

When configured, this is a hybrid executor: oxide-authored Rust kernels plus
cuBLAS and AOT FA4 attention; it is not a claim that FA4 attention was rewritten
in Rust. `INFINITY_OXIDE_ATTENTION=auto` is the default. It selects FA4 when
`INFINITY_FA4_ROOT=/path/to/infinityfa4` is present and the BERT geometry
matches a compiled profile; other geometries use oxide SIMT attention. `simt`
explicitly selects the Rust attention kernel. `fa4` requires the provider and a
matching compiled profile, and rejects unsupported models instead of silently
changing implementations. The per-SM directory must contain the native ABI
1/export ABI 7 manifest, portable-host marker, matching compute capability and
an advertised mode for the selected export. A selected provider's load or launch
failure is an error rather than a retry through another attention path.
Warm up provider initialization before serving requests. Preserve its adjacent
`libcute_dsl_runtime.so`, `libtvm_ffi.so`, license files and provenance.
The provider contains a native C++ ABI shim and CuTe-generated GPU kernels;
the model executor remains Rust. Existing FA4 artifacts for `sm_90`, `sm_100`
and `sm_120` were previously compiled and audited against the current bridge
and profiles. This is separate from the fresh Rust-kernel builds. FA4 operator
runtime validation is H100 only: B200 and RTX artifacts are not runtime
qualified. The GPU tests require `INFINITY_FA4_ROOT` and a matching CUDA device,
are ignored by default, and load every export in both dtypes, compare every
export on small ragged packed batches against a CPU reference, exercise the
shared-boundary wrapper, and exercise separate Q/K boundaries on non-d256
profiles; the SM100 d256 boundary guard is covered by unit validation.
Additional H100 numerical and memory checks cover runtime head counts in both
dtypes, contiguous and padded rows, and ragged lengths 3/130. The tested
geometries include 12/12 and 24/24 heads at width 64, 32/32 heads at width 128,
and GQA with runtime 24/12, 16/4, and 6/1 head counts. This validates operator
geometry; each new model still needs checkpoint-level parity and performance.

FA4 is the target production attention provider as model coverage expands; an
FA2 compatibility backend is outside the oxide roadmap. The pinned provider
declares the 19 profile cases above spanning causal/noncausal, global/local and
MHA/GQA attention at widths 32, 64, 72, 128, and 256. The adapter now selects
every compiled profile, including runtime head-count variations, but operator
coverage is not by itself model or end-to-end qualification. Windows,
score modifications, and separate
Q/K boundaries remain explicit capabilities. Runtime head-count matching uses
validated ratios; the remaining semantics are checked independently. Keep SIMT
useful for reference and bring-up; extend FA4 exports for required model semantics where practical.

The existing GDN provider also exposes reusable AOT kernels, but GDN/Qwen3.5
model execution is not implemented by this initial backend.

## Changing-shape serving replay

On 2026-10-31, a local HTTP replay used 64 public SciFact queries and 256
documents, with natural-text substrings, batch sizes 1–32, and long/short
transitions. Reported request compute tokens ranged from 3 to 8192; sequences
were truncated to 256 tokens. The tokenizer, admission queue, batcher, backend,
and tensor response path all ran. This was public data through the serving
stack, not customer production traffic.

Three fresh processes per runtime ran in balanced order on H100 GPU 0, with
64 HTTP warmup requests and at least 10 seconds **and** 300 requests per client
count. Both oxide builds used BF16 mean pooling, FA4, the same checkpoint,
kernel bundle, and graph-free execution. The previous build was `8bd3d50`.
Values below are medians of process percentiles, not pooled-request percentiles.

| Clients | Previous p50 / p99 ms | Capacity reuse p50 / p99 ms | Previous / new requests/s |
| ---: | ---: | ---: | ---: |
| 1 | 4.362 / 11.894 | 4.166 / 11.692 | 187.0 / 190.1 |
| 8 | 10.283 / 31.387 | 9.980 / 21.667 | 519.5 / 506.5 |

The eight-client p99 process ranges were 29.49–32.20 ms before and
19.51–21.82 ms after. Throughput was roughly flat: its eight-client ranges
overlapped (502.5–537.6 versus 482.6–559.1 requests/s). This supports a
tail-latency improvement on this replay, not a general throughput increase.
The time-bounded streams shared deterministic prefixes but completed different
numbers of requests. Observed request-shape classes exceeded 1000 per process;
these are HTTP proxies, not exact native batch geometries.

Each new process recorded one arena allocation after model warmup across
29,389–31,741 native forwards, with 164,689,688 live arena bytes. All 320
canonical embeddings were bit-identical to the previous oxide build in every
process pair. Sentinel, single-versus-mixed, duplicate/order, finite-output,
and long/short-transition checks passed. All 19 FA4 profiles in both dtypes
passed H100 numerical tests and memory/initialization checking; native BERT
SIMT and FA4 changing-shape sweeps also passed both sanitizers.

An unchanged Torch 2.14.1 reference with graphs enabled measured p50/p99
4.839/18.167 ms at one client and 11.066/24.208 ms at eight. It used the same
checkpoint and the previously qualified native library, but ran through a
private Docker bridge while oxide used loopback. These reference latencies
include that networking difference and do not qualify the latest LibTorch
version. Minimum embedding cosine versus this reference was 0.999952;
recall@10 and every top-1 result matched on the small fixture.
A separate short diagnostic confirmed graph use: two captures, 731 hits, and
1720 misses across warmup, quality, replay, and correctness phases. These
aggregate counters do not isolate replay-phase reuse.

Batch fragmentation remains relevant: one new-process replay submitted
112,691 sequences in 6927 HTTP requests but executed 30,392 native forwards,
or 3.71 sequences per native batch. Inputs tokenize and queue independently.
Partial-client dispatch is intentional and has admission-deadlock regression
coverage; waiting for every sibling is not a safe general replacement.
Aggregate counts do not isolate tokenization from backend scheduling costs.

## Historical MiniLM results before graph-free execution

The results below describe commit `e78c3b3`, before the tensor, loading and
graph-free execution changes. They do not establish the performance of the
current stream-based backend; qualify it separately against the best LibTorch
configuration.

On H100, `sentence-transformers/all-MiniLM-L6-v2` with BF16 and mean pooling
was faster through the hybrid FA4 path. These are HTTP measurements from the
same release binary, checkpoint, GPU and response encoding, with CUDA graphs
enabled for **both** backends. Each backend had four fresh process windows per
point, each with 100 warmup requests and at least 300 measured requests **and**
five seconds of measurement. Intervals resample process windows, not individual
requests. One client sent requests sequentially.

| Tokens / batch | LibTorch p50 ms | Oxide p50 ms | Throughput change | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| 128 / 1 | 0.992 | 0.910 | +8.09% | +3.52% to +11.82% |
| 128 / 8 | 1.251 | 1.196 | +4.82% | +2.57% to +6.84% |
| 256 / 1 | 1.016 | 0.929 | +9.63% | +7.39% to +12.02% |
| Variable length, up to 256 / 8 | 1.303 | 1.244 | +4.50% | +0.44% to +8.36% |

Latency and throughput are measured separately; throughput percentages are not
ratios of p50 latency. The shared GPU held an idle resident service; its jobs and
clocks were unchanged. The LibTorch baseline reused the existing 2.14 native
library after verifying 16 relevant BERT/kernel source blobs matched main
`c95009f`. Both selected backends used the same release executable.

Changing shapes were also measured with four fresh processes per runtime,
64 warmup requests per case, and at least 300 measured requests and five seconds
per case. Token IDs changed between cycles. This table compares the final
bucketed backend with LibTorch and the original single-workspace oxide build.

| Changing-shape cycle | LibTorch p50 ms | Oxide p50 ms | Throughput vs LibTorch | 95% interval | Throughput vs original oxide |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 single-input lengths | 1.310 | 0.901 | +45.07% | +36.53% to +51.66% | +51.66% |
| 32 single-input lengths (8–256 tokens) | 1.306 | 0.898 | +46.07% | +40.10% to +52.27% | +53.40% |
| 4 length patterns, batch 8 | 1.468 | 1.164 | +28.54% | +18.93% to +38.72% | +86.36% |

These are finite repeating geometry sets after warmup, not a claim about every
arrival pattern, cold start or model. The 32-length cycle exercises reuse within
buckets and exceeds the eight-entry cache if exact lengths are used. Pure Rust
SIMT attention was slower than LibTorch in the initial pilot; these throughput
results use hybrid FA4 attention.

For 64 SciFact queries and 256 documents (320 raw embedding vectors), minimum
cosine similarity to LibTorch was **0.999952** and maximum relative L2 error was
**1.02%**. Recall@10 and top-1 accuracy matched; nDCG@10 changed from 0.86204 to
0.86301, with a paired-query 95% delta interval spanning zero. Reusing either
backend's queries with the other's document index preserved recall and top-1
on this fixture. The vectors are not bit-identical, and this small fixture is
not a full MTEB evaluation or a general accuracy guarantee.

## Extending model coverage

The compatibility floor is every model family and task currently supported by
Infinity's LibTorch backend, including its checkpoint aliases, module chains,
precision policies, and state/media behavior. Pooled BF16 BERT is the first
vertical, not the intended coverage limit. Broader model support can extend this
catalog; a family name alone is not evidence that all its variants work.
Keep this catalog synchronized as LibTorch gains families and tasks.

| Coverage track | Families and contracts to port |
| --- | --- |
| Encoders | BERT/RoBERTa/XLM-R/CamemBERT, DistilBERT, ModernBERT, GTE, Nomic, Jina/JinaCode, MPNet and DeBERTa |
| Dense decoders | Llama, Mistral/Ministral, Qwen2/Qwen3 and UltraBERT variants |
| Hybrid and MoE | Qwen3.5 recurrent/convolution history, Qwen3/Qwen3.5/Gemma4 experts, routing and prefix/state reuse |
| Gemma | Gemma3, Gemma4 and EmbeddingGemma2 normalization, position, shared-KV and readout variants |
| Media and multi-vector | Qwen-VL, Pixtral/Mistral3, UltraVBERT/SigLIP and Gemma towers, where their checkpoints support image/video/audio; ColBERT/ColQwen wrappers |
| Task heads | Pooled/raw embeddings, Dense/projection chains, classification/NER, SPLADE, reranking, multi-vector and learned decision readouts |

Keep tensor storage, layout, accumulation, and quantization contracts independent
of the BERT recipe. The next shared operations are LayerNorm/RMSNorm, positional
transforms, attention masks/windows/GQA, activation variants, gated MLPs, reductions, gather/
scatter and task readouts. Add recurrent state, expert routing and media
operators with their first family. Methods and provider capabilities are
resolved at load/workspace preparation; forward uses concrete operations and
borrowed views without a dynamic operator registry on each call.

Track qualification by family, task, precision, device and provider, using exact
checkpoint fixtures and LibTorch parity, quality, latency/throughput, memory and
failure checks. Retiring a model is a separate product decision and does not
count as implementing compatibility. GPU migration can proceed per family;
LibTorch CPU remains during the transition until the CPU strategy is resolved.
Full LibTorch replacement requires resolving that CPU dependency as well.

Preserve checkpoint, pooling, HTTP, precision, and error-isolation contracts as
coverage grows:

1. Establish BERT quality and repeated throughput/latency controls against
   LibTorch, including graph-enabled controls and variable sequence lengths.
2. Refine arena reuse, asynchronous completion, BF16/cuBLASLt projection
   dispatch, and matrix preparation using measured bottlenecks.
3. Qualify the hybrid AOT attention path where its measured benefit warrants
   the additional provider. Avoid a device-to-host copy between kernels.
4. Add Qwen RMSNorm, RoPE, grouped-query attention, gated MLPs and readouts.
5. Integrate GDN and Qwen3.5/D3 state plus typed-decision semantics.
6. Validate MoE, multimodal inputs, and remaining typed heads independently.

NVIDIA describes cuda-oxide as experimental. Pin compiler and runtime changes
and revalidate emitted artifacts. NVIDIA CUDA Rust remains **Apache-2.0** with
its [required notices](https://github.com/NVIDIA/cuda-rust/blob/999094fe568fc3080f80d49e9c44e3680ad1c968/LICENSE);
Infinity's license does not replace third-party grants. Reused FA4/GDN sources
and binaries retain their own licenses and provenance.
