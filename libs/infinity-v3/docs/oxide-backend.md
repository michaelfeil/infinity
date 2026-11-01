<!-- Copyright Infinity Core Contributors. Licensed under the repository LICENSE. -->

# Oxide CUDA backend

The experimental `oxide` backend executes typed encoder and dense decoder
backbones using GPU kernels authored in Rust and compiled by NVIDIA's
cuda-oxide compiler, cuBLAS/cuBLASLt BF16 projections, and FA4 AOT attention.
Its serving process uses NVIDIA's `cuda-core` runtime and has no LibTorch,
PyTorch, Python, or CUDA C++ model execution dependency. Encoder attention can
also use the Rust SIMT kernel; decoder attention requires FA4.

Implementation coverage and checkpoint qualification are separate. The current
catalog below covers pooled BF16 embeddings, with explicit configuration
guards. It does not establish compatibility or better performance for every
variant of these families.

## Current behavior

- BF16 compute with FP32 accumulation for projections and FP32 embedding outputs.
- Single-file or standard sharded safetensors; F32/F16/BF16 checkpoint weights.
  Required shapes, finite values, shard paths, and model options are validated.
  Standard Hugging Face cache snapshot symlinks to the same repository's flat
  `blobs/<SHA1-or-SHA256>` directory are supported. Other external weight paths
  and lexical shard traversal are rejected.
- Packed attention and CLS, mean, or last-token pooled embeddings.
- Existing HTTP endpoints, tokenizer workers, batching interface, response
  encodings, and request-controlled normalization. When `modules.json` is present,
  only a Transformer/Pooling chain with an optional final Normalize is accepted.
  Trained Dense projections, other module chains, raw/token embeddings, classification,
  SPLADE, quantized weights, and non-BF16 serving fail explicitly.

| Typed executor | Accepted configuration |
| --- | --- |
| BERT, RoBERTa, XLM-RoBERTa, CamemBERT | Encoder-only absolute position tables, tanh GELU, no cross-attention. BERT position capacity is at most 4096. Head width is at most 256; a matching provider profile is required for FA4. |
| DistilBERT | The same packed encoder contract, with its checkpoint names/config aliases and fixed LayerNorm epsilon `1e-12`. An optional stored token-type table is validated and preserved; otherwise a zero table is used. |
| Qwen2 | Dense SiLU text backbone, full split-half NeoX RoPE, global causal or noncausal attention, and a matching FA4 profile. The H896/d64 native RMS/rotary recipe is rejected. The exercised GTE-Qwen2-1.5B-instruct geometry is H1536, 12 Q/2 KV heads, d128, noncausal. |
| Qwen3 | Dense SiLU text backbones at H1024 or H4096, d128 learned Q/K head norms, full NeoX RoPE, and a matching FA4 profile. Query projection width may differ from hidden width. Qwen3-Embedding-0.6B and 8B are exercised; expert models and other normalization widths are rejected. |
| Llama | Hidden width 2048, dense SiLU, base unscaled `default` RoPE, causal global attention, and a matching FA4 profile. TinyLlama's 32 Q/4 KV heads at d64 need the appended twentieth profile. H4096, noncausal Llama, and scaled RoPE including Llama3 are rejected. |

Decoder loading also rejects sliding windows, partial rotary dimensions, expert,
recurrent, media and quantization modifiers. Accepted Model/ForCausalLM checkpoint
backbones provide embeddings only; they do not enable generation or a KV cache.
Required tensor shapes and complete Q/K/V and gate/up bias groups are checked.
Decoder sentence-transformers pooling metadata must select last-token only;
the executor can also use explicit CLS or mean pooling selections.
Family names alone do not bypass these guards. Arithmetic and concrete methods
are selected when loading the model: encoder normalization recipes preserve the
native width/family behavior, and Llama uses the native width-2048 RMS and fused
RoPE recipe while Qwen2 retains its separate arithmetic. Qwen3 uses the native
width-1024 four-row RMS recipe or ordinary width-4096 RMS, followed by fused
Q/K head normalization and BF16 rotary arithmetic.

The Rust kernels fuse embedding normalization and QKV preparation; register-cached
384-wide normalization specializations and a parallel mean-pool kernel cover the
MiniLM geometry. Encoder and decoder executors share one ordered-stream engine,
at most eight GPU workspaces, and two owned pinned input/output slots. One GPU
workspace is active and up to seven are cached. Each capacity dimension
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
covering active and cached GPU arenas plus both pinned slots. Model weights,
immutable RoPE tables and driver/provider metadata sit outside that budget.
An oversized GPU arena empties the inactive cache but remains available to
ordered submissions with the same token capacity and sufficient boundary and
selection capacities. A different token capacity releases that arena rather
than caching it. Oversized host staging requires an exclusive lease. Idle pinned
slots are reclaimed before growth; exceptional pinned allocations are released
after completion.

`submit_embed` returns a deferred result whose lease owns its pinned buffers and
completion event. Up to two results may remain outstanding and complete in
either order while GPU work remains ordered on one stream. Completing or dropping
a result drains its work before returning the slot; a third submission waits for
a slot. Shared GPU scratch is never used as the deferred result's storage.

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
PDL is not implemented.

Matrix preflight uses typed `ShapeError::Rejected` and `ShapeError::Fatal`
outcomes. A rejected geometry or unsupported tactic returns an inference error
before request GPU work, restores affected descriptors or invalidates their row
cache, and leaves the engine usable for the next supported shape. Invalid packed
IDs/boundaries and SIMT sequence limits are also checked before dispatch. A fatal
provider/descriptor failure or uncertain execution completion makes the engine
unhealthy and drains or retains owned resources safely; it is not silently
retried through another math method.

## Tensor and matrix foundations

`tensor` provides checked fixed-rank layouts and borrowed `TensorRef`/`TensorMut`
views. Shape, stride, span, offset, and storage arithmetic are checked; mutable
views reject overlapping or broadcast writes. Views allocate nothing, clone no
owners, and make no CUDA calls. They describe a borrow rather than ownership of
unfinished GPU work; enqueue boundaries remain unsafe and the engine retains
all buffers until completion. There is no dynamic operator dispatcher.

`math::LinearPlan` prepares row-major `X W^T` geometry and cuBLAS parameters once
per workspace, including optional row padding. BF16, FP16 and FP32 matrix methods
accumulate in FP32; serving executors still run only in BF16. The backend binds its
context once per forward, rather than once per matrix operation. FP8/FP4 enum
variants describe storage sizes only and do not enable arithmetic or quantized
checkpoint loading. Packed scales and quantization schemes need separate methods.

The regular cuBLAS handle owns a 32 MiB workspace, initialized once before the
backend becomes ready. This avoids uninitialized split-K scratch reads observed
with cuBLAS 13.2; matrix dispatch and algorithm selection are unchanged. The
allocation is included in live device-byte accounting and retained if teardown
cannot establish that GPU use has finished.

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
`libcublas.so.13`, `libcublasLt.so.13`, their dependencies, and the prepared kernel bundle. A build
containing only `oxide` selects it by default; `--backend oxide` is explicit
when the binary contains multiple backends.

## Native kernel bundle

`device/build.sh` compiles all 45 Rust kernels separately for `sm_90`,
`sm_100`, and `sm_120`, assembles each PTX artifact to a cubin, verifies entry
symbols, and writes a manifest recording source/artifact hashes, target,
compiler revision, and commands. Every architecture is natively compiled from
Rust for all 45 exports. The original nine BERT kernels remain byte-identical
on all three targets. Additional exports provide rounded embedding/residual
LayerNorm, native width-768/1024 normalization arithmetic, ordinary or rounded RMSNorm with an optional residual output,
NeoX/interleaved RoPE with separate-product or fused arithmetic, staged or FP32
SiLU multiplication, plain or scaled word gathers, paired CLS/last-row gathers,
native width-2048 decoder RMSNorm, selected BF16-to-FP32 output extraction,
aligned vector width-2048 RMSNorm, and paired/vector8 BF16 SiLU multiplication
with retained round-to-nearest and native BF16 division recipes.
The packed-operation exports preserve their qualified BF16 arithmetic. The
additional native-division SiLU export leaves all previous 34 kernel machine
bodies unchanged on all three targets; an exhaustive finite-BF16 gate sweep
with unit up-values matches the scalar control and native operator bit for bit.
The additional `infinity_silu_mul_bf16_vector8_native` export processes eight
values per thread with 16-byte aligned transactions and leaves all previous 35
machine bodies unchanged. Both decoder recipes select it at load time, using
`ceil(rows * ceil(width / 8) / 256)` blocks of 256 threads for full and reduced
readout rows. Odd widths, unaligned pointers and tail chunks fall back safely
within each row. This changes the selected SiLU export and launch geometry;
projection methods and the qualified regular-cuBLAS path remain unchanged.
The exact width-768 embedding and residual LayerNorm recipes also select
16-byte vector transfers at load time, preserving their four-row CTA geometry,
statistics, rounding and affine arithmetic. Each required or non-null optional
BF16 pointer must be aligned; otherwise the unchanged scalar implementation
runs. All previous 36 machine bodies remain byte-identical on all three targets.
Width-1536 Qwen2 RMSNorm retains six lane values in registers through the
reduction, using the same 256-thread row geometry, FP32 accumulation, rounded
division and BF16 output boundaries. Other Qwen2 widths keep the generic
exports. These two cached exports leave all previous 38 machine bodies unchanged
on all three targets.
Vector RMSNorm likewise uses 16-byte transactions on aligned buffers and a
scalar fallback otherwise. Export availability alone does not expand the serving
model catalog.
Three Qwen3 exports add width-1024 ordinary/residual RMS and fused d128 Q/K
normalization with NeoX RoPE. The latter preserves the native 16-lane centered
statistics and BF16 multiply/FMA boundaries. All previous 40 machine bodies
remain byte-identical on all three targets; all five new primitive GPU checks
match independent native controls and pass memory and initialization checks.
Two width-1536 RMS exports use aligned 16-byte transfers and shared storage
to preserve the scalar lane reduction order. Unaligned buffers retain the
cached scalar recipe. These replace only the prepared H1536 function bindings;
all previous 43 bodies and resource requirements remain unchanged on all three
targets. Native/cached bit controls, aliases, bounds and memory checks pass.
The row-gather and output-extraction kernels copy storage bits
or widen them exactly, support independent tensor widths and repeated output
selections, and need no additional intermediate allocation. Checkpoint loading,
executor guards and full quality/performance qualification remain separate
contracts. Generic RMS reductions need qualification against width-specific
native reduction recipes.

MiniLM still needs the original nine entrypoints; the older six-export bundle
fails startup. Wider encoders require their selected normalization exports, and
the current decoder selector requires the new vector8 entrypoint. Use the current
40-export bundle for the full typed catalog. These are independent target builds;
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
blocks. The guarded H2048 Llama executor uses their vector variants.
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
bit; it does not change earlier mode indices. TinyLlama's causal d64/GQA8
executor requires this 20-profile bundle.

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

The pooled BF16 encoder executor supplies query heads equal to KV heads, head width
`hidden_size / num_attention_heads`, noncausal global attention, and
`1 / sqrt(head_dim)` scaling. It can use FA4 only when its ratio, width, and mask
match a profile (`mha32` for MiniLM, `bert` for standard width-64 MHA). For example,
12/12 and 24/24 heads can use the width-64 global MHA export compiled with
16/16 heads. Decoder executors supply their validated Q/KV ratio, head width and
causality with global attention. Coverage of an export is operator coverage;
each executor and checkpoint still needs separate qualification. End-to-end
FP16 serving is not implemented.

When configured, this is a hybrid executor: oxide-authored Rust kernels plus
cuBLAS and AOT FA4 attention; it is not a claim that FA4 attention was rewritten
in Rust. `INFINITY_OXIDE_ATTENTION=auto` is the default. It selects FA4 when
`INFINITY_FA4_ROOT=/path/to/infinityfa4` is present and the encoder geometry and
packed row alignment match a compiled profile; other encoder geometries use
oxide SIMT attention. `simt` explicitly selects the Rust attention kernel.
`fa4` requires the provider and a
matching compiled profile, and rejects unsupported models instead of silently
changing implementations. These modes configure encoder attention; decoders
always require the matching FA4 provider. The per-SM directory must contain the
native ABI 1/export ABI 7 manifest, portable-host marker, matching compute capability and
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

FA4 is the AOT attention provider; FA2 is not implemented. The pinned provider
declares the 20 profile cases above spanning causal/noncausal, global/local and
MHA/GQA attention at widths 32, 64, 72, 128, and 256. The adapter now selects
every compiled profile, including runtime head-count variations, but operator
coverage is not by itself model or end-to-end qualification. Windows,
score modifications, and separate Q/K boundaries remain explicit capabilities.
Runtime head-count matching uses
validated ratios; the remaining semantics are checked independently.

The existing GDN provider also exposes reusable AOT kernels, but GDN/Qwen3.5
model execution is not implemented by this initial backend.

## Current qualification limits

The current framework passed HTTP quality checks on 320 public SciFact texts
for each of seven checkpoint cases, including changing shapes, order checks and
sentinels. These are checkpoint cases, not a full MTEB or model-catalog claim.

The later H1536 vector RMS control preserves all 320 GTE-Qwen2 vectors bit for
bit and improves median latency from 2.408 to 2.313 ms (-3.94%), with p99 down
3.93%. Short one/seven-token diagnostics also improve. Profiling attributes the
gain to normalization, with no additional dispatch calls or graphs. This GTE
control remains 13.06% slower than matched native LibTorch at 128 tokens.

Qwen3-Embedding-0.6B and 8B additionally pass 320 public-text comparisons each
with the same numerical guards, shape changes, deferred A/B/A and cancellation,
invalid-request recovery, and GPU memory/initialization checks. Their first
balanced H100 controls below use one 128-token sequence and the same native
graph/vendor settings. Positive latency changes mean slower than LibTorch:

| Qwen3 checkpoint | LibTorch p50 ms | Oxide p50 ms | p50 change | p99 change |
| --- | ---: | ---: | ---: | ---: |
| Embedding-0.6B | 1.407 | 1.623 | +15.32% | +14.56% |
| Embedding-8B | 6.388 | 7.073 | +10.72% | +9.21% |

These Qwen3 paths expand functional coverage; their current performance still
misses the speed target. Other Qwen3 widths, tasks and dtypes remain unqualified.

The fixed-shape controls below use one 128-token sequence, BF16, the documented
pool, H100 and eight CPU cores. Each backend has two fresh-process windows of
2,000 timed calls after 20 warmups, in balanced order. LibTorch uses verified
graph replay; Oxide uses ordinary streams. Both use the same cuBLAS 13.2
libraries and twenty-profile FA4 provider. Native timing excludes Rust response
packaging; Oxide timing includes its backend result map. Values are medians
across each backend's two windows; negative changes mean lower Oxide latency.

| Checkpoint | Oxide p50 µs | LibTorch p50 µs | p50 change | p99 change |
| --- | ---: | ---: | ---: | ---: |
| MiniLM-L6, mean | 233.8 | 262.3 | -10.9% | +27.3% |
| MS MARCO DistilBERT, mean | 291.9 | 328.6 | -11.1% | -14.5% |
| All-DistilRoBERTa, mean | 271.6 | 328.2 | -17.2% | -20.7% |
| CamemBERT-base, mean | 531.9 | 549.1 | -3.1% | -1.8% |
| BGE-M3, CLS | 1155.6 | 1078.6 | +7.1% | +9.3% |
| GTE-Qwen2-1.5B, last token | 2429.3 | 2046.7 | +18.7% | +18.1% |
| TinyLlama-1.1B, last token | 1905.6 | 1771.0 | +7.6% | +63.9% |

Performance is mixed. Tail latency varied on the shared host, particularly for
MiniLM and TinyLlama; a separate old/new Oxide control could not isolate a fixed
host-dispatch regression. The H768 and GTE rows use subsequent balanced controls
that also bracketed the candidate with the previous Oxide build. Vector
LayerNorm reduced the RoBERTa and CamemBERT medians by 7.9% and 6.1%, while all 640
HTTP vectors remained bit-identical to that build. CamemBERT's diagnostic
normalization time fell from about 112 to 74 µs per forward.
The GTE cache reduced its median by 0.9% and normalization GPU time from about
248 to 220 µs; its 320 HTTP vectors match both previous Oxide and native outputs
bit for bit. GTE still trails native latency. The combined 40-export build
preserves all 960 previous Oxide HTTP vectors bit for bit and passes the
operator memory checks.
The vector8 SiLU kernel reduced its summed GPU time
from about 77 to 55 µs across TinyLlama's 22 layers in a diagnostic trace, which
does not establish a total-latency win. Quality, tail latency and throughput
remain separate gates for each checkpoint, batch pattern, precision, device and
provider.

The measurements below describe earlier builds and controls; they are retained
as historical evidence, not performance claims for every current executor.

## Earlier changing-shape serving replay

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

## Remaining coverage limits

The typed catalog above does not replace the full LibTorch catalog. ModernBERT,
MPNet, DeBERTa, other decoder families, scaled/partial/windowed position and
attention recipes, recurrent/MoE state, media inputs, learned projections and
task heads remain unsupported. FA4 local/window profiles and standalone device
primitives do not enable those model contracts. CPU execution is also outside
this CUDA backend. See [LibTorch model coverage](libtorch-model-coverage.md) for
the native compatibility inventory.

Methods and provider capabilities are resolved at model/workspace preparation;
forward uses concrete typed operations and borrowed views without a dynamic
operator registry. New coverage requires exact checkpoint fixtures plus native
parity, quality, memory, failure-isolation and balanced performance checks. No
all-models-faster or full LibTorch replacement claim is made.

NVIDIA describes cuda-oxide as experimental. Pin compiler and runtime changes
and revalidate emitted artifacts. NVIDIA CUDA Rust remains **Apache-2.0** with
its [required notices](https://github.com/NVIDIA/cuda-rust/blob/999094fe568fc3080f80d49e9c44e3680ad1c968/LICENSE);
Infinity's license does not replace third-party grants. Reused FA4/GDN sources
and binaries retain their own licenses and provenance.
