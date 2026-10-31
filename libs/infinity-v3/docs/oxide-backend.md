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
MiniLM geometry. The model executor retains at most eight geometry workspaces:
one active plus up to seven inactive device and pinned-host arenas. The reuse key
rounds token count and maximum sequence length up to 32-token boundaries while
keeping sequence-boundary and pooled-selection counts exact. Actual sequence
lengths and selected indices remain authoritative; only unused token capacity is
zero-padded. Inactive workspaces are evicted LRU under the count limit and a
256 MiB budget covering active plus cached arena buffers. Model weights and
driver graph metadata sit outside that budget. One active request may exceed the
budget; doing so empties the inactive cache.

For a warmed workspace, capture includes H2D upload, kernel/cuBLAS work, and D2H
download around stable pinned staging addresses. Replay is one graph enqueue
followed by a final stream synchronization. An uncached geometry still requires
preparation and warmup. This reuse remains experimental and does not promise
gains for all models or traffic patterns.

`INFINITY_OXIDE_CUDA_GRAPHS=0` disables graph capture for controls and debugging;
`1` enables it and is the default. Kernel modules, captured allocations, the
primary context, and streams remain owned until outstanding work drains,
including on errors. Without an explicitly selected AOT provider, attention uses
the Rust SIMT kernel. The pure oxide and hybrid AOT configurations must be
measured separately.

## Separate compiler and serving toolchains

NVIDIA CUDA Rust is pinned at
[`999094fe568fc3080f80d49e9c44e3680ad1c968`](https://github.com/NVIDIA/cuda-rust/tree/999094fe568fc3080f80d49e9c44e3680ad1c968).
The Infinity host builds with the repository's Rust **1.93.1** toolchain;
`cuda-core` requires Rust 1.89 or newer. The separate device package pins
**1.99.1**, `rustc-dev`, Rust sources, and LLVM tools for cuda-oxide and the
nine-kernel bundle. Serving does not load the Rust compiler or compile kernels on
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

`device/build.sh` compiles all nine Rust kernels separately for `sm_90`,
`sm_100`, and `sm_120`, assembles each PTX artifact to a cubin, verifies entry
symbols, and writes a manifest recording source/artifact hashes, target,
compiler revision, and commands. Every architecture is natively compiled from
Rust for all nine exports. The fresh set adds register-cached 384-wide
embedding/LayerNorm kernels and a parallel mean-pool kernel; the earlier bundle
exported six. A bundle with these required new entrypoints must be rebuilt, and
the old six-export bundle fails startup. These are independent target builds;
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

The repository's FA4 provider has noncausal BF16/FP16 exports with 12 heads of width 32
matching MiniLM. The optional adapter uses its raw C ABI, passing device pointers,
tensor strides, packed sequence boundaries, and the same CUDA stream without
ATen or a LibTorch model call. When configured, this is a hybrid executor:
oxide-authored Rust kernels plus cuBLAS and AOT FA4 attention; it is not a claim
that FA4 attention was rewritten in Rust.

`INFINITY_OXIDE_ATTENTION=auto` is the default. It selects FA4 when
`INFINITY_FA4_ROOT=/path/to/infinityfa4` is present and the model has 12 heads of
width 32; other geometries use oxide SIMT attention. `simt` explicitly selects
the Rust attention kernel. `fa4` requires the provider and qualified geometry,
and rejects unsupported models instead of silently changing implementations.
The per-SM directory must contain the native ABI 1/export ABI 7 manifest,
portable-host marker, matching compute capability and advertised mode 7. A
selected provider's load or launch failure is an error rather than a retry
through another attention path.
Warm up provider initialization before graph capture. Preserve its adjacent
`libcute_dsl_runtime.so`, `libtvm_ffi.so`, license files and provenance.
The provider contains a native C++ ABI shim and CuTe-generated GPU kernels;
the model executor remains Rust. Existing FA4 artifacts for `sm_90`, `sm_100`
and `sm_120` were previously compiled and audited against the current bridge
and profiles. This is separate from the fresh Rust-kernel builds, and does not
establish runtime qualification on B200 or RTX hardware.

The existing GDN provider also exposes reusable AOT kernels, but GDN/Qwen3.5
model execution is not implemented by this initial backend.

## Measured MiniLM results

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

Preserve checkpoint, pooling, HTTP, precision, and error-isolation contracts as
coverage grows:

1. Establish BERT quality and repeated throughput/latency controls against
   LibTorch, including graph-enabled controls and variable sequence lengths.
2. Refine arena reuse, asynchronous completion, BF16/cuBLASLt projection
   dispatch, and graph preparation using measured bottlenecks.
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
