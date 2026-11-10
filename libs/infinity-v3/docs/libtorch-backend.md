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
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cuda --dtype float16 --port 8000
```

Choose a version from the [container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3).
Use `--device-id 0` for one GPU; otherwise each visible GPU runs a complete replica.
For native LibTorch CPU builds, use `--torch-device cpu --dtype float32`.
CPU builds from current `main` use LibTorch. Dynamic FP8 and CUDA-only kernels
require the NVIDIA image.

## Optional quantization

Use `--dynamic-quantization` to select a format for each model:

```bash
infinity_v3 --model-id /path/to/model \
  --torch-device cuda --dtype bfloat16 --dynamic-quantization fp8
```

`none` preserves the model dtype. `fp8` selects rowwise FP8; `mxfp8` and `nvfp4`
select the corresponding block-scaled MLP format on eligible Blackwell BF16
models. In all-linear scope, attention projections still use FP8. Unsupported
format/model combinations fail during initialization rather than selecting a
different format. Quantization can change outputs; check accuracy on your task.

Use `--dynamic-quantization-scope mlp` to restrict quantization to MLP projections.
The default `auto` uses all-linear scope for eligible dense decoders and MLP-only
scope otherwise. See [model coverage](libtorch-model-coverage.md) and
[per-model configuration](cli.md#quantization).

Portable FP8 projections fuse output scaling and conversion into one kernel.
They retain the two separately rounded FP32 scale products and the final
model-dtype rounding, including subnormal values. Aligned matrices use vector
loads and stores; other contiguous layouts use the same arithmetic with scalar
accesses. The existing Hopper row-scaled projection policy is unchanged.

## InfinityFA4

The NVIDIA image enables FlashAttention-4 for supported attention shapes and
uses existing attention kernels otherwise. To disable it for troubleshooting,
set `INFINITY_TORCH_FA4=0` before starting the server.

## OneJev block attention

For compatible decision checkpoints, use `--decision-protocol onejev` with
`--dtype bfloat16`. The checkpoint configuration selects block attention;
shared prefixes are reused within a backend batch. This is not a persistent
cross-request cache.

For dense Qwen3.5, `--dynamic-quantization fp8` uses all-linear dynamic FP8 by
default and enables the FP8 optimizations automatically. GDN activations remain
BF16 and stored recurrent states remain FP32.

Dense FP8 MLPs combine portable projection scaling, rounded SwiGLU and row
quantization for aligned intermediate widths up to 20480. The fused SwiGLU path
keeps the two FP32 scale products and model-dtype rounding of the separate
operations; wider rows, projection bias or clipping retain their existing
recipes for that gated form. Gated rounded GELU uses the same fused boundary,
including reversed gate/up layouts, and follows the native gated kernel's selected
layout recipe: aligned vector rows use the native half-pair HFMA GELU staging,
while ragged or unaligned rows retain its scalar separate multiply/add staging.
This is distinct from the unary encoder GELU recipe. Legacy decoder SwiGLU keeps
its frozen scalar bytes. Unary FP8 MLPs can fuse ReLU, Candle-rounded SiLU, or
encoder-rounded GELU with the projection, an optional separate bias, and row quantization,
including for Hopper's rowwise projection. Custom rounded recipes opt in
explicitly; standard erf/tanh GELU, ordinary SiLU, packed gate/up activation
without an explicit rounded recipe, clipping, and casts keep the composed path.
Qwen3.5 post-attention normalization can also produce the dense MLP's FP8 input
directly, omitting its unused BF16 copy.

Cached GDN suffixes use the existing tensor-core prefill provider automatically
whenever an immutable FP32 prefix state is available. Batch size and per-row
length do not select a different arithmetic path. Prefix capture, native
checkpoint resumption, and unsupported provider capabilities retain native
recurrent execution. BF16 normalized Q/K and chunk arithmetic can change
probabilities and selected answers, including with FP8 enabled. The GDN provider and native
backend must be rebuilt together for provider ABI 2; old provider bundles are
rejected during configuration.

Send requests to `/v1/systemone`. See [decision inputs](cli.md#structured-decision-inputs)
for the request schema, image content parts and input limits.

## Checkpoint loading

Local and Hugging Face checkpoints are supported. Keep the model cache on a
persistent volume to avoid repeat downloads. Startup logs report device checks,
checkpoint loading, warmup and readiness for each replica.

The default reader needs no additional container capabilities. For repeated
loads from a managed model-cache mount, `INFINITY_TORCH_WEIGHT_READER_DIRECT_IO=0`
allows filesystem page-cache reads. See [startup measurements](modernbert-startup.md)
for the effect of local storage and cache state.

### GPU kernel cache

Set `INFINITY_KERNEL_CACHE_PREFIX=/mnt/distributed` to default both
`CUDA_CACHE_PATH` and `PYTORCH_KERNEL_CACHE_PATH` to that existing, writable
directory. Explicit cache paths take precedence.

Setup only sets the environment once per process, before CUDA and Tokio start;
it performs no GPU discovery or cache-directory access. Explicit CPU and Python
backends skip it. CUDA manages driver compatibility, and PyTorch's kernel cache
keys include the compilation architecture and NVRTC version. Use a filesystem
with atomic rename support. This redirects native JIT caches; startup savings
depend on whether the workload compiles kernels at runtime.

## Build from source

Run from `libs/infinity-v3` with LibTorch 2.14.1, CMake 3.25+, a C++20 compiler
and, for CUDA builds, a compatible CUDA toolkit:

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo build --release -p infinity_v3-router --no-default-features --features libtorch,http
```

See the [contributor guide](../CONTRIBUTING.md) for repository setup and
[licensing](../LICENSING.md) for required notices when distributing a build.
Detailed kernel experiments and test recipes remain in the
[archived developer notes](https://github.com/michaelfeil/infinity/blob/87bb80308e64b284d60c41b234b766032ed49866/libs/infinity-v3/docs/libtorch-backend.md).
