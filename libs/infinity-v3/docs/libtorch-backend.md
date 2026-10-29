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
The published `cpu-*` image uses Candle rather than LibTorch; its model coverage
is separate from the native LibTorch catalog.

## Optional dynamic FP8

Add `--enable-fp8-dynamic` to enable FP8 projections on supported models and GPUs:

```bash
infinity_v3 --model-id /path/to/model \
  --torch-device cuda --dtype bfloat16 --enable-fp8-dynamic
```

FP8 is opt-in and can change outputs; check accuracy on your task before enabling
it. Model and hardware support varies. Use `INFINITY_TORCH_FP8_SCOPE=mlp` to
restrict eligible models to MLP projections. See [model coverage](libtorch-model-coverage.md).

## InfinityFA4

The NVIDIA image enables FlashAttention-4 for supported attention shapes and
uses existing attention kernels otherwise. To disable it for troubleshooting,
set `INFINITY_TORCH_FA4=0` before starting the server.

## OneJev block attention

For compatible decision checkpoints, use `--decision-protocol onejev` with
`--dtype bfloat16`. The checkpoint configuration selects block attention;
shared prefixes are reused within a backend batch. This is not a persistent
cross-request cache.

For dense Qwen3.5, `--enable-fp8-dynamic` uses all-linear dynamic FP8 by
default and enables the FP8 optimizations automatically. GDN activations remain
BF16 and stored recurrent states remain FP32.

`INFINITY_TORCH_GDN_SUFFIX_PREFILL=1` separately opts cached GDN suffixes into
the existing tensor-core prefill kernels. Batch geometry is selected automatically.
This setting is off by default because BF16 normalized Q/K and chunk arithmetic
can change probabilities and selected answers, including with FP8 enabled.
Validate those changes on your workload before enabling it. The GDN provider and native
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

## Build from source

Run from `libs/infinity-v3` with LibTorch 2.15.0, CMake 3.25+, a C++20 compiler
and, for CUDA builds, CUDA 13.2 (matching the PyTorch CUDA distribution):

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo build --release -p infinity_v3-router --no-default-features --features libtorch,http
```

See the [contributor guide](../CONTRIBUTING.md) for repository setup and
[licensing](../LICENSING.md) for required notices when distributing a build.
Detailed kernel experiments and test recipes remain in the
[archived developer notes](https://github.com/michaelfeil/infinity/blob/87bb80308e64b284d60c41b234b766032ed49866/libs/infinity-v3/docs/libtorch-backend.md).
