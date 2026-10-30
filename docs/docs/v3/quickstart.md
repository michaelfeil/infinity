# Quickstart

Infinity v3 is the native Rust server with LibTorch inference. Use the
documentation version matching your image; `main` documents ongoing development.

## Run on an NVIDIA GPU

CUDA inference requires SM80 (Ampere) or newer; SM75/Turing is unsupported.

Use the versioned NVIDIA image from the
[container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3).
The 3.0.0 launch command is:

```bash
docker run --rm --gpus all -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cuda --dtype float16 \
  --hostname 0.0.0.0 --port 8000
```

The image starts `infinity_v3` directly. There is no `v2` subcommand.
All visible CUDA GPUs run complete model replicas by default.
Choose another task or checkpoint from the
[supported models and configuration](docs/supported-models.md) catalog.

Versioned tags such as `nvidia-3.0.0` identify a release. `nvidia-latest` follows
stable releases; `nvidia-main` follows development builds. Pin a version or digest
for reproducible deployments.

## CPU deployments

CPU builds from current `main` use LibTorch. The `cpu-main` development image
runs the same native backend on CPU:

```bash
docker run --rm -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:cpu-main \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cpu --dtype float32 \
  --hostname 0.0.0.0 --port 8000
```

For release images, use documentation matching that release. To build locally,
follow the [LibTorch source instructions](docs/libtorch-backend.md#build-from-source).

## Send an embedding request

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2","input":["A quick example."]}'
```

Open [localhost:8000/docs](http://localhost:8000/docs) for the running server's
API documentation. `/health` reports readiness. See [CLI and request options](docs/cli.md)
for model routing, batching, truncation and decision inputs.

## Moving from Python / v2

Use the v3 container and CLI instead of `infinity_emb v2`. Check the
[supported models and configuration](docs/supported-models.md) and
[API reference](api.md) for your checkpoint and request format. Python engine
APIs and the v2 backend choices do not carry over to the native server.

The [Python / v2 docs](../cli_v2.md) remain available
during migration. See [v3 licensing](LICENSING.md) for the native project's terms.
