# Quickstart

Infinity v3 is the native Rust server with LibTorch inference. These preview
docs track `main`; available features depend on the image version you deploy.

## Run on an NVIDIA GPU

Choose a published `nvidia-3.0.0-alpha.*` tag from the
[container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3).
For example:

```bash
docker run --rm --gpus all -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0-alpha.16 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cuda --dtype float16 \
  --hostname 0.0.0.0 --port 8000
```

The image starts `infinity_v3` directly. There is no `v2` subcommand.
Each visible GPU runs a model replica; use `--device-id 0` to select one GPU.

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
[supported model and task coverage](docs/libtorch-model-coverage.md) and
[API reference](api.md) for your checkpoint and request format. Python engine
APIs and the v2 backend choices do not carry over to the native server.

The [Python / v2 docs](../cli_v2.md) remain available
during migration. See [v3 licensing](LICENSING.md) for the native project's terms.
