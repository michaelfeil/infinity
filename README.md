# Infinity v3 ♾️

Infinity serves embeddings, reranking, classification, and typed decision models
through a REST API. The native v3 server is written in Rust and uses LibTorch,
the C++ interface of PyTorch, for model inference.

**Looking for Python / v2?** See the [Python server README](libs/infinity_emb/README.md)
and [v2 documentation](https://michaelfeil.eu/infinity/main/deploy/).
The `infinity-emb` Python package and `infinity_emb v2` command belong to that server.

## History

- **2026/10:** Native Rust/LibTorch v3 server for embeddings, reranking, classification and typed decisions.
- **2025/07:** Blackwell support in the Python server.
- **2024:** Python / v2 gained multi-model and multimodal serving, Python clients, and broader GPU, CPU, ONNX and TensorRT deployment options.
- **2023/10:** Initial Infinity release.

See the [Python / v2 README](libs/infinity_emb/README.md) for the full earlier updates.

## Run with Docker

Use an NVIDIA GPU with compute capability SM80 (Ampere) or newer and NVIDIA
Container Toolkit:

```bash
docker run --rm --gpus all -p 8000:8000 -v infinity-cache:/data \
  ghcr.io/michaelfeil/infinity_v3:nvidia-3.0.0 \
  --model-id sentence-transformers/all-MiniLM-L6-v2 \
  --torch-device cuda --dtype bfloat16 \
  --hostname 0.0.0.0 --port 8000
```

The image starts `infinity_v3` directly. All visible CUDA GPUs are used by
default, with a complete model replica on each GPU. Pooling is selected from the
checkpoint metadata. Pin a published version or image digest from the
[container registry](https://github.com/michaelfeil/infinity/pkgs/container/infinity_v3)
for reproducible deployments.

Send an embedding request:

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2","input":["A document about semantic search."]}'
```

Open [localhost:8000/docs](http://localhost:8000/docs) for the running server's
interactive API. `/health` reports readiness after loading and warmup.

## Models and tasks

The [supported model catalog](libs/infinity-v3/docs/supported-models.md) groups
named checkpoints by task, with configuration, input requirements, support basis,
and model-license notes:

- Dense text and multimodal embeddings through `/v1/embeddings`.
- Token-vector embeddings for late-interaction retrieval.
- Reranking through `/rerank`.
- Sequence classification and reward scoring through `/v1/classify`.
- Token classification and named-entity recognition through `/predict_tokens`.
- Typed decisions through `/v1/systemone`.
- SPLADE sparse embeddings through `/embed_sparse`.

Support depends on the checkpoint's architecture and trained task head. Each GPU
must fit the full model, including all experts for MoE checkpoints.

## Documentation

- [Quickstart](https://michaelfeil.eu/infinity/main/v3/quickstart/)
- [Native v3 overview, performance and configuration](libs/infinity-v3/README.md)
- [Supported models and configuration](libs/infinity-v3/docs/supported-models.md)
- [CLI and request options](libs/infinity-v3/docs/cli.md)
- [API reference](https://michaelfeil.eu/infinity/main/v3/api/)
- [Embedding formats and clients](libs/infinity-v3/docs/embedding-tensors.md)
- [Build and contribute](libs/infinity-v3/CONTRIBUTING.md)
- [Python / v2 server](libs/infinity_emb/README.md)

## Licensing

Covered original v3 contributions use the
[Infinity Restricted MIT License 1.0](libs/infinity-v3/LICENSING.md),
with retained third-party and earlier grants. This is a custom source-available
license; see the [full v3 license](libs/infinity-v3/LICENSE) for its terms.
The Python packages retain the [MIT License](LICENSE). Model weights and
dependencies have their own licenses.
