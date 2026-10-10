# Infinity — Python

Infinity is a high-throughput, low-latency inference server for text embeddings,
reranking, classification, and image/audio embeddings. It provides a FastAPI
server and a Python API, with dynamic batching and multiple model backends.

This package contains the Python implementation, launched with `infinity_emb v2`.
[Infinity v3](https://github.com/michaelfeil/infinity/tree/main/libs/infinity_v3)
is a separate, planned Rust project in the same repository.

## Install and run

```bash
pip install 'infinity-emb[all]'
infinity_emb v2 --model-id BAAI/bge-small-en-v1.5
```

Run `infinity_emb v2 --help` for available options. Multiple models can be served
by repeating `--model-id`.

For NVIDIA GPUs, the existing Docker image is:

```bash
docker run --gpus all -p 7997:7997 \
  michaelf34/infinity:latest \
  v2 --model-id BAAI/bge-small-en-v1.5 --port 7997
```

The CPU image is `michaelf34/infinity:latest-cpu`. Published version tags can be
used instead of the moving `latest` tags.

## Python API

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(model_name_or_path="BAAI/bge-small-en-v1.5")
)

async def main():
    async with engine:
        embeddings, usage = await engine.embed(sentences=["Hello, Infinity!"])
        print(embeddings, usage)

asyncio.run(main())
```

## Documentation and development

- [Full usage guide](https://github.com/michaelfeil/infinity#readme)
- [Documentation](https://michaelfeil.github.io/infinity/)
- [Contributor guide](https://github.com/michaelfeil/infinity/blob/main/docs/docs/contribution.md)

Development commands run from `libs/infinity_emb`. This package owns its Poetry
dependencies, tests, and Dockerfiles. Its package version follows the existing
`0.x.y` release series independently of the CLI generation.

Infinity's Python implementation is released under the
[MIT license](https://github.com/michaelfeil/infinity/blob/main/LICENSE).
