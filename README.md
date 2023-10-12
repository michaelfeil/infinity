# Infinity ♾️
Embedding Inference Server - finding TGI for embeddings

## Why Infinity:
Infinity provides the following features:
- **Fast inference**: The inference server is built on top of [torch](https:) and [ctranslate2](https://github.com/OpenNMT/CTranslate2) under the hood, getting most out of your **CUDA** or **CPU** hardware.
- **Dynamic, optimal batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your GPU/CPU as soon as ready. 
- **Correct and tested implementation**: Unit and end-to-end tested. API embeddings are identical to [sentence-transformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
- **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API specs are aligned to OpenAI. See below on how to get started.

# Demo:
A quick demo of launching: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) with batch-size=2 and sending 3 requests via cURL.
![](docs/demo_v0_0_1.gif)

# Getting started
Install via Poetry and Python
```bash
cd libs/infinity_emb
poetry install --extras all
```

### Launch via Python
```Python
from infinity_emb import create server
create_server()
```

### or launch the `create_server()` command via CLI
```bash
infinity_emb --help
```

### or launch the CLI using a pre-built docker container
Get the Python
```bash
model=sentence-transformers/all-MiniLM-L6-v2
port=8080
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port --engine ctranslate2
```
The download path at runtime, can be controlled via the environment variable `SENTENCE_TRANSFORMERS_HOME`.


# Documentation
After startup, the Swagger Ui will be available under `{url}:{port}/docs`, in this case `http://localhost:8080/docs`.