# Infinity ♾️
Embedding Inference Server - finding TGI for embeddings. Infinity is developed under MIT Licence - https://github.com/michaelfeil/infinity

<!-- PROJECT SHIELDS -->
[![codecov](https://codecov.io/gh/michaelfeil/infinity/branch/main/graph/badge.svg?token=NMVQY5QOFQ)]
[![CI](https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg)]
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
--------

## Why Infinity:
Infinity provides the following features:
- **Deploy virtually any SentenceTransformer** - deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
- **Fast inference**: The inference server is built on top of [torch](https:) and [ctranslate2](https://github.com/OpenNMT/CTranslate2) under the hood, getting most out of your **CUDA** or **CPU** hardware.
- **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your GPU/CPU as soon as ready. 
- **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are identical to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
- **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API specs are aligned to OpenAI. See below on how to get started.

# Demo:
With **infinity** we can launch any SentenceTransformer model via the API.
In this gif below, we use [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), deployed at batch-size=2. After initialization, from a second terminal 3 requests  (payload 1,1,and 5 sentences) are sent via cURL.
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

# Contribute and Develop

Install via Poetry 1.6.1 and Python3.10 on Ubuntu 22.04
```bash
cd libs/infinity_emb
poetry install --extras all --with test
```

To pass the CI:
```bash
cd libs/infinity_emb
make format
make lint
poetry run pytest ./tests
```