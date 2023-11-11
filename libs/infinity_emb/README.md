
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

# Infinity ♾️
[![codecov][codecov-shield]][codecov-url]
[![ci][ci-shield]][ci-url]
[![Downloads][pepa-shield]][pepa-url]

Infinity is a high-throughput, low-latency REST API for serving vector embeddings, supporting a wide range of sentence-transformer models and frameworks. Infinity is developed under MIT Licence: https://github.com/michaelfeil/infinity

## Why Infinity:
Infinity provides the following features:
- **Deploy virtually any SentenceTransformer** - deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
- **Fast inference backends**: The inference server is built on top of [torch](https://github.com/pytorch/pytorch), [fastembed(onnx-cpu)](https://github.com/qdrant/fastembed) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), getting most out of your **CUDA** or **CPU** hardware.
- **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your GPU/CPU as soon as ready. 
- **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are identical to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
- **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API are aligned to [OpenAI's Embedding specs](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). See below on how to get started.

# Infinity demo:
In this gif below, we use [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), deployed at batch-size=2. After initialization, from a second terminal 3 requests  (payload 1,1,and 5 sentences) are sent via cURL.
![](docs/demo_v0_0_1.gif)

# Getting started

Install via pip
```bash
pip install infinity-emb[all]
```

<details>
  <summary>Install from source with Poetry</summary>
  
  Advanced:
  To install via Poetry use Poetry 1.6.1, Python 3.10 on Ubuntu 22.04
  ```bash
  git clone https://github.com/michaelfeil/infinity
  cd infinity
  cd libs/infinity_emb
  poetry install --extras all
  ```
</details>


### Launch via Python
```Python
from infinity_emb import create server
fastapi_app = create_server()
```
or use the AsyncAPI directly.:

```python
from infinity_emb import AsyncEmbeddingEngine, transformer
sentences = ["Embedded this is sentence via Infinity.", "Paris is in France."]
engine = AsyncEmbeddingEngine(engine=transformer.InferenceEngine.torch)
async with engine: # engine starts with engine.astart()
    embeddings = np.array(await engine.embed(sentences))
# engine stops with engine.astop()
```

### or launch the `create_server()` command via CLI
```bash
infinity_emb --help
```

### or launch the CLI using a pre-built docker container

```bash
model=BAAI/bge-small-en-v1.5
port=8080
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
```
The download path at runtime, can be controlled via the environment variable `SENTENCE_TRANSFORMERS_HOME`.

### Launch FAQ:
<details>
  <summary>What are embedding models?</summary>
  Embedding models can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification, clustering, or semantic search. 
  And it also can be used in vector databases for LLMs. 
  
  The most know architecture are encoder-only transformers such as BERT, and most popular implementation include [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/).
</details>

<details>
  <summary>What models are supported?</summary>
  
  All models of the sentence transformers org are supported https://huggingface.co/sentence-transformers / sbert.net. 
  LLM's like LLAMA2-7B are not intended for deployment.

  With the command `--engine torch` the model must be compatible with https://github.com/UKPLab/sentence-transformers/.
    - only models from Huggingface are supported.
  
  With the command `--engine ctranslate2`
    - only `BERT` models are supported.
    - only models from Huggingface are supported.
  
  For the latest trends, you might want to check out one of the folloing models.
    https://huggingface.co/spaces/mteb/leaderboard
    
</details>

<details>
  <summary>Launching multiple models in one dockerfile</summary>
  
  Multiple models on one GPU is in experimental mode. You can use the following temporary solution:
  ```Dockerfile
  FROM michaelf34/infinity:latest
  # Dockerfile-ENTRYPOINT for multiple models via multiple ports
  ENTRYPOINT ["/bin/sh", "-c", \
   "(. /app/.venv/bin/activate && infinity_emb --port 8080 --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 &);\
   (. /app/.venv/bin/activate && infinity_emb --port 8081 --model-name-or-path intfloat/e5-large-v2 )"]
  ```
  
  You can build and run it via:  
  ```bash
  docker build -t custominfinity . && docker run -it --gpus all -p 8080:8080 -p 8081:8081 custominfinity
  ```

  Both models now run on two instances in one dockerfile servers.
     
</details>

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



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/michaelfeil/infinity.svg?style=for-the-badge
[contributors-url]: https://github.com/michaelfeil/infinity/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/michaelfeil/infinity.svg?style=for-the-badge
[forks-url]: https://github.com/michaelfeil/infinity/network/members
[stars-shield]: https://img.shields.io/github/stars/michaelfeil/infinity.svg?style=for-the-badge
[stars-url]: https://github.com/michaelfeil/infinity/stargazers
[issues-shield]: https://img.shields.io/github/issues/michaelfeil/infinity.svg?style=for-the-badge
[issues-url]: https://github.com/michaelfeil/infinity/issues
[license-shield]: https://img.shields.io/github/license/michaelfeil/infinity.svg?style=for-the-badge
[license-url]: https://github.com/michaelfeil/infinity/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/michael-feil
[pepa-shield]: https://static.pepy.tech/badge/infinity-emb
[pepa-url]: https://www.pepy.tech/projects/infinity-emb
[codecov-shield]: https://codecov.io/gh/michaelfeil/infinity/branch/main/graph/badge.svg?token=NMVQY5QOFQ
[codecov-url]: https://codecov.io/gh/michaelfeil/infinity/branch/main
[ci-shield]: https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg
[ci-url]: https://github.com/michaelfeil/infinity/actions
