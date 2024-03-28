
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

# Infinity ♾️
[![codecov][codecov-shield]][codecov-url]
[![ci][ci-shield]][ci-url]
[![Downloads][pepa-shield]][pepa-url]

Infinity is a high-throughput, low-latency REST API for serving vector embeddings, supporting all sentence-transformer models and frameworks. Infinity is developed under [MIT License](https://github.com/michaelfeil/infinity/blob/main/LICENSE). Infinity powers inference behind [Gradient.ai](https://gradient.ai).

## Why Infinity
* **Deploy any model from MTEB**: deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
* **Fast inference backends**: The inference server is built on top of [torch](https://github.com/pytorch/pytorch), [optimum(onnx/tensorrt)](https://huggingface.co/docs/optimum/index) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), using FlashAttention to get the most out of your **NVIDIA CUDA**, **AMD ROCM**, **CPU**, **AWS INF2** or **APPLE MPS** accelerator.
* **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your device as soon as ready. 
* **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are correctly embedded. Lets API users create embeddings till infinity and beyond.
* **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API are aligned to [OpenAI's Embedding specs](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). View the docs at [https://michaelfeil.eu/infinity](https://michaelfeil.eu/infinity) on how to get started.

### Infinity demo
In this demo [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), deployed at batch-size=2. After initialization, from a second terminal 3 requests  (payload 1,1,and 5 sentences) are sent via cURL.
![](docs/demo_v0_0_1.gif)

### Latest News 🔥
- [2024/03] infinity supports now experimental int8 (cpu/cuda) and fp8 (H100/MI300) support
- [2024/03] Docs are online: https://michaelfeil.eu/infinity/latest/
- [2024/02] Community meetup at the [Run:AI Infra Club](https://discord.gg/7D4fbEgWjv)
- [2024/01] TensorRT / ONNX inference

## Getting started

### Launch the cli via pip install
```bash
pip install infinity-emb[all]
```
After your pip install, with your venv active, you can run the CLI directly.

```bash
infinity_emb --model-name-or-path BAAI/bge-small-en-v1.5
```
Check the `--help` command to get a description for all parameters.
```bash
infinity_emb --help
```

### Launch the CLI using a pre-built docker container (recommended)
Instead of installing the CLI via pip, you may also use docker to run infinity. 
Make sure you mount your accelerator, i.e. install nvidia-docker and activate with `--gpus all`.

```bash
port=7997
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path BAAI/bge-small-en-v1.5 --port $port
```
The download path at runtime can be controlled via the environment variable `HF_HOME`.

### Launch it via the Python API

Instead of the cli & RestAPI you can directly interface with the Python API. 
This gives you most flexibility. The Python API builds on `asyncio` with its `await/async` features, to allow concurrent processing of requests.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
engine = AsyncEmbeddingEngine.from_args(EngineArgs(model_name_or_path = "BAAI/bge-small-en-v1.5", engine="torch"))

async def main(): 
    async with engine: # engine starts with engine.astart()
        embeddings, usage = await engine.embed(sentences=sentences)
    # engine stops with engine.astop()
asyncio.run(main())
```

### Launch on the cloud via dstack

dstack allows you to provision a VM instance on the cloud of your choice. Write a service configuration file as below for the deployment of `BAAI/bge-small-en-v1.5` model wrapped in Infinity.

```yaml
type: service

image: michaelf34/infinity:latest
env:
  - MODEL_ID=BAAI/bge-small-en-v1.5
commands:
  - infinity_emb --model-name-or-path $MODEL_ID --port 80
port: 80
```

Then, simply run the following dstack command. After this, a prompt will appear to let you choose which VM instance to deploy the Infinity.

```shell
dstack run . -f infinity/serve.dstack.yml --gpu 16GB
```

For more detailed tutorial and general information about dstack, visit the [official doc](https://dstack.ai/examples/infinity/#run-the-configuration).


## Non-embedding features
### Reranking

Reranking gives you a score for similarity between a query and multiple documents. 
Use it in conjunction with a VectorDB+Embeddings, or as standalone for small amount of documents.
Please select a model from huggingface that is a AutoModelForSequenceClassification with one class classification.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
query = "What is the python package infinity_emb?"
docs = ["This is a document not related to the python package infinity_emb, hence...", 
    "Paris is in France!",
    "infinity_emb is a package for sentence embeddings and rerankings using transformer models in Python!"]
engine_args = EngineArgs(model_name_or_path = "BAAI/bge-reranker-base", engine="torch")

engine = AsyncEmbeddingEngine.from_args(engine_args)
async def main(): 
    async with engine:
        ranking, usage = await engine.rerank(query=query, docs=docs)
        print(list(zip(ranking, docs)))
asyncio.run(main())
```

When using the CLI, use this command to launch rerankers:
```bash
infinity_emb --model-name-or-path BAAI/bge-reranker-base
```

<details>
  <summary>You can also use text-classification (beta):</summary>
  
  Use text classification with Infinity's `classify` feature, which allows for sentiment analysis, emotion detection, and more classification tasks.

  Note: PR's to speed this section up are welcome. Currently the backend uses huggingface pipelines + dynamic batching. On top of that, a ~40% speedup should be possible.
  ```python
  import asyncio
  from infinity_emb import AsyncEmbeddingEngine, EngineArgs

  sentences = ["This is awesome.", "I am bored."]
  engine_args = EngineArgs(model_name_or_path = "SamLowe/roberta-base-go_emotions", 
      engine="torch", model_warmup=True)
  engine = AsyncEmbeddingEngine.from_args(engine_args)
  async def main(): 
      async with engine:
          predictions, usage = await engine.classify(sentences=sentences)
          return predictions, usage
  asyncio.run(main())
  ```

  Running via CLI requires a new FastAPI schema and server integration - PR's are also welcome there.
  
</details>


## Launch FAQ:
<details>
  <summary>What are embedding models?</summary>
  Embedding models can map any text to a low-dimensional dense vector which can be used for tasks like retrieval, classification, clustering, or semantic search. 
  And it also can be used in vector databases for LLMs. 
  
  The most known architecture are encoder-only transformers such as BERT, and most popular implementation include [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/).
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
  
  For the latest trends, you might want to check out one of the following models.
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

  Both models now run on two instances in one dockerfile servers. Otherwise, you could build your own FastAPI/flask instance, which wraps around the Async API.
     
</details>

<details>
  <summary>Using Langchain with Infinity</summary>
  
  Infinity has a official integration into `pip install langchain>=0.342`. 
  You can find more documentation on that here:
  https://python.langchain.com/docs/integrations/text_embedding/infinity

  ```python
  from langchain.embeddings.infinity import InfinityEmbeddings
  from langchain.docstore.document import Document
  
  documents = [Document(page_content="Hello world!", metadata={"source": "unknown"})]

  emb_model = InfinityEmbeddings(model="BAAI/bge-small", infinity_api_url="http://localhost:7997/v1")
  print(emb_model.embed_documents([doc.page_content for doc in docs]))
  ```
</details>

## Documentation
View the docs at [https://michaelfeil.eu/infinity](https://michaelfeil.eu/infinity) on how to get started.
After startup, the Swagger Ui will be available under `{url}:{port}/docs`, in this case `http://localhost:7997/docs`. You can also find a interactive preview here: https://michaelfeil-infinity.hf.space/docs

## Contribute and Develop

Install via Poetry 1.7.1 and Python3.11 on Ubuntu 22.04
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

All contributions must be made in a way to be compatible with the MIT License of this repo. 

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
[pepa-shield]: https://static.pepy.tech/badge/infinity-emb
[pepa-url]: https://www.pepy.tech/projects/infinity-emb
[codecov-shield]: https://codecov.io/gh/michaelfeil/infinity/branch/main/graph/badge.svg?token=NMVQY5QOFQ
[codecov-url]: https://codecov.io/gh/michaelfeil/infinity/branch/main
[ci-shield]: https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg
[ci-url]: https://github.com/michaelfeil/infinity/actions
