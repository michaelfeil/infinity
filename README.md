
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

## Why Infinity:
Infinity provides the following features:
* **Deploy any model from MTEB**: deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
* **Fast inference backends**: The inference server is built on top of [torch](https://github.com/pytorch/pytorch), [optimum(onnx/tensorrt)](https://github.com/qdrant/fastembed) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), using FlashAttention to get the most out of **CUDA**, **ROCM**, **CPU** or **MPS** chips.
* **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your device as soon as ready. Similar max throughput on GPU as text-embeddings-inference.
* **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are identical to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
* **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API are aligned to [OpenAI's Embedding specs](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). See below on how to get started.

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
  To install via Poetry use Poetry 1.7.1, Python 3.11 on Ubuntu 22.04
  ```bash
  git clone https://github.com/michaelfeil/infinity
  cd infinity
  cd libs/infinity_emb
  poetry install --extras all
  ```
</details>

### Launch the CLI using a pre-built docker container (recommended)
```bash
model=BAAI/bge-small-en-v1.5
port=7997
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
```
The download path at runtime, can be controlled via the environment variable `SENTENCE_TRANSFORMERS_HOME`.

### or launch the cli after the pip install
After your pip install, with your venv activate, you can run the CLI directly.
Check the `--help` command to get a description for all parameters.

```bash
infinity_emb --help
```

### or launch it via Python

You can use in a async context with asyncio. 
This gives you most flexibility, but is a bit more advanced.
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

### or launch on the cloud via dstack

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

<details>
  <summary>You can also use text-classification (beta):</summary>
  
  Note: PR's to speed this section up are welcome, a 40% speedup is propable, currently the backend uses huggingface pipelines + dynamic batching.
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
</details>


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

# Documentation
After startup, the Swagger Ui will be available under `{url}:{port}/docs`, in this case `http://localhost:7997/docs`.

# Contribute and Develop

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
