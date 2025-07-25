
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
[![DOI](https://zenodo.org/badge/703686617.svg)](https://zenodo.org/doi/10.5281/zenodo.11406462)
![Docker pulls](https://img.shields.io/docker/pulls/michaelf34/infinity)


Infinity is a high-throughput, low-latency REST API for serving text-embeddings, reranking models, clip, clap and colpali. Infinity is developed under [MIT License](https://github.com/michaelfeil/infinity/blob/main/LICENSE).

## Why Infinity
* **Deploy any model from HuggingFace**: deploy any embedding, reranking, clip and sentence-transformer model from [HuggingFace]( https://huggingface.co/models?other=text-embeddings-inference&sort=trending)
* **Fast inference backends**: The inference server is built on top of [PyTorch](https://github.com/pytorch/pytorch), [optimum (ONNX/TensorRT)](https://huggingface.co/docs/optimum/index) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), using FlashAttention to get the most out of your **NVIDIA CUDA**, **AMD ROCM**, **CPU**, **AWS INF2** or **APPLE MPS** accelerator. Infinity uses dynamic batching and tokenization dedicated in worker threads.
* **Multi-modal and multi-model**: Mix-and-match multiple models. Infinity orchestrates them.
* **Tested implementation**: Unit and end-to-end tested. Embeddings via infinity are correctly embedded. Lets API users create embeddings till infinity and beyond.
* **Easy to use**: Built on [FastAPI](https://fastapi.tiangolo.com/). Infinity CLI v2 allows launching of all arguments via Environment variable or argument. OpenAPI aligned to [OpenAI's API specs](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). View the docs at [https://michaelfeil.github.io/infinity](https://michaelfeil.github.io/infinity/) on how to get started.

<p align="center">
  <a href="https://github.com/basetenlabs/truss-examples/tree/7025918c813d08d718b8939f44f10651a0ff2c8c/custom-server/infinity-embedding-server"><img src="https://avatars.githubusercontent.com/u/54861414" alt="Logo Baseten.co" width="50"/></a>
  <a href="https://github.com/runpod-workers/worker-infinity-embedding"><img src="https://github.com/user-attachments/assets/24f1906d-31b8-4e16-a479-1382cbdea046" alt="Logo Runpod" width="50"/></a>
  <a href="https://www.truefoundry.com/cognita"><img src="https://github.com/user-attachments/assets/1b515b0f-2332-4b12-be82-933056bddee4" alt="Logo TrueFoundry" width="50"/></a>
  <a href="https://vast.ai/article/serving-infinity"><img src="https://github.com/user-attachments/assets/8286d620-f403-48f5-bd7f-f471b228ae7b" alt="Logo Vast" width="46"/></a>
  <a href="https://www.dataguard.de"><img src="https://github.com/user-attachments/assets/3fde1ac6-c299-455d-9fc2-ba4012799f9c" alt="Logo DataGuard" width="50"/></a>
  <a href="https://community.sap.com/t5/artificial-intelligence-and-machine-learning-blogs/bring-open-source-llms-into-sap-ai-core/ba-p/13655167"><img src="https://github.com/user-attachments/assets/743e932b-ed5b-4a71-84cb-f28235707a84" alt="Logo SAP" width="47"/></a>
  <a href="https://x.com/StuartReid1929/status/1763434100382163333"><img src="https://github.com/user-attachments/assets/477a4c54-1113-434b-83bc-1985f10981d3" alt="Logo Nosible" width="44"/></a>
  <a href="https://github.com/freshworksinc/freddy-infinity"><img src="https://github.com/user-attachments/assets/a68da78b-d958-464e-aaf6-f39132be68a0" alt="Logo FreshWorks" width="50"/></a>
  <a href="https://github.com/dstackai/dstack/tree/master/examples/deployment/infinity"><img src="https://github.com/user-attachments/assets/9cde2d6b-dc16-4f0a-81ba-535a84321467" alt="Logo Dstack" width="50"/></a>
  <a href="https://embeddedllm.com/blog/"><img src="https://avatars.githubusercontent.com/u/148834374" alt="Logo JamAI" width="50"/></a>
  <a href="https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct#infinity_emb"><img src="https://avatars.githubusercontent.com/u/1961952" alt="Logo Alibaba Group" width="50"/></a>
  <a href="https://github.com/bentoml/BentoInfinity/"><img src="https://avatars.githubusercontent.com/u/49176046" alt="Logo BentoML" width="50"/></a>
  <a href="https://x.com/bo_wangbo/status/1766371909086724481"><img src="https://avatars.githubusercontent.com/u/60539444" alt="Logo JinaAi" width="50"/></a>
  <a href="https://github.com/dwarvesf/llm-hosting"><img src="https://avatars.githubusercontent.com/u/10388449" alt="Logo Dwarves Foundation" width="50"/></a>
  <a href="https://github.com/huggingface/chat-ui/blob/daf695ea4a6e2d081587d7dbcae3cacd466bf8b2/docs/source/configuration/embeddings.md#openai"><img src="https://avatars.githubusercontent.com/u/25720743" alt="Logo HF" width="50"/></a>
  <a href="https://www.linkedin.com/posts/markhng525_join-me-and-ekin-karabulut-at-the-ai-infra-activity-7163233344875393024-LafB?utm_source=share&utm_medium=member_desktop"><img src="https://avatars.githubusercontent.com/u/86131705" alt="Logo Gradient.ai" width="50"/></a>
</p> 

### Latest News 🔥

- [2024/11] AMD, CPU, ONNX docker images
- [2024/10] `pip install infinity_client`
- [2024/07] Inference deployment example via [Modal](./infra/modal/README.md) and a [free GPU deployment](https://infinity.modal.michaelfeil.eu/)
- [2024/06] Support for multi-modal: clip, text-classification & launch all arguments from env variables
- [2024/05] launch multiple models using the `v2` cli, including `--api-key`
- [2024/03] infinity supports experimental int8 (cpu/cuda) and fp8 (H100/MI300) support
- [2024/03] Docs are online: https://michaelfeil.github.io/infinity/latest/
- [2024/02] Community meetup at the [Run:AI Infra Club](https://discord.gg/7D4fbEgWjv)
- [2024/01] TensorRT / ONNX inference
- [2023/10] Initial release

## Getting started
### Launch the cli via pip install
```bash
pip install infinity-emb[all]
```
After your pip install, with your venv active, you can run the CLI directly.

```bash
infinity_emb v2 --model-id BAAI/bge-small-en-v1.5
```
Check the `v2 --help` command to get a description for all parameters.
```bash
infinity_emb v2 --help
```
### Launch the CLI using a pre-built docker container (recommended)
Instead of installing the CLI via pip, you may also use docker to run `michaelf34/infinity`. 
Make sure you mount your accelerator ( i.e. install `nvidia-docker` and activate with `--gpus all`). 

```bash
port=7997
model1=michaelfeil/bge-small-en-v1.5
model2=mixedbread-ai/mxbai-rerank-xsmall-v1
volume=$PWD/data

docker run -it --gpus all \
 -v $volume:/app/.cache \
 -p $port:$port \
 michaelf34/infinity:latest \
 v2 \
 --model-id $model1 \
 --model-id $model2 \
 --port $port
```
The cache path inside the docker container is set by the environment variable `HF_HOME`.

#### Specialized docker images
<details>
  <summary>Docker container for CPU</summary>
  Use the `latest-cpu` image or `x.x.x-cpu` for slimer image. 
  Run like any other cpu-only docker image. 
  Optimum/Onnx is often the prefered engine. 

  ```
  docker run -it \
  -v $volume:/app/.cache \
  -p $port:$port \
  michaelf34/infinity:latest-cpu \
  v2 \
  --engine optimum \
  --model-id $model1 \
  --model-id $model2 \
  --port $port
  ```
</details>

<details>
  <summary>Docker Container for ROCm (MI200 Series and MI300 Series)</summary>
  Use the `latest-rocm` image or `x.x.x-rocm` for rocm compatible inference.
  **This image is currently not build via CI/CD (to large), consider pinning to exact version.**
  Make sure you have ROCm is correctly installed and ready to use with Docker.

  Visit [Docs](https://michaelfeil.github.io/infinity) for more info.
</details>
 
<details>
  <summary>Docker Container for Onnx-GPU, Cuda Extensions, TensorRT</summary>
  Use the `latest-trt-onnx` image or `x.x.x-trt-onnx` for nvidia compatible inference.
  **This image is currently not build via CI/CD (to large), consider pinning to exact version.**

  This image has support for:
  - ONNX-Cuda "CudaExecutionProvider" 
  - ONNX-TensorRT "TensorRTExecutionProvider" (may not always work due to version mismatch with ORT)
  - CudaExtensions and packages, e.g. Tri-Dao's `pip install flash-attn` package when using Pytorch.
  - nvcc compiler support
  
  ```
  docker run -it \
  -v $volume:/app/.cache \
  -p $port:$port \
  michaelf34/infinity:latest-trt-onnx \
  v2 \
  --engine optimum \
  --device cuda \
  --model-id $model1 \
  --port $port
  ```
</details>

#### Using local models with Docker container

In order to deploy a local model with a docker container, you need to mount the model inside the container and specify the path in the container to the launch command.

Example:
```bash
git lfs install 
cd /tmp
mkdir models && cd models && git clone https://huggingface.co/BAAI/bge-small-en-v1.5
docker run -it   -v /tmp/models:/models  -p 8081:8081  michaelf34/infinity:latest v2  --model-id "/models/bge-small-en-v1.5" --port 8081
```

#### Advanced CLI usage

<details>
  <summary>Launching multiple models at once</summary>
  
  Since `infinity_emb>=0.0.34`, you can use cli `v2` method to launch multiple models at the same time.
  Checkout `infinity_emb v2 --help` for all args and validation.

  Multiple Model CLI Playbook:                                                                                         
   - 1. cli options can be repeated e.g. `v2 --model-id model/id1 --model-id model/id2 --batch-size 8 --batch-size 4`. This will create two models `model/id1` and `model/id2`
   - 2. or adapt the defaults by setting ENV Variables separated by `;`: `INFINITY_MODEL_ID="model/id1;model/id2;" && INFINITY_BATCH_SIZE="8;4;"`
   - 3. single items are broadcasted to `--model-id` length,  `v2 --model-id model/id1 --model-id/id2 --batch-size 8` making both models have batch-size 8.
   - 4. Everything is broadcasted to the number of `--model-id` + API requests are routed to the `--served-model-name/--model-id`
</details>

<details>
  <summary>Using environment variables instead of the cli</summary>
  All CLI arguments are also launchable via environment variables.

  Environment variables start with `INFINITY_{UPPER_CASE_SNAKE_CASE}` and often match the `--{lower-case-kebab-case}` cli arguments.
  
  The following two are equivalent:
  - CLI `infinity_emb v2 --model-id BAAI/bge-base-en-v1.5`
  - ENV-CLI: `export INFINITY_MODEL_ID="BAAI/bge-base-en-v1.5" && infinity_emb v2`

  Multiple arguments can be used via `;` syntax: `INFINITY_MODEL_ID="model/id1;model/id2;"`
</details>

<details>
  <summary>API Key</summary>
  Supply an `--api-key secret123` via CLI or ENV INFINITY_API_KEY="secret123".
</details>

<details>
  <summary>Chosing the fastest engine</summary>
  
  With the command `--engine torch` the model must be compatible with https://github.com/UKPLab/sentence-transformers/ and AutoModel

  With the command `--engine optimum`, there must be an onnx file. Models from https://huggingface.co/Xenova are recommended.
  
  With the command `--engine ctranslate2`
    - only `BERT` models are supported.
</details>

<details>
  <summary>Telemetry opt-out</summary>
  
  See which telemetry is collected: https://michaelfeil.eu/infinity/main/telemetry/
  ```
  # Disable
  export INFINITY_ANONYMOUS_USAGE_STATS="0"
  ```
</details>

### Supported Tasks and Models by Infinity

Infinity aims to be the inference server supporting most functionality for embeddings, reranking and related RAG tasks. The following  Infinity tests 15+ architectures and all of the below cases in the Github CI.
Click on the sections below to find tasks and **validated example models**.

<details>
  <summary>Text Embeddings</summary>
  
  Text embeddings measure the relatedness of text strings. Embeddings are used for search, clustering, recommendations.
  Think about a private deployed version of openai's text embeddings. https://platform.openai.com/docs/guides/embeddings

  Tested embedding models:
  - [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
  - [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)
  - [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
  - [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
  - [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
  - [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
  - [jinaai/jina-embeddings-v3](nomic-ai/nomic-embed-text-v1.5)
  - [BAAI/bge-m3, no sparse](https://huggingface.co/BAAI/bge-m3)
  - decoder-based models. Keep in mind that they are ~20-100x larger (&slower) than bert-small models:
    - [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/discussions/20)
    - [Salesforce/SFR-Embedding-2_R](https://huggingface.co/Salesforce/SFR-Embedding-2_R/discussions/6)
    - [Alibaba-NLP/gte-Qwen2-7B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct/discussions/39)

  Other models:
  - Most embedding model are likely supported: https://huggingface.co/models?pipeline_tag=feature-extraction&other=text-embeddings-inference&sort=trending
  - Check MTEB leaderboard for models https://huggingface.co/spaces/mteb/leaderboard.
</details>

<details>
  <summary>Reranking</summary>
  Given a query and a list of documents, Reranking indexes the documents from most to least semantically relevant to the query.
  Think like a locally deployed version of https://docs.cohere.com/reference/rerank
  
  Tested reranking models:
  - [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
  - [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
  - [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
  - [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
  - [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
  - [jinaai/jina-reranker-v1-turbo-en](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)

  Other reranking models:
  - Reranking Models supported by infinity are bert-style classification Models with one category.
  - Most reranking model are likely supported: https://huggingface.co/models?pipeline_tag=text-classification&other=text-embeddings-inference&sort=trending
  - https://huggingface.co/models?pipeline_tag=text-classification&sort=trending&search=rerank
</details>

<details>
  <summary>Multi-modal and cross-modal - image and audio embeddings</summary>
  Specialized embedding models that allow for image<->text or image<->audio search. 
  Typically, these models allow for text<->text, text<->other and other<->other search, with accuracy tradeoffs when going cross-modal.
  
  Image<->text models can be used for e.g. photo-gallery search, where users can type in keywords to find photos, or use a photo to find related images.
  Audio<->text models are less popular, and can be e.g. used to find music songs based on a text description or related music songs.
  
  Tested image<->text models:
  - [wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
  - [jinaai/jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1)
  - [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
  - Models of type: ClipModel / SiglipModel in `config.json`
  
  Tested audio<->text models:
  - [Clap Models from LAION](https://huggingface.co/collections/laion/clap-contrastive-language-audio-pretraining-65415c0b18373b607262a490)
  - limited number open source organizations training these models
  - * Note: The sampling rate of the audio data needs to match the model *

  Not supported:
  - Plain vision models e.g. nomic-ai/nomic-embed-vision-v1.5
</details>

<details>
  <summary>ColBert-style late-interaction Embeddings</summary>
  ColBert Embeddings don't perform any special Pooling methods, but return the raw **token embeddings**.
  The **token embeddings** are then to be scored with the MaxSim Metric in a VectorDB (Qdrant / Vespa)
  
  For usage via the RestAPI, late-interaction embeddings may best be transported via `base64` encoding.
  Example notebook: https://colab.research.google.com/drive/14FqLc0N_z92_VgL_zygWV5pJZkaskyk7?usp=sharing
  
  Tested colbert models:
  - [colbert-ir/colbertv2.0](https://huggingface.co/colbert-ir/colbertv2.0)
  - [jinaai/jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2)
  - [mixedbread-ai/mxbai-colbert-large-v1](https://huggingface.co/mixedbread-ai/mxbai-colbert-large-v1)
  - [answerai-colbert-small-v1 - click link for instructions](https://huggingface.co/answerdotai/answerai-colbert-small-v1/discussions/14)

</details>

<details>
  <summary>ColPali-style late-interaction Image<->Text Embeddings</summary>
  Similar usage to ColBert, but scanning over an image<->text instead of only text.
  
  For usage via the RestAPI, late-interaction embeddings may best be transported via `base64` encoding.
  Example notebook: https://colab.research.google.com/drive/14FqLc0N_z92_VgL_zygWV5pJZkaskyk7?usp=sharing
  
  Tested ColPali/ColQwen models:
  - [vidore/colpali-v1.2-merged](https://huggingface.co/michaelfeil/colpali-v1.2-merged)
  - [michaelfeil/colqwen2-v0.1](https://huggingface.co/michaelfeil/colqwen2-v0.1)
  - No lora adapters supported, only "merged" models.
</details>

<details>
  <summary>Text classification</summary>
  A bert-style multi-label text classification. Classifies it into distinct categories. 
  
  Tested models:
  - [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), financial news classification
  - [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions), text to emotion categories.
  - bert-style text-classifcation models with more than >1 label in `config.json`
</details>

### Infinity usage via the Python API

Instead of the cli & RestAPI use infinity's interface via the Python API. 
This gives you most flexibility. The Python API builds on `asyncio` with its `await/async` features, to allow concurrent processing of requests. Arguments of the CLI are also available via Python.

#### Embeddings
```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
array = AsyncEngineArray.from_args([
  EngineArgs(model_name_or_path = "BAAI/bge-small-en-v1.5", engine="torch", embedding_dtype="float32", dtype="auto")
])

async def embed_text(engine: AsyncEmbeddingEngine): 
    async with engine: 
        embeddings, usage = await engine.embed(sentences=sentences)
    # or handle the async start / stop yourself.
    await engine.astart()
    embeddings, usage = await engine.embed(sentences=sentences)
    await engine.astop()
asyncio.run(embed_text(array[0]))
```

#### Reranking

Reranking gives you a score for similarity between a query and multiple documents. 
Use it in conjunction with a VectorDB+Embeddings, or as standalone for small amount of documents.
Please select a model from huggingface that is a AutoModelForSequenceClassification compatible model with one class classification.

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
query = "What is the python package infinity_emb?"
docs = ["This is a document not related to the python package infinity_emb, hence...", 
    "Paris is in France!",
    "infinity_emb is a package for sentence embeddings and rerankings using transformer models in Python!"]
array = AsyncEmbeddingEngine.from_args(
  [EngineArgs(model_name_or_path = "mixedbread-ai/mxbai-rerank-xsmall-v1", engine="torch")]
)

async def rerank(engine: AsyncEmbeddingEngine): 
    async with engine:
        ranking, usage = await engine.rerank(query=query, docs=docs)
        print(list(zip(ranking, docs)))
    # or handle the async start / stop yourself.
    await engine.astart()
    ranking, usage = await engine.rerank(query=query, docs=docs)
    await engine.astop()

asyncio.run(rerank(array[0]))
```

When using the CLI, use this command to launch rerankers:
```bash
infinity_emb v2 --model-id mixedbread-ai/mxbai-rerank-xsmall-v1
```

#### Image-Embeddings: CLIP models

CLIP models are able to encode images and text at the same time. 

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

sentences = ["This is awesome.", "I am bored."]
images = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
engine_args = EngineArgs(
    model_name_or_path = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", 
    engine="torch"
)
array = AsyncEngineArray.from_args([engine_args])

async def embed(engine: AsyncEmbeddingEngine): 
    await engine.astart()
    embeddings, usage = await engine.embed(sentences=sentences)
    embeddings_image, _ = await engine.image_embed(images=images)
    await engine.astop()

asyncio.run(embed(array["wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"]))
```

#### Audio-Embeddings: CLAP models

CLAP models are able to encode audio and text at the same time. 

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
import requests
import soundfile as sf
import io

sentences = ["This is awesome.", "I am bored."]

url = "https://bigsoundbank.com/UPLOAD/wav/2380.wav"
raw_bytes = requests.get(url, stream=True).content

audios = [raw_bytes]
engine_args = EngineArgs(
    model_name_or_path = "laion/clap-htsat-unfused",
    dtype="float32", 
    engine="torch"

)
array = AsyncEngineArray.from_args([engine_args])

async def embed(engine: AsyncEmbeddingEngine): 
    await engine.astart()
    embeddings, usage = await engine.embed(sentences=sentences)
    embedding_audios = await engine.audio_embed(audios=audios)
    await engine.astop()

asyncio.run(embed(array["laion/clap-htsat-unfused"]))
```

#### Text Classification 

Use text classification with Infinity's `classify` feature, which allows for sentiment analysis, emotion detection, and more classification tasks.

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

sentences = ["This is awesome.", "I am bored."]
engine_args = EngineArgs(
    model_name_or_path = "SamLowe/roberta-base-go_emotions", 
    engine="torch", model_warmup=True)
array = AsyncEngineArray.from_args([engine_args])

async def classifier(engine: AsyncEmbeddingEngine): 
    async with engine:
        predictions, usage = await engine.classify(sentences=sentences)
    # or handle the async start / stop yourself.
    await engine.astart()
    predictions, usage = await engine.classify(sentences=sentences)
    await engine.astop()
asyncio.run(classifier(array["SamLowe/roberta-base-go_emotions"]))
```

### Infinity usage via the Python Client

Infinity has a generated client code for RestAPI client side usage.

If you want to call a remote infinity instance via RestAPI, install the following package locally:
```bash
pip install infinity_client
```

For more information, check out the Client Readme
https://github.com/michaelfeil/infinity/tree/main/libs/client_infinity/infinity_client

## Integrations:
- [Serverless deployments at Runpod](https://github.com/runpod-workers/worker-infinity-embedding)
- [Truefoundry Cognita](https://github.com/truefoundry/cognita)
- [Langchain example](https://github.com/langchain-ai/langchain)
- [imitater - A unified language model server built upon vllm and infinity.](https://github.com/the-seeds/imitater)
- [Dwarves Foundation: Deployment examples using Modal.com](https://github.com/dwarvesf/llm-hosting)
- [infiniflow/Ragflow](https://github.com/infiniflow/ragflow)
- [SAP Core AI](https://github.com/SAP-samples/btp-generative-ai-hub-use-cases/tree/main/10-byom-oss-llm-ai-core)
- [gpt_server - gpt_server is an open-source framework designed for production-level deployment of LLMs (Large Language Models) or Embeddings.](https://github.com/shell-nlp/gpt_server)
- [KubeAI: Kubernetes AI Operator for inferencing](https://github.com/substratusai/kubeai)
- [LangChain](https://python.langchain.com/docs/integrations/text_embedding/infinity)
- [Batched, modification of the Batching algoritm in Infinity](https://github.com/mixedbread-ai/batched)

## Documentation
View the docs at [https:///michaelfeil.github.io/infinity](https://michaelfeil.github.io/infinity) on how to get started.
After startup, the Swagger Ui will be available under `{url}:{port}/docs`, in this case `http://localhost:7997/docs`. You can also find a interactive preview here: https://infinity.modal.michaelfeil.eu/docs (and https://michaelfeil-infinity.hf.space/docs)

## Contribute and Develop

Install via Poetry 1.8.1, Python3.11 on Ubuntu 22.04
```bash
cd libs/infinity_emb
poetry install --extras all --with lint,test
```

To pass the CI:
```bash
cd libs/infinity_emb
make precommit
```

All contributions must be made in a way to be compatible with the MIT License of this repo. 

### Citation
```
@software{feil_2023_11630143,
  author       = {Feil, Michael},
  title        = {Infinity - To Embeddings and Beyond},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.11630143},
  url          = {https://doi.org/10.5281/zenodo.11630143}
}
```

### 💚 Current contributors <a name="Current contributors"></a>

<a href="https://github.com/michaelfeil/infinity/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=michaelfeil/infinity" />
</a>

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
[license-url]: https://github.com/michaelfeil/infinity/blob/main/LICENSE
[pepa-shield]: https://static.pepy.tech/badge/infinity-emb
[pepa-url]: https://www.pepy.tech/projects/infinity-emb
[codecov-shield]: https://codecov.io/gh/michaelfeil/infinity/branch/main/graph/badge.svg?token=NMVQY5QOFQ
[codecov-url]: https://codecov.io/gh/michaelfeil/infinity/branch/main
[ci-shield]: https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg
[ci-url]: https://github.com/michaelfeil/infinity/actions
