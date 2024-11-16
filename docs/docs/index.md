# [Infinity](https://github.com/michaelfeil/infinity)

Infinity is a high-throughput, low-latency REST API for serving vector embeddings, supporting all sentence-transformer models and frameworks. Infinity is developed under [MIT License](https://github.com/michaelfeil/infinity/blob/main/LICENSE). Infinity powers inference behind [Gradient.ai](https://gradient.ai) and other Embedding API providers.

## Why Infinity

Infinity provides the following features:

* **Deploy any model from MTEB**: deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
* **Fast inference backends**: The inference server is built on top of [torch](https://github.com/pytorch/pytorch), [optimum(onnx/tensorrt)](https://huggingface.co/docs/optimum/index) and [CTranslate2](https://github.com/OpenNMT/CTranslate2), using FlashAttention to get the most out of **CUDA**, **ROCM**, **CPU** or **MPS** device.
* **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your device as soon as ready. Similar max throughput on GPU as text-embeddings-inference.
* **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are identical to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
* **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API are aligned to [OpenAI's Embedding specs](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings). See below on how to get started.

## Getting started

Install `infinity_emb` via pip
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

### or launch the cli after the pip install
After your pip install, with your venv activate, you can run the CLI directly.
Check the `--help` command to get a description for all parameters.

```bash
infinity_emb --help
```

## Launch FAQ
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
  <summary>Using Langchain with Infinity</summary>
  Now available under # Python Integrations in the side panel.  
  ```
</details>


<details>
  <summary>Question not answered here?</summary>

  There is a Discussion section on the Github of Infinity:
  https://github.com/michaelfeil/infinity/discussions

</details>
  
