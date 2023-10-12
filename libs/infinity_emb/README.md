
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
![codecov](https://codecov.io/gh/michaelfeil/infinity/branch/main/graph/badge.svg?token=NMVQY5QOFQ)
![CI](https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg)

Embedding Inference Server - finding TGI for embeddings. Infinity is developed under MIT Licence - https://github.com/michaelfeil/infinity




## Why Infinity:
Infinity provides the following features:
- **Deploy virtually any SentenceTransformer** - deploy the model you know from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/)
- **Fast inference**: The inference server is built on top of [torch](https:) and [ctranslate2](https://github.com/OpenNMT/CTranslate2) under the hood, getting most out of your **CUDA** or **CPU** hardware.
- **Dynamic batching**: New embedding requests are queued while GPU is busy with the previous ones. New requests are squeezed intro your GPU/CPU as soon as ready. 
- **Correct and tested implementation**: Unit and end-to-end tested. Embeddings via infinity are identical to [SentenceTransformers](https://github.com/UKPLab/sentence-transformers/) (up to numerical precision). Lets API users create embeddings till infinity and beyond.
- **Easy to use**: The API is built on top of [FastAPI](https://fastapi.tiangolo.com/), [Swagger](https://swagger.io/) makes it fully documented. API specs are aligned to OpenAI. See below on how to get started.

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