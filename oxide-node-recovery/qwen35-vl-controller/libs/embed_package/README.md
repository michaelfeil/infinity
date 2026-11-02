# embed
A stable, blazing fast and easy-to-use inference library with a focus on a sync-to-async API

[![ci][ci-shield]][ci-url]
[![Downloads][pepa-shield]][pepa-url]

## Installation
```bash
pip install embed
```

## Why embed?

Embed makes it easy to load any embedding, classification and reranking models from Huggingface. 
It leverages [Infinity](https://github.com/michaelfeil/infinity) as backend for async computation, batching, and Flash-Attention-2.

![CPU Benchmark Diagram](docs/l4_cpu.png)
Benchmarking on an Nvidia-L4 instance. Note: CPU uses bert-small, CUDA uses Bert-large. [Methodology](https://michaelfeil.eu/infinity/0.0.51/benchmarking/).

```python
from embed import BatchedInference
from concurrent.futures import Future

# Run any model
register = BatchedInference(
    model_id=[
        # sentence-embeddings
        "michaelfeil/bge-small-en-v1.5",
        # sentence-embeddings and image-embeddings
        "jinaai/jina-clip-v1",
        # classification models
        "philschmid/tiny-bert-sst2-distilled",
        # rerankers
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
    ],
    # engine to `torch` or `optimum`
    engine="torch",
    # device `cuda` (Nvidia/AMD) or `cpu`
    device="cpu",
)

sentences = ["Paris is in France.", "Berlin is in Germany.", "A image of two cats."]
images = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
question = "Where is Paris?"

future: "Future" = register.embed(
    sentences=sentences, model_id="michaelfeil/bge-small-en-v1.5"
)
future.result()
register.rerank(
    query=question, docs=sentences, model_id="mixedbread-ai/mxbai-rerank-xsmall-v1"
)
register.classify(model_id="philschmid/tiny-bert-sst2-distilled", sentences=sentences)
register.image_embed(model_id="jinaai/jina-clip-v1", images=images)

# manually stop the register upon termination to free model memory.
register.stop()
```

All functions return `Futures(vector_embedding, token_usage)`, enables you to `wait` for them and removes batching logic from your code.

```python
>>> embedding_fut = register.embed(sentences=sentences, model_id="michaelfeil/bge-small-en-v1.5")
>>> print(embedding_fut)
<Future at 0x7fa0e97e8a60 state=pending>
>>> time.sleep(1) and print(embedding_fut)
<Future at 0x7fa0e97e9c30 state=finished returned tuple>
>>> embedding_fut.result()
([array([-3.35943862e-03, ..., -3.22808176e-02], dtype=float32)], 19)
```

# Licence and Contributions
embed is licensed as MIT. All contribrutions need to adhere to the MIT License. Contributions are welcome.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/michaelfeil/embed.svg?style=for-the-badge
[contributors-url]: https://github.com/michaelfeil/embed/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/michaelfeil/embed.svg?style=for-the-badge
[forks-url]: https://github.com/michaelfeil/embed/network/members
[stars-shield]: https://img.shields.io/github/stars/michaelfeil/embed.svg?style=for-the-badge
[stars-url]: https://github.com/michaelfeil/embed/stargazers
[issues-shield]: https://img.shields.io/github/issues/michaelfeil/embed.svg?style=for-the-badge
[issues-url]: https://github.com/michaelfeil/embed/issues
[license-shield]: https://img.shields.io/github/license/michaelfeil/embed.svg?style=for-the-badge
[license-url]: https://github.com/michaelfeil/embed/blob/master/LICENSE.txt
[pepa-shield]: https://static.pepy.tech/badge/embed
[pepa-url]: https://www.pepy.tech/projects/embed
[ci-shield]: https://github.com/michaelfeil/infinity/actions/workflows/ci.yaml/badge.svg
[ci-url]: https://github.com/michaelfeil/infinity/actions
