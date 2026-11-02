# Embed (package)
A stable, blazing fast and easy-to-use inference library with a focus on a sync-to-async API

## Installation
```bash
pip install embed
```

## Why embed?

Embed makes it easy to load any embedding, classification and reranking models from Huggingface. 
It's a wrapper around `SyncEngine` from `infinity_emb`, but updated less frequently and disentrangles pypy and docker releases of infinity.

```python
from embed import BatchedInference

# Run any model
register = BatchedInference(model_id=[
  # sentence-embeddings
  "michaelfeil/bge-small-en-v1.5",
  # sentence-embeddings and image-embeddings
  "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
  # classification models
  "philschmid/tiny-bert-sst2-distilled",
  # rerankers
  "mixedbread-ai/mxbai-rerank-xsmall-v1"
],
# engine to `torch` or `optimum`
engine="torch",
# device `cuda` (Nvidia/AMD) or `cpu`
device="cpu"
)

sentences = ["Paris is in France.", "Berlin is in Germany.", "I love SF"]
images = ["http://images.cocodataset.org/val2017/000000039769.jpg"]
question = "Where is Paris?"

register.embed(sentences=sentences, model_id="michaelfeil/bge-small-en-v1.5")
register.rerank(query=question, docs=sentences, model_id="mixedbread-ai/mxbai-rerank-xsmall-v1")
register.classify(model_id="philschmid/tiny-bert-sst2-distilled", sentences=sentences)
register.image_embed(model_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", images=images)

# Always manually stop the register upon termination to free model memory.
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
