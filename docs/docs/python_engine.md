# Python Engine 

## Launch via Python

You can use in a async context with asyncio. 
This gives you most flexibility, but is a bit more advanced.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

sentences = [
    "Embed this is sentence via Infinity.",
    "Paris is in France."
]
engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(model_name_or_path = "BAAI/bge-small-en-v1.5", engine="torch")
)

async def main(): 
    async with engine: 
        # entering context: engine starts with engine.astart()
        embeddings, usage = await engine.embed(
            sentences=sentences)
    # engine stops with engine.astop()
asyncio.run(main())
```

# ReRanker

Reranking gives you a score for similarity between a query and multiple documents. 
Use it in conjunction with a VectorDB+Embeddings, or as standalone for small amount of documents.
Please select a model from huggingface that is a AutoModelForSequenceClassification with one class classification.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
query = "What is the python package infinity_emb?"
docs = [
    "This is a document not related to the python package infinity_emb, hence...", 
    "Paris is in France!",
    "infinity_emb is a package for sentence embeddings and rerankings using transformer models in Python!"
]
engine_args = EngineArgs(
    model_name_or_path = "BAAI/bge-reranker-base", 
    engine="torch")

engine = AsyncEmbeddingEngine.from_args(engine_args)
async def main(): 
    async with engine:
        ranking, usage = await engine.rerank(
            query=query, docs=docs)
        print(list(zip(ranking, docs)))
asyncio.run(main())
```

# Text-Classification (Beta)
  
```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

sentences = ["This is awesome.", "I am bored."]
engine_args = EngineArgs(
    model_name_or_path = "SamLowe/roberta-base-go_emotions", 
    engine="torch", model_warmup=True)
engine = AsyncEmbeddingEngine.from_args(engine_args)
async def main(): 
    async with engine:
        predictions, usage = await engine.classify(
            sentences=sentences)
        return predictions, usage
asyncio.run(main())
```