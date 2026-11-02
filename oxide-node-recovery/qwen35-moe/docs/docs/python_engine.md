# Python Engine Integration

## Launching Embedding generation with Python

Use asynchronous programming in Python using `asyncio` for flexible and efficient embedding processing with Infinity. This advanced method allows for concurrent execution of different requests, making it ideal for high-throughput embedding generation.

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine
from infinity_emb.log_handler import logger
logger.setLevel(5) # Debug

# Define sentences for embedding
sentences = ["Embed this sentence via Infinity.", "Paris is in France."]
# Initialize the embedding engine with model specifications
array = AsyncEngineArray.from_args([
    EngineArgs(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        engine="torch", 
        lengths_via_tokenize=True
    )]
)

async def embed_image(engine: AsyncEmbeddingEngine): 
    await engine.astart()  # initializes  the engine
    job1 = asyncio.create_task(engine.embed(sentences=sentences))
    # submit a second job in parallel
    job2 = asyncio.create_task(engine.embed(sentences=["Hello world"]))
    # usage is total token count according to tokenizer.
    embeddings, usage = await job1
    embeddings2, usage2 = await job2
    # Embeddings are now available for use - they ran in the same batch.
    print(f"for {sentences}, generated embeddings {len(embeddings)} with tot_tokens={usage}")
    await engine.astop() 

asyncio.run(
    embed_image(array["BAAI/bge-small-en-v1.5"])
)
```

## Reranker

Reranking gives you a score for similarity between a query and multiple documents. 
Use it in conjunction with a VectorDB+Embeddings, or as standalone for small amount of documents.
Please select a model from huggingface that is a AutoModelForSequenceClassification with one class classification.

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

Example models:
- [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)
- [jinaai/jina-reranker-v1-turbo-en](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)

## CLIP models

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

Example models:
- [wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M)
- [jinaai/jina-clip-v1](https://huggingface.co/jinaai/jina-clip-v1) (requires `pip install timm`)
- Currently no support for pure vision models: nomic-ai/nomic-embed-vision-v1.5, ..


## Text Classification 

Use text classification with Infinity's `classify` feature, which allows for sentiment analysis, emotion detection, and more classification tasks.

```python
import asyncio
from infinity_emb import AsyncEngineArray, EngineArgs, AsyncEmbeddingEngine

sentences = ["This is awesome.", "I am bored."]
engine_args = EngineArgs(
    model_name_or_path = "SamLowe/roberta-base-go_emotions", 
    engine="torch", model_warmup=True)
array = AsyncEngineArray.from_args([engine_args])

async def classifier(): 
    async with engine:
        predictions, usage = await engine.classify(sentences=sentences)
    # or handle the async start / stop yourself.
    await engine.astart()
    predictions, usage = await engine.classify(sentences=sentences)
    await engine.astop()
asyncio.run(classifier(array["SamLowe/roberta-base-go_emotions"]))
```

Example models:
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)

