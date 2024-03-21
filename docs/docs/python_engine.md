# Python Engine Integration

## Launching Embedding generation with Python

Use asynchronous programming in Python using `asyncio` for flexible and efficient embedding processing with Infinity. This advanced method allows for concurrent execution of different requests, making it ideal for high-throughput embedding generation.

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from infinity_emb.log_handler import logger
logger.setLevel(5) # Debug

# Define sentences for embedding
sentences = ["Embed this sentence via Infinity.", "Paris is in France."]
# Initialize the embedding engine with model specifications
engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(
        model_name_or_path="BAAI/bge-small-en-v1.5",
        engine="torch", 
        lengths_via_tokenize=True
    )
)

async def main(): 
    async with engine:  # Context manager initializes and terminates the engine
        
        job1 = asyncio.create_task(engine.embed(sentences=sentences))
        # submit a second job in parallel
        job2 = asyncio.create_task(engine.embed(sentences=["Hello world"]))
        # usage is total token count according to tokenizer.
        embeddings, usage = await job1
        embeddings2, usage2 = await job2
        # Embeddings are now available for use - they ran in the same batch.
    print(f"for {sentences}, generated embeddings {len(embeddings)} with tot_tokens={usage}")
asyncio.run(
    main()
)
```

## Reranker

Reranking gives you a score for similarity between a query and multiple documents. 
Use it in conjunction with a VectorDB+Embeddings, or as standalone for small amount of documents.
Please select a model from HuggingFace that can be loaded with AutoModelForSequenceClassification & has exactly one class for classification. E.g. "BAAI/bge-reranker-base". Further models are usually listed as `rerank` models on HuggingFace https://huggingface.co/models?pipeline_tag=text-classification&sort=trending&search=rerank. 

```python
import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

# Define your query and documents
query = "What is the python package infinity_emb?"
docs = [
    "This document is unrelated to the python package infinity_emb.", 
    "Paris is in France!",
    "infinity_emb is a package for generating sentence embeddings."
]

# Configure the reranking engine
engine_args = EngineArgs(model_name_or_path="BAAI/bge-reranker-base", engine="torch")
engine = AsyncEmbeddingEngine.from_args(engine_args)

async def main(): 
    async with engine:
        ranking, usage = await engine.rerank(query=query, docs=docs)
        # Display ranked documents
        print(list(zip(ranking, docs)))
asyncio.run(main())
```

You can launch the 
```bash
infinity_emb --model-name-or-path BAAI/bge-reranker-base
```

## Text Classification (Beta)

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