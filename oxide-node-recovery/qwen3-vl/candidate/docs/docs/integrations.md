# Python Integrations

## Cognita
https://github.com/truefoundry/cognita

## RagFlow
https://github.com/infiniflow/ragflow

## Langchain (from running server)
Infinity has an official integration into `pip install langchain>=0.342`. 
You can find more documentation on that here:
https://python.langchain.com/docs/integrations/text_embedding/infinity

### Langchain integration with running infinity API server
This code snippet assumes you have a server running at `http://localhost:7997/v1`
```python
from langchain.embeddings.infinity import InfinityEmbeddings
from langchain.docstore.document import Document

documents = [Document(page_content="Hello world!", metadata={"source": "unknown"})]

emb_model = InfinityEmbeddings(model="BAAI/bge-small", infinity_api_url="http://localhost:7997/v1")
print(emb_model.embed_documents([doc.page_content for doc in documents]))
```

### Langchain integration without running infinity API server and Python Inference.
```python
from langchain.embeddings.infinity import InfinityEmbeddings
from langchain.docstore.document import Document

embeddings = InfinityEmbeddingsLocal(
    model="sentence-transformers/all-MiniLM-L6-v2",
    # revision
    revision=None,
    # best to keep at 32
    batch_size=32,
    # for AMD/Nvidia GPUs via torch
    device="cuda",
    # warm up model before execution
)
documents = [Document(page_content="Hello world!", metadata={"source": "unknown"})]

# important: use engine inside of `async with` statement to start/stop the batching engine.
async with embeddings:
    # avoid closing and starting the engine often.
    # rather keep it running.
    # you may call `await embeddings.__aenter__()` and `__aexit__()
    # if you are sure when to manually start/stop execution` in a more granular way
    documents_embedded = await embeddings.aembed_documents(documents)
    query_result = await embeddings.aembed_query(query)
    print("embeddings created successful")
print(documents_embedded, query_result)
```

## LLama-Index
Details regarding LLama-Index integration will be announced soon - Contributions welcome.