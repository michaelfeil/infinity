from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from infinity_emb.fastapi_schemas.pymodels import (
    OpenAIEmbeddingInput,
    OpenAIEmbeddingResult,
)
from infinity_emb.fastapi_schemas.convert import list_embeddings_to_response
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
import asyncio
import numpy as np

# benchmark settings
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = os.environ.get("BATCH_SIZE", 32)
BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "infinity")
DEVICE = os.environ.get("DEVICE", "cpu")

# load the right settings
USE_FASTEMBED = BENCHMARK_NAME == "fastembed"
USE_INFINITY = BENCHMARK_NAME == "infinity"

if USE_FASTEMBED:
    model = TextEmbedding(MODEL_NAME, threads=None)
elif USE_INFINITY:
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path=MODEL_NAME,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            lengths_via_tokenize=False,
            model_warmup=True,
            engine="torch" if DEVICE.startswith("cuda") else "optimum",
        )
    )
else:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)


def encode_fastembed(text: list[str]):
    return model.query_embed(text, batch_size=BATCH_SIZE)


def encode_sentence_transformer(text: list[str]):
    # not using multi_process_encode
    # as its too slower for len(texts) < 10000 
    # and parallel interference
    return model.encode(text, batch_size=BATCH_SIZE)


async def encode_infinity(text: list[str]):
    return (await engine.embed(text))[0]

@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_INFINITY:
        async with engine:
            yield
    else:
        yield


app = FastAPI(
    description="start via `uvicorn simple_app:app --port 7997 --reload`",
    lifespan=lifespan,
)


@app.post("/embeddings")
async def embed(request: OpenAIEmbeddingInput) -> OpenAIEmbeddingResult:
    """the goal of this code is to write an as simple as possible server
    that can we rebuild by any other p
    """
    # dispatch async to that multiple requests can be handled at the same time
    sentences = request.input if isinstance(request.input, list) else [request.input]

    if USE_FASTEMBED:
        encoded = await asyncio.to_thread(encode_fastembed, sentences)
    elif USE_INFINITY:
        encoded = await encode_infinity(sentences)
    else:
        encoded = await asyncio.to_thread(encode_sentence_transformer, sentences)

    # response parsing
    response = list_embeddings_to_response(
        np.array(encoded), MODEL_NAME, sum(len(t) for t in sentences)
    )
    return OpenAIEmbeddingResult(**response)
