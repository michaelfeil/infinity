from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, responses

from infinity_emb.fastapi_schemas.pymodels import (
    MultiModalOpenAIEmbedding,
    OpenAIEmbeddingResult,
)
from embed import BatchedInference
from infinity_emb import AsyncEmbeddingEngine, EngineArgs
import asyncio
import numpy as np
import torch

# benchmark settings
BATCH_SIZE = os.environ.get("BATCH_SIZE", 32)
BENCHMARK_NAME = os.environ.get("BENCHMARK_NAME", "")
USE_FASTEMBED = BENCHMARK_NAME == "fastembed"
USE_INFINITY = BENCHMARK_NAME == "infinity"
USE_EMBED = BENCHMARK_NAME == "embed"
DEVICE = os.environ.get("DEVICE", "cpu")

# load large for cuda, small for cpu. (benchmarking large on cpu takes too long)
MODEL_NAME = (
    "BAAI/bge-small-en-v1.5" if DEVICE == "cpu" else "BAAI/bge-large-en-v1.5"
)

# model loading

if USE_FASTEMBED:
    print(f"Using fastembed with model {MODEL_NAME}")
    model = TextEmbedding(MODEL_NAME, threads=None)
elif USE_EMBED:
    print("Using embed")
    assert BATCH_SIZE == 32
    register = BatchedInference(model_id=MODEL_NAME,
                                device=DEVICE,
                                engine="torch" if DEVICE.startswith("cuda") else "optimum")
elif USE_INFINITY:
    print("Using infinity")
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
    print("Using sentence transformer")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    if DEVICE == "cuda":
        model.half()


def encode_fastembed(text: list[str]):
    return list(model.passage_embed(text, batch_size=BATCH_SIZE))

def encode_embed(text: list[str]):
    return register.embed(sentences=text, model_id=0).result()[0]

def encode_sentence_transformer(text: list[str]):
    # not using multi_process_encode
    # as its too slower for len(texts) < 10000
    # and parallel interference
    with torch.inference_mode():
        return model.encode(text, batch_size=BATCH_SIZE)


async def encode_infinity(text: list[str]):
    return (await engine.embed(sentences=text))[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if USE_INFINITY:
        async with engine:
            yield
    elif USE_EMBED:
        yield
        register.stop()
    else:
        yield


app = FastAPI(
    description="start via `uvicorn simple_app:app --port 7997 --reload`",
    lifespan=lifespan,
)

if USE_INFINITY:
    @app.post(
        "/embeddings",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
    )
    async def embed(request: MultiModalOpenAIEmbedding) -> OpenAIEmbeddingResult:
        """the goal of this code is to write an as simple as possible server
        that can we rebuild by any other p
        """
        # dispatch async to that multiple requests can be handled at the same time
        sentences = request.input if isinstance(request.input, list) else [request.input]

        encoded = await encode_infinity(sentences)
        # response parsing
        return OpenAIEmbeddingResult.to_embeddings_response(
            encoded, MODEL_NAME, sum(len(t) for t in sentences)
        )
else:
    @app.post(
        "/embeddings",
        response_model=OpenAIEmbeddingResult,
        response_class=responses.ORJSONResponse,
    )
    def embed(request: MultiModalOpenAIEmbedding) -> OpenAIEmbeddingResult:
        """the goal of this code is to write an as simple as possible server
        that can we rebuild by any other p
        """
        # dispatch async to that multiple requests can be handled at the same time
        sentences = request.input if isinstance(request.input, list) else [request.input]

        if USE_EMBED:
            encoded = encode_embed(sentences)
        elif USE_FASTEMBED:
            encoded = encode_fastembed(sentences)
        else:
            encoded = encode_sentence_transformer(sentences)

        # response parsing
        return OpenAIEmbeddingResult.to_embeddings_response(
            encoded, MODEL_NAME, sum(len(t) for t in sentences)
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7997)
