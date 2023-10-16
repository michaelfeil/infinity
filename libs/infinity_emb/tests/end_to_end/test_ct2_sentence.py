import asyncio
import random
import time

import numpy as np
import pytest
import torch
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from sentence_transformers import SentenceTransformer  # type: ignore

from infinity_emb import create_server
from infinity_emb.transformer.sentence_transformer import CT2SentenceTransformer
from infinity_emb.transformer.utils import InferenceEngine

PREFIX = "/v1_torch"
MODEL: str = pytest.DEFAULT_BERT_MODEL  # type: ignore

batch_size = 64 if torch.cuda.is_available() else 8

app = create_server(
    model_name_or_path=MODEL,
    batch_size=batch_size,
    url_prefix=PREFIX,
    engine=InferenceEngine.ctranslate2,
)


@pytest.fixture
def model_base() -> SentenceTransformer:
    return SentenceTransformer(MODEL)


@pytest.fixture()
async def client():
    async with AsyncClient(
        app=app, base_url="http://test", timeout=20
    ) as client, LifespanManager(app):
        yield client


def test_load_model(model_base):
    # this makes sure that the error below is not based on a slow download
    # or internal pytorch errors
    s = ["This is a test sentence."]
    e1 = model_base.encode(s)
    e2 = CT2SentenceTransformer(MODEL).encode(s)
    np.testing.assert_almost_equal(e1, e2, decimal=6)


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"].get("id", "") == MODEL
    assert isinstance(rdata["data"].get("stats"), dict)


@pytest.mark.anyio
async def test_embedding(client, model_base):
    possible_inputs = [
        ["This is a test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
    ]

    for inp in possible_inputs:
        response = await client.post(
            f"{PREFIX}/embeddings", json=dict(input=inp, model=MODEL)
        )
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        rdata = response.json()
        assert "data" in rdata and isinstance(rdata["data"], list)
        assert all("embedding" in d for d in rdata["data"])
        assert len(rdata["data"]) == len(inp)

        want_embeddings = model_base.encode(inp)

        for embedding, st_embedding in zip(rdata["data"], want_embeddings):
            np.testing.assert_almost_equal(embedding["embedding"], st_embedding)


@pytest.mark.performance
@pytest.mark.anyio
async def test_batch_embedding(client, get_sts_bechmark_dataset, model_base):
    sentences = []
    for d in get_sts_bechmark_dataset:
        for item in d:
            sentences.append(item.texts[0])
    random.shuffle(sentences)
    sentences = sentences[::2] if torch.cuda.is_available() else sentences[::16]
    # sentences = sentences[:batch_size*2]
    dummy_sentences = ["test" * 512] * batch_size

    async def _post_batch(inputs):
        return await client.post(
            f"{PREFIX}/embeddings", json=dict(input=inputs, model=MODEL)
        )

    response = await _post_batch(inputs=dummy_sentences)

    _request_size = int(batch_size * 1.5)
    tasks = [
        _post_batch(inputs=sentences[sl : sl + _request_size])
        for sl in range(0, len(sentences), _request_size)
    ]
    start = time.perf_counter()
    _responses = await asyncio.gather(*tasks)
    end = time.perf_counter()
    time_api = end - start

    responses = []
    for response in _responses:
        responses.extend(response.json()["data"])
    for i in range(len(responses)):
        responses[i] = responses[i]["embedding"]

    model_base.encode(
        dummy_sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    start = time.perf_counter()
    encodings = model_base.encode(sentences, batch_size=batch_size).tolist()
    end = time.perf_counter()
    time_st = end - start
    np.testing.assert_almost_equal(np.array(responses), np.array(encodings), decimal=6)
    assert time_api / time_st < 2.5
