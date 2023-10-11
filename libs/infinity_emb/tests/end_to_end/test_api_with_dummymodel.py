import asyncio
import random
import time
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from infinity_emb import create_server
from infinity_emb.inference.models import InferenceEngine

PREFIX = "/v2"
MODEL_NAME = str(uuid4())
BATCH_SIZE = 16
app = create_server(
    model_name_or_path=MODEL_NAME,
    batch_size=BATCH_SIZE,
    url_prefix=PREFIX,
    engine=InferenceEngine.debugengine,
)


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client, LifespanManager(
        app
    ):
        yield client


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"].get("id", "") == MODEL_NAME
    assert isinstance(rdata["data"].get("stats"), dict)

    # ready test
    response = await client.get("/ready")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_embedding(client):
    possible_inputs = [
        ["This is a test sentence."],
        ["This is a test sentence.", "This is another test sentence."],
    ]
    for inp in possible_inputs:
        response = await client.post(
            f"{PREFIX}/embeddings", json=dict(input=inp, model=MODEL_NAME)
        )
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        rdata = response.json()
        assert "data" in rdata and isinstance(rdata["data"], list)
        assert all("embedding" in d for d in rdata["data"])
        assert len(rdata["data"]) == len(inp)
        for embedding, sentence in zip(rdata["data"], inp):
            assert len(sentence) == embedding["embedding"][0]


@pytest.mark.anyio
async def test_batch_embedding(client, get_sts_bechmark_dataset):
    sentences = []
    for d in get_sts_bechmark_dataset:
        for item in d:
            sentences.append(item.texts[0])
    random.shuffle(sentences)
    sentences = sentences

    async def _post_batch(inputs):
        return await client.post(
            f"{PREFIX}/embeddings", json=dict(input=inputs, model=MODEL_NAME)
        )

    _request_size = BATCH_SIZE // 2
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

    print(time_api)
