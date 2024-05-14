import asyncio
import json
import pathlib
import random
import sys
import time
from unittest import TestCase
from uuid import uuid4

import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import InferenceEngine

PREFIX = ""
MODEL_NAME = str(uuid4())
MODEL_NAME_2 = str(uuid4())
BATCH_SIZE = 16

PATH_OPENAPI = pathlib.Path(__file__).parent.parent.parent.parent.parent.joinpath(
    "docs", "assets", "openapi.json"
)

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL_NAME,
            batch_size=BATCH_SIZE,
            engine=InferenceEngine.debugengine,
        ),
        EngineArgs(
            model_name_or_path=MODEL_NAME_2,
            batch_size=BATCH_SIZE,
            engine=InferenceEngine.debugengine,
        ),
    ],
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
    assert rdata["data"][0].get("id", "") == MODEL_NAME
    assert rdata["data"][1].get("id", "") == MODEL_NAME_2
    assert isinstance(rdata["data"][0].get("stats"), dict)

    # ready test
    respnse_health = await client.get("/health")
    assert respnse_health.status_code == 200
    assert "unix" in respnse_health.json()


@pytest.mark.parametrize("model_name", [MODEL_NAME, MODEL_NAME_2])
@pytest.mark.anyio
async def test_embedding_max_length(client, model_name):
    # TOO long
    input = "%_" * 4097 * 15
    response = await client.post(
        f"{PREFIX}/embeddings", json=dict(input=input, model=model_name)
    )
    assert response.status_code == 422, f"{response.status_code}, {response.text}"
    # works
    input = "%_" * 4096 * 15
    response = await client.post(
        f"{PREFIX}/embeddings", json=dict(input=input, model=model_name)
    )
    assert response.status_code == 200, f"{response.status_code}, {response.text}"
    assert response.json()["model"] == model_name


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


@pytest.mark.skipif(sys.platform != "linux", reason="Only check on linux")
@pytest.mark.skipif(not PATH_OPENAPI.exists(), reason="openapi.json does not exist")
@pytest.mark.anyio
async def test_openapi_same_as_docs_file(client):
    assert (
        PATH_OPENAPI.exists()
    ), f"openapi.json file does not exist, it should be in {PATH_OPENAPI.resolve()}"

    openapi_req = await client.get("/openapi.json")
    assert openapi_req.status_code == 200
    openapi_json = openapi_req.json()
    openapi_json_expected = json.loads(PATH_OPENAPI.read_text())
    openapi_json["info"].pop("version")
    openapi_json_expected["info"].pop("version")
    tc = TestCase()
    tc.maxDiff = 100000
    assert openapi_json["openapi"] == openapi_json_expected["openapi"]
    tc.assertDictEqual(openapi_json["info"], openapi_json_expected["info"])
    tc.assertDictEqual(openapi_json["paths"], openapi_json_expected["paths"])
    # tc.assertDictEqual(openapi_json["components"], openapi_json_expected["components"])
