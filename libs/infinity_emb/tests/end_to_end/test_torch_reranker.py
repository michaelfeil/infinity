import asyncio
import itertools

import pytest
import torch
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from transformers import pipeline  # type: ignore[import-untyped]

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

PREFIX = "/v1_reranker"
MODEL: str = pytest.DEFAULT_RERANKER_MODEL  # type: ignore[assignment]
batch_size = 32 if torch.cuda.is_available() else 8

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL,
            batch_size=batch_size,
            engine=InferenceEngine.torch,
            device=Device.auto if not torch.backends.mps.is_available() else Device.cpu,
        )
    ],
)


@pytest.fixture
def model_base() -> pipeline:
    return pipeline(model=MODEL, task="text-classification")


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url="http://test", timeout=20) as client, LifespanManager(
        app
    ):
        yield client


def test_load_model(model_base):
    # this makes sure that the error below is not based on a slow download
    # or internal pytorch errors
    def predict(a, b):
        return model_base.predict({"text": a, "text_pair": b})["score"]

    assert (
        1.0
        > predict("Where is Paris?", "Paris is the capital of France.")
        > predict("Where is Paris?", "Berlin is the capital of Germany.")
        > predict("Where is Paris?", "You can now purchase my favorite dish")
        > 0.001
    )


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)


@pytest.mark.anyio
async def test_reranker(client, model_base, helpers):
    query = "Where is the Eiffel Tower located?"
    documents = [
        "My favorite dish is a hamburger.",
        "The Eiffel Tower is located in the United States.",
        "The Eiffel Tower is located in Paris, France",
        "The Eiffel Tower is located in the United Kingdom.",
        "I enjoiy flying to the moon.",
    ]
    response = await client.post(
        f"{PREFIX}/rerank",
        json={"model": MODEL, "query": query, "documents": documents},
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "model" in rdata
    assert "usage" in rdata
    rdata_results = rdata["results"]
    rdata_results = sorted(rdata_results, key=lambda x: x["index"], reverse=False)

    predictions = [model_base.predict({"text": query, "text_pair": doc}) for doc in documents]

    assert len(rdata_results) == len(predictions)
    for i, pred in enumerate(predictions):
        assert abs(rdata_results[i]["relevance_score"] - pred["score"]) < 0.01


@pytest.mark.anyio
async def test_reranker_top_n(client):
    async def test_inner(client, query: str, documents: list[str], return_docs: bool, top_n: int):
        response = await client.post(
            f"{PREFIX}/rerank",
            json={
                "model": MODEL,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": return_docs,
            },
        )
        assert response.status_code == 200
        rdata = response.json()
        rdata_results = rdata["results"]
        assert len(rdata_results) == min(top_n, len(documents))
        assert rdata_results[0]["index"] == documents.index(
            "The Eiffel Tower is located in Paris, France"
        )
        if return_docs:
            assert rdata_results[0]["document"] == "The Eiffel Tower is located in Paris, France"

        return True

    query = "Where is the Eiffel Tower located?"
    documents = [
        "The Eiffel Tower is located in the United States.",
        "The Eiffel Tower is located in Paris, France",
        "The Eiffel Tower is located in the United Kingdom.",
    ]

    cases = [
        test_inner(client, query, docs, return_docs, top_n)
        for docs in itertools.permutations(documents, r=len(documents))
        for top_n in [1, 2, 8000]
        for return_docs in [True, False]
    ]
    results = await asyncio.gather(*cases)
    assert all(results)


@pytest.mark.anyio
async def test_reranker_invalid_top_n(client):
    query = "Where is the Eiffel Tower located?"
    documents = [
        "The Eiffel Tower is located in Paris, France",
        "The Eiffel Tower is located in the United States.",
        "The Eiffel Tower is located in the United Kingdom.",
    ]
    response = await client.post(
        f"{PREFIX}/rerank",
        json={"model": MODEL, "query": query, "documents": documents, "top_n": -1},
    )
    assert response.status_code == 422

    response = await client.post(
        f"{PREFIX}/rerank",
        json={"model": MODEL, "query": query, "documents": documents, "top_n": 0},
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_reranker_cant_embed_or_classify(client):
    documents = [
        "The Eiffel Tower is located in Paris, France",
        "The Eiffel Tower is located in the United States.",
        "The Eiffel Tower is located in the United Kingdom.",
    ]
    response = await client.post(
        f"{PREFIX}/embeddings",
        json={"model": MODEL, "input": documents},
    )
    assert response.status_code == 400

    response = await client.post(
        f"{PREFIX}/classify",
        json={"model": MODEL, "input": documents},
    )
    assert response.status_code == 400
