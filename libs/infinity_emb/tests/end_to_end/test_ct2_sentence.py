import sys

import numpy as np
import pytest
import torch
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine
from infinity_emb.transformer.embedder.ct2 import (
    CT2SentenceTransformer,
)

PREFIX = "/v1_torch"
MODEL: str = pytest.DEFAULT_BERT_MODEL  # type: ignore[assignment]

batch_size = 64 if torch.cuda.is_available() else 8

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL,
            batch_size=batch_size,
            engine=InferenceEngine.ctranslate2,
            device=Device.cpu,
        )
    ],
)


@pytest.fixture
def model_base() -> SentenceTransformer:
    return SentenceTransformer(MODEL, device="cpu")


@pytest.fixture()
async def client():
    async with AsyncClient(
        app=app, base_url="http://test", timeout=20
    ) as client, LifespanManager(app):
        yield client


@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
def test_load_model(model_base):
    # this makes sure that the error below is not based on a slow download
    # or internal pytorch errors
    s = ["This is a test sentence."]
    e1 = model_base.encode(s)
    e2 = CT2SentenceTransformer(
        engine_args=EngineArgs(
            model_name_or_path=MODEL, device="cpu", bettertransformer=False
        )
    ).encode(s)
    np.testing.assert_almost_equal(e1, e2, decimal=6)


@pytest.mark.anyio
@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)


@pytest.mark.anyio
@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
async def test_embedding(client, model_base, helpers):
    await helpers.embedding_verify(client, model_base, prefix=PREFIX, model_name=MODEL)


@pytest.mark.performance
@pytest.mark.anyio
@pytest.mark.skipif(sys.platform == "darwin", reason="Does not run on macOS")
async def test_batch_embedding(client, get_sts_bechmark_dataset, model_base, helpers):
    await helpers.util_batch_embedding(
        client=client,
        sts_bechmark_dataset=get_sts_bechmark_dataset,
        model_base=model_base,
        prefix=PREFIX,
        model_name=MODEL,
        batch_size=batch_size,
        downsample=32,
    )
