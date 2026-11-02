import pytest
import torch
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
import numpy as np
from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

PREFIX = "/v1_sentence_transformers_colbert"
MODEL: str = pytest.DEFAULT_COLBERT_MODEL  # type: ignore[assignment]
batch_size = 64 if torch.cuda.is_available() else 8

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


# @pytest.fixture
# def model_base() -> SentenceTransformer:
#     # model = SentenceTransformer(MODEL)
#     # if model.device == "cuda":
#     #     model = model.to(torch.float16)
#     # return model
#     model


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url="http://test", timeout=20) as client, LifespanManager(
        app
    ):
        yield client


# def test_load_model(model_base):
#     # this makes sure that the error below is not based on a slow download
#     # or internal pytorch errors
#     model_base.encode(["This is a test sentence."])


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)


@pytest.mark.anyio
async def test_embedding(client):
    response = await client.post(
        f"{PREFIX}/embeddings", json=dict(input=["This is a test", "hi", "hi"], model=MODEL)
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert len(rdata["data"]) == 3
    # TODO: Check if start and end tokens should be embedded
    # TODO: Check if normalization is applied or should be applied?
    assert len(rdata["data"][0]["embedding"]) == 6  # This is a test -> 6 tokens
    assert len(rdata["data"][1]["embedding"]) == 3  # hi -> 3 tokens
    np.testing.assert_allclose(
        rdata["data"][1]["embedding"], rdata["data"][2]["embedding"], atol=5e-3
    )
