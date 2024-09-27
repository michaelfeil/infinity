import pytest
import torch
from asgi_lifespan import LifespanManager
from fastapi import status
from httpx import AsyncClient

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

PREFIX = "/v1_ct2"
MODEL: str = pytest.DEFAULT_VISION_MODEL  # type: ignore[assignment]
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


@pytest.fixture()
async def client():
    async with AsyncClient(
        app=app, base_url="http://test", timeout=20
    ) as client, LifespanManager(app):
        yield client


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)
    assert "image_embed" in rdata["data"][0]["capabilities"]


@pytest.mark.anyio
async def test_vision_single(client):
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    response = await client.post(
        f"{PREFIX}/embeddings_image",
        json={"model": MODEL, "input": image_url},
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "model" in rdata
    assert "usage" in rdata
    rdata_results = rdata["data"]
    assert rdata_results[0]["object"] == "embedding"
    assert len(rdata_results[0]["embedding"]) > 0


@pytest.mark.anyio
@pytest.mark.parametrize("no_of_images", [1, 5, 10])
async def test_vision_multiple(client, no_of_images):
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg"
    ] * no_of_images

    response = await client.post(
        f"{PREFIX}/embeddings_image",
        json={"model": MODEL, "input": image_urls},
    )
    assert response.status_code == 200
    rdata = response.json()
    rdata_results = rdata["data"]
    assert len(rdata_results) == no_of_images
    if no_of_images:
        assert "model" in rdata
        assert "usage" in rdata
        assert rdata_results[0]["object"] == "embedding"
        assert len(rdata_results[0]["embedding"]) > 0


@pytest.mark.anyio
async def test_vision_fail(client):
    image_url = "https://www.google.com/404"

    response = await client.post(
        f"{PREFIX}/embeddings_image",
        json={"model": MODEL, "input": image_url},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST

@pytest.mark.anyio
async def test_vision_empty(client):
    image_url_empty = []
    response = await client.post(
        f"{PREFIX}/embeddings_image",
        json={"model": MODEL, "input": image_url_empty},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY