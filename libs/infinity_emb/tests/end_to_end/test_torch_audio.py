import base64

import numpy as np
import pytest
import torch
from asgi_lifespan import LifespanManager
from fastapi import status
from httpx import AsyncClient

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

PREFIX = "/v1_audio"
MODEL: str = pytest.DEFAULT_AUDIO_MODEL  # type: ignore[assignment]
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
    async with AsyncClient(app=app, base_url="http://test", timeout=20) as client, LifespanManager(
        app
    ):
        yield client


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)
    assert "audio_embed" in rdata["data"][0]["capabilities"]


@pytest.mark.anyio
async def test_audio_single(client):
    audio_url = pytest.AUDIO_SAMPLE_URL

    response = await client.post(
        f"{PREFIX}/embeddings_audio",
        json={"model": MODEL, "input": audio_url},
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "model" in rdata
    assert "usage" in rdata
    rdata_results = rdata["data"]
    assert rdata_results[0]["object"] == "embedding"
    assert len(rdata_results[0]["embedding"]) > 0


@pytest.mark.anyio
async def test_audio_single_text_only(client):
    text = "a sound of a at"

    response = await client.post(
        f"{PREFIX}/embeddings",
        json={"model": MODEL, "input": text},
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "model" in rdata
    assert "usage" in rdata
    rdata_results = rdata["data"]
    assert rdata_results[0]["object"] == "embedding"
    assert len(rdata_results[0]["embedding"]) > 0


@pytest.mark.anyio
async def test_meta(client, helpers):
    audio_url = pytest.AUDIO_SAMPLE_URL

    text_input = ["a beep", "a horse", "a fish"]
    audio_input = [audio_url]
    response_text = await client.post(
        f"{PREFIX}/embeddings",
        json={"model": MODEL, "input": text_input},
    )
    response_audio = await client.post(
        f"{PREFIX}/embeddings_audio",
        json={"model": MODEL, "input": audio_input},
    )

    assert response_text.status_code == 200
    assert response_audio.status_code == 200

    rdata_text = response_text.json()
    rdata_results_text = rdata_text["data"]

    rdata_audio = response_audio.json()
    rdata_results_audio = rdata_audio["data"]

    embeddings_audio_beep = rdata_results_audio[0]["embedding"]
    embeddings_text_beep = rdata_results_text[0]["embedding"]
    embeddings_text_horse = rdata_results_text[1]["embedding"]
    embeddings_text_fish = rdata_results_text[2]["embedding"]

    assert helpers.cosine_similarity(
        embeddings_audio_beep, embeddings_text_beep
    ) > helpers.cosine_similarity(embeddings_audio_beep, embeddings_text_fish)
    assert helpers.cosine_similarity(
        embeddings_audio_beep, embeddings_text_beep
    ) > helpers.cosine_similarity(embeddings_audio_beep, embeddings_text_horse)


@pytest.mark.anyio
async def test_audio_multiple(client):
    for route in [f"{PREFIX}/embeddings_audio", f"{PREFIX}/embeddings"]:
        for no_of_audios in [1, 3]:
            audio_urls = [pytest.AUDIO_SAMPLE_URL] * no_of_audios
            for _ in range(3):
                response = await client.post(
                    route,
                    json={
                        "model": MODEL,
                        "input": audio_urls,
                        "modality": "audio",
                    },
                )
                if response.status_code == 200:
                    break
            assert response.status_code == 200
            rdata = response.json()
            rdata_results = rdata["data"]
            assert len(rdata_results) == no_of_audios
            assert "model" in rdata
            assert "usage" in rdata
            assert rdata_results[0]["object"] == "embedding"
            assert len(rdata_results[0]["embedding"]) > 0


@pytest.mark.anyio
async def test_audio_base64(client, audio_sample):
    bytes_downloaded = audio_sample[0].content
    base_64_audio = base64.b64encode(bytes_downloaded).decode("utf-8")

    response = await client.post(
        f"{PREFIX}/embeddings_audio",
        json={
            "model": MODEL,
            "input": [
                "data:audio/wav;base64," + base_64_audio,
                audio_sample[1],
            ],
        },
    )
    assert response.status_code == 200
    rdata = response.json()
    assert "model" in rdata
    assert "usage" in rdata
    rdata_results = rdata["data"]
    assert rdata_results[0]["object"] == "embedding"
    assert len(rdata_results[0]["embedding"]) > 0

    np.testing.assert_array_almost_equal(
        rdata_results[0]["embedding"], rdata_results[1]["embedding"], decimal=4
    )


@pytest.mark.anyio
async def test_audio_base64_fail(client):
    base_64_audio = "somethingsomething"

    response = await client.post(
        f"{PREFIX}/embeddings_audio",
        json={
            "model": MODEL,
            "input": [
                "data:audio/wav;base64," + base_64_audio,
                pytest.AUDIO_SAMPLE_URL,
            ],
        },
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    for route in [f"{PREFIX}/embeddings_audio", f"{PREFIX}/embeddings"]:
        audio_url = "https://www.google.com/404"

        response = await client.post(
            route,
            json={
                "model": MODEL,
                "input": audio_url,
                "modality": "audio",
            },
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    audio_url_empty = []

    response_empty = await client.post(
        f"{PREFIX}/embeddings",
        json={
            "model": MODEL,
            "input": audio_url_empty,
            "modality": "audio",
        },
    )
    assert response_empty.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    response_unsupported = await client.post(
        f"{PREFIX}/classify",
        json={"model": MODEL, "input": ["test"]},
    )
    assert response_unsupported.status_code == status.HTTP_400_BAD_REQUEST
