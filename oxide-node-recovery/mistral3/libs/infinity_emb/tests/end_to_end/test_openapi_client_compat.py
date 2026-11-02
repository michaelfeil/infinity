# type: ignore

import base64

import numpy as np
import pytest
import requests
from asgi_lifespan import LifespanManager
import time
from httpx import AsyncClient
from openai import APIConnectionError, AsyncOpenAI

from infinity_emb import create_server
from infinity_emb.args import EngineArgs

PREFIX = ""
baseurl = "http://openaicompat"
batch_size = 8
api_key = "some_dummy_key"

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=pytest.DEFAULT_AUDIO_MODEL,
            batch_size=batch_size,
            device="cpu",
        ),
        EngineArgs(
            model_name_or_path=pytest.DEFAULT_IMAGE_MODEL,
            batch_size=batch_size,
        ),
        EngineArgs(
            model_name_or_path=pytest.DEFAULT_BERT_MODEL,
            batch_size=batch_size,
            device="cpu",
        ),
    ],
    api_key=api_key,
)


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url=baseurl, timeout=20) as client, LifespanManager(app):
        yield client


def url_to_base64(url, modality="image"):
    """small helper to convert url to base64 without server requiring access to the url"""
    for i in range(3):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        raise Exception(f"Failed to download {url}")
    response.raise_for_status()
    base64_encoded = base64.b64encode(response.content).decode("utf-8")
    mimetype = f"{modality}/{url.split('.')[-1]}"
    return f"data:{mimetype};base64,{base64_encoded}"


@pytest.mark.anyio
async def test_openai(client: AsyncClient):
    client_oai = AsyncOpenAI(api_key=api_key, base_url=baseurl, http_client=client)

    async with client_oai:
        # test audio
        emb1_audio_from_text = await client_oai.embeddings.create(
            model=pytest.DEFAULT_AUDIO_MODEL,
            input=[
                "the sound of a beep",
                "the sound of a cat",
                "the sound of a dog",
                "the sound of a bird",
            ],
            encoding_format="float",
            extra_body={"modality": "text"},
        )
        emb1_audio = await client_oai.embeddings.create(
            model=pytest.DEFAULT_AUDIO_MODEL,
            input=[url_to_base64(pytest.AUDIO_SAMPLE_URL, "audio")],
            encoding_format="float",
            extra_body={"modality": "audio"},
        )
        emb1_1_audio = await client_oai.embeddings.create(
            model=pytest.DEFAULT_AUDIO_MODEL,
            input=[pytest.AUDIO_SAMPLE_URL],
            encoding_format="float",
            extra_body={"modality": "audio"},
        )
        # test: image
        emb_1_image_from_text = await client_oai.embeddings.create(
            model=pytest.DEFAULT_IMAGE_MODEL,
            input=["a cat", "a dog", "a bird"],
            encoding_format="float",
            extra_body={"modality": "text"},
        )
        emb_1_image = await client_oai.embeddings.create(
            model=pytest.DEFAULT_IMAGE_MODEL,
            input=[url_to_base64(pytest.IMAGE_SAMPLE_URL, "image")],  # image is a cat
            encoding_format="float",
            extra_body={"modality": "image"},
        )
        emb_1_1_image = await client_oai.embeddings.create(
            model=pytest.DEFAULT_IMAGE_MODEL,
            input=[pytest.IMAGE_SAMPLE_URL],
            encoding_format="float",
            extra_body={"modality": "image"},
        )

        # test: text
        emb_1_text = await client_oai.embeddings.create(
            model=pytest.DEFAULT_BERT_MODEL,
            input=["a cat", "a cat", "a bird"],
            encoding_format="float",
            extra_body={"modality": "text"},
        )

        # test: text matryoshka
        emb_1_text_matryoshka_dim = await client_oai.embeddings.create(
            model=pytest.DEFAULT_BERT_MODEL,
            input=["a cat", "a cat", "a bird"],
            encoding_format="float",
            dimensions=64,
            extra_body={"modality": "text"},
        )
        assert len(emb_1_text_matryoshka_dim.data[0].embedding) == 64

    # test AUDIO: cosine distance of beep to cat and dog
    np.testing.assert_allclose(
        emb1_audio.data[0].embedding, emb1_1_audio.data[0].embedding, rtol=1e-4, atol=1e-4
    )
    assert all(
        np.dot(emb1_audio.data[0].embedding, emb1_audio_from_text.data[0].embedding)
        > np.dot(emb1_audio.data[0].embedding, emb1_audio_from_text.data[i].embedding)
        for i in range(1, 4)
    )

    # test IMAGE: cosine distance of cat to dog and bird
    np.testing.assert_allclose(
        emb_1_image.data[0].embedding, emb_1_1_image.data[0].embedding, rtol=1e-4, atol=1e-4
    )
    assert all(
        np.dot(emb_1_image.data[0].embedding, emb_1_image_from_text.data[0].embedding)
        > np.dot(emb_1_image.data[0].embedding, emb_1_image_from_text.data[i].embedding)
        for i in range(1, 3)
    )

    # test TEXT: cosine distance of cat to dog and bird
    np.testing.assert_allclose(
        emb_1_text.data[0].embedding, emb_1_text.data[1].embedding, rtol=1e-4, atol=1e-4
    )

    # wrong key
    with pytest.raises(APIConnectionError):
        client_oai = AsyncOpenAI(api_key="some_wrong", base_url=baseurl, http_client=client)
        async with client_oai:
            await client_oai.embeddings.create(
                model=pytest.DEFAULT_AUDIO_MODEL,
                input=[pytest.AUDIO_SAMPLE_URL],
                encoding_format="float",
                extra_body={"modality": "audio"},
            )
