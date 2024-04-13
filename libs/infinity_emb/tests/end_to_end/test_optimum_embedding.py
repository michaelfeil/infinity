import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient
from sentence_transformers import SentenceTransformer  # type: ignore

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine

PREFIX = "/v1_optimum"
MODEL: str = (
    "michaelfeil/bge-small-en-v1.5"  #  pytest.DEFAULT_BERT_MODEL  # type: ignore
)

batch_size = 8

app = create_server(
    url_prefix=PREFIX,
    engine_args_list=[
        EngineArgs(
            model_name_or_path=MODEL,
            batch_size=batch_size,
            engine=InferenceEngine.optimum,
            device=Device.cpu,
        )
    ],
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


@pytest.mark.anyio
async def test_model_route(client):
    response = await client.get(f"{PREFIX}/models")
    assert response.status_code == 200
    rdata = response.json()
    assert "data" in rdata
    assert rdata["data"][0].get("id", "") == MODEL
    assert isinstance(rdata["data"][0].get("stats"), dict)


@pytest.mark.anyio
async def test_embedding(client, model_base, helpers):
    await helpers.embedding_verify(
        client, model_base, prefix=PREFIX, model_name=MODEL, decimal=2
    )


@pytest.mark.performance
@pytest.mark.anyio
async def test_batch_embedding(client, get_sts_bechmark_dataset, model_base, helpers):
    await helpers.util_batch_embedding(
        client=client,
        sts_bechmark_dataset=get_sts_bechmark_dataset,
        model_base=model_base,
        prefix=PREFIX,
        model_name=MODEL,
        batch_size=batch_size,
        downsample=16,
        decimal=1,
    )
