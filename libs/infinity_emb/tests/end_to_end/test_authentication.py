import pytest
from asgi_lifespan import LifespanManager
from httpx import AsyncClient

from infinity_emb import create_server
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import InferenceEngine

PREFIX = ""
MODEL_NAME = "dummy/model1"
MODEL_NAME_2 = "dummy/model2"
BATCH_SIZE = 16
API_KEY = "dummy-password"

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
    api_key=API_KEY,
)


@pytest.fixture()
async def client():
    async with AsyncClient(app=app, base_url="http://test") as client, LifespanManager(
        app
    ):
        yield client


@pytest.mark.anyio
async def test_authentication(client):
    for method, route, payload in [
        ["GET", "/models", None],
        ["POST", "/embeddings", {"model": MODEL_NAME, "input": "sentence"}],
        [
            "POST",
            "/rerank",
            {
                "model": MODEL_NAME,
                "query": "Where is Munich?",
                "documents": ["Munich is in Germany."],
            },
        ],
    ]:
        for authenticated in [False, True]:
            headers = {"Authorization": f"Bearer {API_KEY}"} if authenticated else {}
        response = await client.request(
            method,
            route,
            headers=headers,
            json=payload,
        )
        assert (response).status_code in (
            [200, 400] if authenticated else [401]
        ), f"route: {route}, payload: {payload}, response: {response.json()}"
