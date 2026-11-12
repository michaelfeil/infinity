import asyncio
from http import HTTPStatus
from importlib.metadata import requires

import httpx
import pytest
from infinity_client import AuthenticatedClient, Client
from infinity_client.api.default import models
from packaging.requirements import Requirement


def test_package_supports_httpx_028():
    dependencies = [Requirement(value) for value in requires("infinity_client")]
    httpx_dependency = next(value for value in dependencies if value.name == "httpx")
    assert "0.28.1" in httpx_dependency.specifier


@pytest.fixture(params=[False, True], ids=["anonymous", "authenticated"])
def client(request):
    authenticated = request.param

    def respond(http_request):
        assert http_request.method == "GET"
        assert http_request.url == "https://infinity.test/models"
        expected_auth = "Bearer test-token" if authenticated else None
        assert http_request.headers.get("Authorization") == expected_auth
        return httpx.Response(200, json={"object": "list", "data": []})

    kwargs = {
        "base_url": "https://infinity.test",
        "httpx_args": {"transport": httpx.MockTransport(respond)},
    }
    if authenticated:
        return AuthenticatedClient(token="test-token", **kwargs)
    return Client(**kwargs)


def test_sync_models_response(client):
    with client:
        response = models.sync_detailed(client=client)
    assert response.status_code == HTTPStatus.OK
    assert response.parsed.object_ == "list"
    assert response.parsed.data == []


def test_async_models_response(client):
    async def get_models():
        async with client:
            return await models.asyncio_detailed(client=client)

    response = asyncio.run(get_models())
    assert response.status_code == HTTPStatus.OK
    assert response.parsed.object_ == "list"
    assert response.parsed.data == []
