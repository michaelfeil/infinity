from infinity_client import Client
import requests
from infinity_client.models import OpenAIModelInfo
from infinity_client.api.default import models, health
from infinity_client.types import Response
import pytest


def test_model_info(server_available):
    client = Client(base_url=pytest.URL)

    with client as client:
        model_info: OpenAIModelInfo = models.sync(client=client)
        # or if you need more info (e.g. status_code)
        response: Response[OpenAIModelInfo] = models.sync_detailed(client=client)
