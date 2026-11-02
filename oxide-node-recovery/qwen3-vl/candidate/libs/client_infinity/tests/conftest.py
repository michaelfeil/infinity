import requests
import time
import pytest

pytest.URL = "http://0.0.0.0:7994"


@pytest.fixture
def server_available() -> bool:
    i = 0

    while i < 10:
        try:
            requests.get(pytest.URL)
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1 + i)
            i += 1
