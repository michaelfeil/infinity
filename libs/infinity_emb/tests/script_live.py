import json
import timeit

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LIVE_URL = "http://localhost:8001/v1"


def embedding_live_performance():
    sample = [f"Test count {i} {(list(range(i % (384))))} " for i in range(2048)]

    json_d = json.dumps({"input": sample, "model": "model"})
    session = requests.Session()
    req = session.get(f"{LIVE_URL}/models")
    assert req.status_code == 200

    batch_size = req.json()["data"]["stats"]["batch_size"]
    model_name = req.json()["data"]["id"]
    print(f"batch_size is {batch_size}")
    model = SentenceTransformer(model_name_or_path=model_name)

    def local(data: str):
        enc = model.encode(data, batch_size=batch_size)
        assert len(enc) == len(data)
        return enc

    def remote(json_data: bytes):
        req = session.post(f"{LIVE_URL}/embeddings", data=json_data)
        assert req.status_code == 200
        return req

    local_resp = local(sample)
    remote_resp = [d["embedding"] for d in remote(json_d).json()["data"]]
    np.testing.assert_almost_equal(local_resp, remote_resp, 6)
    print("Both methods provide the identical output.")

    print("Measuring latency via SentenceTransformers")
    latency_st = timeit.timeit("local(sample)", number=10, globals=locals())
    print("SentenceTransformers latency: ", latency_st)
    model = None

    print("Measuring latency via requests")
    latency_request = timeit.timeit("remote(json_d)", number=10, globals=locals())
    print(f"Request latency: {latency_request}")

    assert latency_st * 1.1 > latency_request


if __name__ == "__main__":
    embedding_live_performance()
