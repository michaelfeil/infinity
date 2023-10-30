import concurrent.futures
import json
import timeit
from functools import partial

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

LIVE_URL = "http://localhost:8001/v1"


def embedding_live_performance():
    tp = concurrent.futures.ThreadPoolExecutor()
    sample = [f"Test count {i} {(list(range(i % (384))))} " for i in range(2048)]

    json_d = json.dumps({"input": sample, "model": "model"})
    session = requests.Session()
    req = session.get(f"{LIVE_URL}/models")
    assert req.status_code == 200

    batch_size = req.json()["data"]["stats"]["batch_size"]
    model_name = req.json()["data"]["id"]
    print(f"batch_size is {batch_size}, model={model_name}")
    model = SentenceTransformer(model_name_or_path=model_name)

    def local(data: list[str], iters=1):
        data_in = data * iters
        enc = model.encode(data_in, batch_size=batch_size)
        assert len(enc) == len(data_in)
        return enc[: len(data)]

    def remote(json_data: bytes, iters=1):
        fn = partial(session.post, data=json_data)
        req = list(tp.map(fn, [f"{LIVE_URL}/embeddings"] * iters))
        assert req[0].status_code == 200
        return req[0]

    local_resp = local(sample)
    remote_resp = [d["embedding"] for d in remote(json_d).json()["data"]]
    np.testing.assert_almost_equal(local_resp, remote_resp, 6)
    print("Both methods provide the identical output.")

    print("Measuring latency via SentenceTransformers")
    latency_st = timeit.timeit("local(sample, iters=5)", number=2, globals=locals())
    print("SentenceTransformers latency: ", latency_st)
    model = None

    print("Measuring latency via requests")
    latency_request = timeit.timeit(
        "remote(json_d, iters=5)", number=2, globals=locals()
    )
    print(f"Request latency: {latency_request}")

    assert latency_st * 1.1 > latency_request


if __name__ == "__main__":
    embedding_live_performance()
