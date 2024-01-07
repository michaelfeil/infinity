import concurrent.futures
import json
import time
import timeit
from functools import partial

import numpy as np
import requests  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

LIVE_URL = "http://localhost:7997"


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
    model.half()

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
    np.testing.assert_almost_equal(local_resp, remote_resp, 2)
    for r, e in zip(local_resp, remote_resp):
        cosine_sim = np.dot(r, e) / (np.linalg.norm(e) * np.linalg.norm(r))
        assert cosine_sim > 0.99
    print("Both methods provide the identical output.")

    print("Measuring latency via SentenceTransformers")
    latency_st = timeit.timeit("local(sample, iters=1)", number=2, globals=locals())
    print("SentenceTransformers latency: ", latency_st)
    model = None

    print("Measuring latency via requests")
    latency_request = timeit.timeit(
        "remote(json_d, iters=1)", number=2, globals=locals()
    )
    print(f"Infinity request latency: {latency_request}")


def latency_single():
    session = requests.Session()

    def _post(i):
        time.sleep(0.02)
        json_d = json.dumps({"input": [str(i)], "model": "model"})
        s = time.perf_counter()
        res = session.post(f"{LIVE_URL}/embeddings", data=json_d)
        e = time.perf_counter()
        assert res.status_code == 200
        return (e - s) * 10**3

    _post("hi")
    times = [_post(i) for i in range(32)]
    print(f"{np.median(times)}+-{np.std(times)}")


if __name__ == "__main__":
    embedding_live_performance()
