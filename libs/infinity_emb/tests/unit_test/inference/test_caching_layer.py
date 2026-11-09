import asyncio
import threading

import diskcache_community
import numpy as np
import pytest

from infinity_emb.inference import caching_layer
from infinity_emb.primitives import EmbeddingInner, EmbeddingSingle


@pytest.mark.anyio
async def test_cache():
    global INFINITY_CACHE_VECTORS

    loop = asyncio.get_event_loop()
    shutdown = threading.Event()
    try:
        INFINITY_CACHE_VECTORS = True
        sentence = "dummy"
        embedding = np.random.random(5).tolist()
        c = caching_layer.Cache(
            cache_name=f"pytest_{hash((sentence, tuple(embedding)))}", shutdown=shutdown
        )

        sample = EmbeddingInner(
            content=EmbeddingSingle(sentence=sentence), future=loop.create_future()
        )
        sample_embedded = EmbeddingInner(
            content=EmbeddingSingle(sentence=sentence),
            future=loop.create_future(),
            embedding=None,
        )
        await sample_embedded.complete(embedding)
        await c.aget_complete(sample_embedded)
        # add the embedded sample
        await asyncio.sleep(0.5)
        # launch the ba
        await c.aget_complete(sample)
        assert sample.future.done()
        assert sample.embedding is not None
        np.testing.assert_array_equal(sample.embedding, embedding)
    finally:
        INFINITY_CACHE_VECTORS = False
        shutdown.set()


planted_calls = []


def planted():
    planted_calls.append(True)
    return np.zeros(5)


class Planted:
    def __reduce__(self):
        return (planted, ())


def test_cache_planted_pickle_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setitem(vars(caching_layer.MANAGER), "cache_dir", tmp_path)
    planted_calls.clear()
    shutdown = threading.Event()
    try:
        c = caching_layer.Cache(cache_name="planted", shutdown=shutdown)
        with diskcache_community.Cache(c._cache.directory, disk=diskcache_community.Disk) as writer:
            writer.set("dummy", Planted())
        assert c._get("dummy") is None
        assert planted_calls == []
    finally:
        shutdown.set()
