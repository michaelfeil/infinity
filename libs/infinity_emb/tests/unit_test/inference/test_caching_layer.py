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


@pytest.mark.anyio
async def test_cache_planted_pickle_is_a_miss():
    loop = asyncio.get_event_loop()
    shutdown = threading.Event()
    try:
        c = caching_layer.Cache(cache_name="pytest_planted", shutdown=shutdown)
        with diskcache_community.Cache(
            c._cache.directory, disk=diskcache_community.Disk
        ) as other_writer:
            other_writer.set("dummy", Planted())

        sample = EmbeddingInner(
            content=EmbeddingSingle(sentence="dummy"), future=loop.create_future()
        )
        lookup = asyncio.create_task(c.aget_complete(sample))
        await asyncio.sleep(0.5)
        assert not sample.future.done()
        assert planted_calls == []
        await sample.complete(np.ones(5))
        await lookup
    finally:
        shutdown.set()
