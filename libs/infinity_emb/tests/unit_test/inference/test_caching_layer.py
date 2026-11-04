import asyncio
import threading

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


@pytest.mark.anyio
async def test_consumer_survives_failed_write(monkeypatch):
    """a raising `_cache.add` must not kill the only queue consumer.

    Before the fix, the exception escaped `_consume_queue`, ended the single writer
    thread, and was swallowed by the never-retrieved Future from `_threadpool.submit`.
    `_add_q` then had no consumer at all and grew for the lifetime of the process.
    """
    shutdown = threading.Event()
    try:
        c = caching_layer.Cache(cache_name="pytest_write_error", shutdown=shutdown)
        seen: list[str] = []
        real_add = c._cache.add

        def flaky(**kwargs):
            seen.append(kwargs["key"])
            if kwargs["key"] == "boom":
                raise RuntimeError("simulated disk failure")
            return real_add(**kwargs)

        monkeypatch.setattr(c._cache, "add", flaky)
        c._add_q.put(("boom", [1.0]))
        c._add_q.put(("fine", [2.0]))

        for _ in range(100):
            await asyncio.sleep(0.05)
            if seen == ["boom", "fine"]:
                break

        # the writer processed the item *after* the one that raised
        assert seen == ["boom", "fine"]
        assert c._get("fine") == [2.0]
    finally:
        shutdown.set()
