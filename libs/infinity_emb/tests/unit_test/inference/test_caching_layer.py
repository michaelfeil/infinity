import asyncio
import threading

import numpy as np
import pytest

from infinity_emb.inference import caching_layer
from infinity_emb.inference.primitives import EmbeddingResult


@pytest.mark.anyio
async def test_cache():
    global INFINITY_CACHE_VECTORS

    loop = asyncio.get_event_loop()
    shutdown = threading.Event()
    try:
        INFINITY_CACHE_VECTORS = True
        c = caching_layer.Cache(cache_name="pytest", shutdown=shutdown)
        sentence = "dummy"
        embedding = np.random.random(30)
        sample = EmbeddingResult(sentence=sentence, future=loop.create_future())
        sample_embedded = EmbeddingResult(
            sentence=sentence, future=loop.create_future(), embedding=embedding
        )
        sample_embedded.complete()
        # add the embedded sample
        await c.add([sample_embedded])
        await asyncio.sleep(0.5)
        # launch the ba
        await c.aget_complete(sample)
        assert sample.future.done()
        assert sample.embedding is not None
        assert (sample.embedding == embedding).all()
    finally:
        INFINITY_CACHE_VECTORS = False
        shutdown.set()
