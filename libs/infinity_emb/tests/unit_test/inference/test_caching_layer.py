import asyncio
import threading

import numpy as np
import pytest

from infinity_emb.inference.caching_layer import Cache
from infinity_emb.primitives import EmbeddingInner, EmbeddingSingle


@pytest.mark.timeout(20)
@pytest.mark.anyio
async def test_cache():
    loop = asyncio.get_event_loop()
    shutdown = threading.Event()
    try:
        sentence = "dummy"
        embedding = np.random.random(5).tolist()
        c = Cache(
            cache_name=f"pytest_{hash((sentence, tuple(embedding)))}", shutdown=shutdown
        )

        sample = EmbeddingInner(content=EmbeddingSingle(sentence))
        sample_embedded = EmbeddingInner(
            content=EmbeddingSingle(sentence),
        )
        sample_embedded.set_result(embedding)
        fut = loop.create_future()
        fut.set_result(None)
        await c.aget(sample_embedded, fut)
        await asyncio.sleep(1)
        # launch the ba
        await c.aget(sample, loop.create_future())
        result = sample.get_result()
        assert result is not None
        np.testing.assert_array_equal(result, embedding)
    finally:
        shutdown.set()
