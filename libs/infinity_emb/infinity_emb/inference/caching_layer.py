# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil
"""
This file contains the experimental code for retrieving and storing result of embeddings to diskcache
which may reduce latency.
"""

import asyncio
import hashlib
import json
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from infinity_emb._optional_imports import CHECK_DISKCACHE
from infinity_emb.env import MANAGER
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    AbstractSingle,
    AudioSingle,
    EmbeddingSingle,
    ImageSingle,
    QueueItemInner,
    ReRankSingle,
)

if CHECK_DISKCACHE.is_available:
    import diskcache as dc  # type: ignore[import-untyped]


class Cache:
    """wrapper around DiskCache. The Diskcache in infinity `races` against the model inference.

    The concept is that with the code be
    """

    def __init__(self, cache_name: str, shutdown: threading.Event) -> None:
        """
        cache_name: filename for diskcache
        shutdown: the shutdown event for the model worker & cache
        """
        CHECK_DISKCACHE.mark_required()

        self._shutdown = shutdown
        self._add_q: queue.Queue = queue.Queue()
        dir = MANAGER.cache_dir / "cache_vectors" f"cache_{cache_name}"
        logger.info(f"caching vectors under: {dir}")
        self._cache = dc.Cache(dir, size_limit=2**28)
        self.is_running = False
        self.startup()

    def startup(self):
        if not self.is_running:
            self._threadpool = ThreadPoolExecutor()
            self._threadpool.submit(self._consume_queue)

    @staticmethod
    def _pre_hash(content: AbstractSingle) -> bytes:
        """Hash the input type and full content, including image/audio metadata.

        Scheduling strings only estimate input lengths and are not cache identities.
        Byte keys also avoid reusing entries written with the old string keys.
        """
        state: Any = content
        if isinstance(content, (EmbeddingSingle, ReRankSingle)):
            # JSON preserves query/document boundaries without pickle string memoization.
            state = (type(content), json.dumps(content.to_input()))
        elif isinstance(content, ImageSingle):
            image = content.image
            # Pillow's own pickle state uses an RGB palette and loses palette alpha.
            state = (
                type(content),
                image.mode,
                image.size,
                image.tobytes(),
                image.getpalette("RGBA"),
                dict(sorted(image.info.items())),
            )
        elif isinstance(content, AudioSingle):
            state = (
                type(content),
                content.sampling_rate,
                content.audio.dtype.str,
                content.audio.shape,
                content.audio.tobytes(),
            )
        return b"content-v1:" + hashlib.sha256(pickle.dumps(state, protocol=4)).digest()

    def _consume_queue(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._add_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is not None:
                k, v = item
                self._cache.add(key=k, value=v, expire=86400)
            self._add_q.task_done()
        self._threadpool.shutdown(wait=True)

    def _get(self, content: AbstractSingle) -> tuple[bytes, Any]:
        """Compute the key once, off the event loop, and look up the cached result."""
        key = self._pre_hash(content)
        return key, self._cache.get(key=key)

    async def aget_complete(self, item: QueueItemInner) -> None:
        """sets the item.complete() and sets embedding, if in cache."""
        key, result = await to_thread(self._get, self._threadpool, item.content)
        if result is not None:
            # update item with cached result
            if not item.future.done():
                await item.complete(result)
        else:
            # result is not in cache yet, lets wait for it and add it
            result_new = await item.get_result()
            await asyncio.sleep(1e-3)
            self._add_q.put((key, result_new))
