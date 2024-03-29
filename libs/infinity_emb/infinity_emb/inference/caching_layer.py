import asyncio
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Union

from infinity_emb._optional_imports import CHECK_DISKCACHE
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.primitives import EmbeddingReturnType, QueueItemInner

if CHECK_DISKCACHE.is_available:
    import diskcache as dc  # type: ignore[import-untyped]

INFINITY_CACHE_VECTORS = (
    bool(os.environ.get("INFINITY_CACHE_VECTORS", False))
    and CHECK_DISKCACHE.is_available
)


class Cache:
    def __init__(self, cache_name: str, shutdown: threading.Event) -> None:
        CHECK_DISKCACHE.mark_required()
        from infinity_emb.transformer.utils import infinity_cache_dir

        self._shutdown = shutdown
        self._add_q: queue.Queue = queue.Queue()
        dir = os.path.join(infinity_cache_dir(), "cache_vectors", f"cache_{cache_name}")
        logger.info(f"caching vectors under: {dir}")
        self._cache = dc.Cache(dir, size_limit=2**28)
        self.is_running = False
        self.startup()

    def startup(self):
        if not self.is_running:
            self._threadpool = ThreadPoolExecutor()
            self._threadpool.submit(self._consume_queue)

    @staticmethod
    def _hash(key: Union[str, Any]) -> str:
        return str(key)

    def _consume_queue(self) -> None:
        while not self._shutdown.is_set():
            try:
                item = self._add_q.get(timeout=1)
            except queue.Empty:
                continue
            if item is not None:
                k, v = item
                self._cache.add(key=self._hash(k), value=v, expire=86400)
            self._add_q.task_done()
        self._threadpool.shutdown(wait=True)

    def _get(self, sentence: str) -> Union[None, EmbeddingReturnType, list[float]]:
        """sets the item.complete() and sets embedding, if in cache."""
        return self._cache.get(key=self._hash(sentence))

    async def aget_complete(self, item: QueueItemInner) -> None:
        """sets the item.complete() and sets embedding, if in cache."""
        item_as_str = item.content.str_repr()
        result = await to_thread(self._get, self._threadpool, item_as_str)
        if result is not None:
            # update item with cached result
            if not item.future.done():
                await item.complete(result)
        else:
            # result is not in cache yet, lets wait for it and add it
            result_new = await item.get_result()
            await asyncio.sleep(1e-3)
            self._add_q.put((item_as_str, result_new))
