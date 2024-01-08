import asyncio
import os
import queue
import threading
from typing import Any, List, Union

from infinity_emb.log_handler import logger
from infinity_emb.primitives import EmbeddingReturnType, QueueItemInner
from infinity_emb.transformer.utils import infinity_cache_dir

try:
    import diskcache as dc  # type: ignore

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

INFINITY_CACHE_VECTORS = (
    bool(os.environ.get("INFINITY_CACHE_VECTORS", False)) and DISKCACHE_AVAILABLE
)


class Cache:
    def __init__(self, cache_name: str, shutdown: threading.Event) -> None:
        if not DISKCACHE_AVAILABLE:
            raise ImportError(
                "diskcache is not available. "
                "install via `pip install infinity-emb[cache]`"
            )

        self._shutdown = shutdown
        self._add_q: queue.Queue = queue.Queue()
        dir = os.path.join(infinity_cache_dir(), "cache_vectors", f"cache_{cache_name}")
        logger.info(f"caching vectors under: {dir}")
        self._cache = dc.Cache(dir, size_limit=2**28)
        self.is_running = False

    async def _verify_running(self):
        if not self.is_running:
            asyncio.create_task(asyncio.to_thread(self._consume_queue))

    @staticmethod
    def _hash(key: Union[str, Any]) -> str:
        return str(key)

    def _consume_queue(self) -> None:
        self.is_running = True
        while not self._shutdown.is_set():
            try:
                item = self._add_q.get(timeout=1)
            except queue.Empty:
                continue
            if item is not None:
                k, v = item
                self._cache.add(key=self._hash(k), value=v, expire=86400)
            self._add_q.task_done()
        self.is_running = False

    def _get(self, sentence: str) -> Union[None, EmbeddingReturnType, List[float]]:
        return self._cache.get(key=self._hash(sentence))

    async def aget(self, item: QueueItemInner, future: asyncio.Future) -> None:
        """Sets result to item and future, if in cache.
        If not in cache, sets future to be done when result is set.
        """
        await self._verify_running()
        item_as_str = item.content.str_repr()
        result = await asyncio.to_thread(self._get, item_as_str)
        if result is not None:
            # update item with cached result
            if item.get_result() is None:
                item.set_result(result)
                try:
                    future.set_result(None)
                except asyncio.InvalidStateError:
                    pass
        else:
            await future
            result = item.get_result()
            self._add_q.put((item_as_str, result))
