import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Union

import numpy as np

from infinity_emb.inference.primitives import EmbeddingResult
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.transformer.utils import infinity_cache_dir

try:
    import diskcache as dc

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
        self._add_q = queue.Queue()
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

    async def add(self, items: List[EmbeddingResult]) -> None:
        """add the item if in cache."""
        for item in items[::-1]:
            self._add_q.put((item.sentence, item.embedding))

    def _get(self, sentence: str) -> Union[None, np.ndarray, List[float]]:
        """sets the item.complete() and sets embedding, if in cache."""
        return self._cache.get(key=self._hash(sentence))

    async def aget_complete(self, item: EmbeddingResult) -> None:
        """sets the item.complete() and sets embedding, if in cache."""
        embedding = await to_thread(self._get, self._threadpool, item.sentence)
        if embedding is not None and not item.future.done():
            item.embedding = embedding
            item.complete()
