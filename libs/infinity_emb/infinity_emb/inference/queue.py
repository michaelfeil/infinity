import asyncio
import enum
import threading
from typing import Dict, List, Optional, Union

from infinity_emb.inference.caching_layer import Cache
from infinity_emb.primitives import (
    EmbeddingReturnType,
    PrioritizedQueueItem,
    QueueItemInner,
)


class QueueSignal(enum.Enum):
    KILL = "kill"


class CustomFIFOQueue:
    def __init__(self) -> None:
        """"""
        self._lock_queue_event = threading.Lock()
        self._queue: List[PrioritizedQueueItem] = []
        # event that indicates items in queue.
        self._sync_event = threading.Event()

    def __len__(self):
        return len(self._queue)

    async def extend(self, items: List[PrioritizedQueueItem]):
        with self._lock_queue_event:
            self._queue.extend(items)
        self._sync_event.set()

    def pop_optimal_batches(
        self, size: int, max_n_batches: int = 4, timeout=0.2, **kwargs
    ) -> Union[List[List[QueueItemInner]], None]:
        """
        pop batch `up to size` + `continuous (sorted)` from queue

        Args:
            size (int): max size of batch
            max_n_batches: number of batches to be poped and sorted.
            timeout (float, optional): timeout until None is returned. Defaults to 0.2.
            latest_first (bool, optional): guarantees processing of oldest item in list.
                As latest first requires getting argmin of created timestamps,
                which is slow.  Defaults to False.

        returns:
            None: if there is not a single item in self._queue after timeout
            else: List[EmbeddingInner] with len(1<=size)
        """
        if not self._queue:
            if not self._sync_event.wait(timeout):
                return None

        # slice as many batches as possible
        n_batches = min(max_n_batches, max(1, len(self._queue) // size))
        size_batches = size * n_batches

        with self._lock_queue_event:
            new_items_l = self._queue[:size_batches]
            self._queue = self._queue[size_batches:]
            if not self._queue:
                self._sync_event.clear()

        if n_batches > 1:
            # sort the sentences by len ->
            # optimal padding per batch
            new_items_l.sort()

        new_items: List[List[QueueItemInner]] = []
        for i in range(n_batches):
            mini_batch = new_items_l[size * i : size * (i + 1)]
            mini_batch_e: List[QueueItemInner] = [mi.item for mi in mini_batch]
            if mini_batch_e:
                new_items.append(mini_batch_e)
        if new_items:
            return new_items
        else:
            return None


class ResultKVStoreFuture:
    def __init__(self, cache: Optional[Cache] = None) -> None:
        self._kv: Dict[str, asyncio.Future] = {}
        self._cache = cache
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return self._loop

    def __len__(self):
        return len(self._kv)

    async def register_item(self, item: QueueItemInner) -> None:
        """create future for item"""
        uuid = item.get_id()
        self._kv[uuid] = self.loop.create_future()
        return None

    async def wait_for_response(self, item: QueueItemInner) -> EmbeddingReturnType:
        """wait for future to return"""
        uuid = item.get_id()
        if self._cache:
            asyncio.create_task(self._cache.aget(item, self._kv[uuid]))
        # get new item
        item = await self._kv[uuid]

        return item.get_result()

    async def mark_item_ready(self, item: QueueItemInner) -> None:
        """mark item as ready. Item.get_result() must be set before calling this"""
        uuid = item.get_id()
        # faster than .pop
        fut = self._kv[uuid]

        try:
            fut.set_result(item)
            # logger.debug(f"marked {uuid} as ready with {len(item.get_result())}")
        except asyncio.InvalidStateError:
            pass
        del self._kv[uuid]
