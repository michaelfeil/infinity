import asyncio
import bisect
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

from infinity_emb.inference.models import BaseTransformer, get_lengths_with_tokenize
from infinity_emb.inference.primitives import (
    EmbeddingResult,
    NpEmbeddingType,
    OverloadStatus,
    PrioritizedQueueItem,
)
from infinity_emb.inference.threading_asyncio import EventTS
from infinity_emb.log_handler import logger


class CustomPrioQueue:
    def __init__(self, sort_on_arrival=True) -> None:
        self._lock_queue_event = threading.Lock()  # lock queue and
        # queue is a always sorted
        self._queue: List[PrioritizedQueueItem] = []
        self._sync_event = threading.Event()
        self._sort_on_arrival = sort_on_arrival

    def __len__(self):
        return len(self._queue)

    async def extend(self, items: List[PrioritizedQueueItem]):
        with self._lock_queue_event:
            if self._sort_on_arrival:
                for item in items:
                    bisect.insort(self._queue, item)
            else:
                self._queue.extend(items)

            self._sync_event.set()

    def pop_optimal_batch(
        self, size: int, timeout=0.2
    ) -> Union[List[EmbeddingResult], None]:
        """
        pop batch `up to size` + `continuous (sorted)` from queue

        returns:
            None: if there is not a single item in self._queue
            else: List[EmbeddingResult] with len(1<=size)
        """
        if not self._queue:
            if not self._sync_event.wait(timeout):
                return None

        if len(self._queue) > size:
            # pick a random continuous slice, at beginning or at the end.
            start = random.randrange(-size, len(self._queue))
            start = max(0, min(len(self._queue) - size, start))
        else:
            start = 0
        end = start + size

        if not self._sort_on_arrival:
            self._queue.sort()

        with self._lock_queue_event:
            new_items = self._queue[start:end]
            self._queue = self._queue[:start] + self._queue[end:]
            if not self._queue:
                self._sync_event.clear()

        return list(n.item for n in new_items)


class ResultKVStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kv: Dict[str, NpEmbeddingType] = {}

    def __len__(self):
        return len(self._kv)

    async def wait_for_response(self, uuid: str, event: EventTS) -> NpEmbeddingType:
        await event.wait()
        # with self._lock:
        response = self._kv[uuid]
        del self._kv[uuid]
        return response

    async def extend(self, batch: List[EmbeddingResult]) -> None:
        """extend store with results"""
        # # with self._lock:
        # for item in batch:
        #     # first create (while _lock)
        #     self._kv[item.uuid] = item.embedding
        #     # all done, mark EmbeddingResult for collection
        #     item.event.set()
        _update = {item.uuid: item.embedding for item in batch}

        self._kv.update(_update)
        for item in batch:
            # all done, mark EmbeddingResult for collection
            item.event.set()


class BatchHandler:
    def __init__(
        self,
        model: BaseTransformer,
        max_batch_size: int,
        threadpool: ThreadPoolExecutor,
        max_queue_wait: int = 64_000,
        verbose=False,
    ) -> None:
        self._shutdown = threading.Event()
        self._queue_prio = CustomPrioQueue()
        self._result_store = ResultKVStore()
        self._feature_queue: queue.Queue = queue.Queue(4)
        self._postprocess_queue: queue.Queue = queue.Queue(4)
        self.max_batch_size = max_batch_size
        self.model = model
        self.max_queue_wait = max_queue_wait
        self._threadpool = threadpool
        self._ready = False
        self._verbose = verbose

    def shutdown(self):
        self._shutdown.set()

    async def schedule(self, sentences: List[str]) -> tuple[List[NpEmbeddingType], int]:
        """Schedule a sentence to be embedded. Awaits until embedded.

        Args:
            sentences (List[str]): Sentences to be embedded
            prio (List[int]): priority for this embedding

        Returns:
            NpEmbeddingType: embedding as 1darray
        """
        # add an unique identifier
        uuid_event = []
        prioqueue = []

        prios, usage = get_lengths_with_tokenize(
            sentences
        )  # , self.model.tokenize_lengths)

        for s, p in zip(sentences, prios):
            inner = EmbeddingResult(sentence=s, event=EventTS(self._threadpool))
            item = PrioritizedQueueItem(item=inner, priority=p)
            uuid_event.append((inner.uuid, inner.event))
            prioqueue.append(item)
        await self._queue_prio.extend(prioqueue)

        gather_results = [
            self._result_store.wait_for_response(uuid, event)
            for uuid, event in uuid_event
        ]
        embeddings = await asyncio.gather(*gather_results)
        return embeddings, usage

    def is_overloaded(self) -> bool:
        # start consuming
        return len(self._queue_prio) > self.max_queue_wait

    def ready(self) -> bool:
        return self._ready

    def overload_status(self) -> OverloadStatus:
        """
        returns info about the queue status
        """
        return OverloadStatus(
            queue_fraction=len(self._queue_prio) / self.max_queue_wait,
            queue_absolute=len(self._queue_prio),
            results_absolute=len(self._result_store),
        )

    def _preprocess_batch(self):
        """loops and checks if the _core_batch has worked on all items"""
        self._ready = True
        logger.info("ready to batch requests.")
        try:
            while not self._shutdown.is_set():
                # patience:
                # do not pop a batch if self._feature_queue still has an item left
                # - until GPU / _core_batch starts processing the previous item
                # - or if many items are queued anyhow, so that a good batch
                #   may be popped already.
                if not self._feature_queue.empty() and (
                    self._feature_queue.full()
                    or (len(self._queue_prio) < self.max_batch_size * 4)
                ):
                    # add some stochastic delay
                    time.sleep(2e-3)
                    continue
                # decision to attemp to pop a batch
                # -> will happen if a single datapoint is available
                batch = self._queue_prio.pop_optimal_batch(self.max_batch_size)
                if not batch:
                    # not a single sentence available / len=0, wait for more
                    continue
                # optimal batch has been selected ->
                # lets tokenize it and move tensors to GPU.
                sentences = [item.sentence for item in batch]
                feat = self.model.encode_pre(sentences)
                if self._verbose:
                    logger.debug(
                        "[📦] batched %s requests, queue remaining:  %s",
                        len(sentences),
                        len(self._queue_prio),
                    )
                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        self._feature_queue.put((feat, batch), timeout=1)
                        break
                    except queue.Full:
                        continue
        except Exception as ex:
            logger.exception(ex)
            exit("_preprocess_batch crashed")

    def _core_batch(self):
        """waiting for preprocessed batches (on device)
        and do the forward pass / `.encode`
        """
        try:
            while not self._shutdown.is_set():
                try:
                    core_batch = self._feature_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                (feat, batch) = core_batch
                embed = self.model.encode_core(feat)
                if self._verbose:
                    logger.debug("[🏃] Inference done on batch_size=%s", len(batch))

                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        self._postprocess_queue.put((embed, batch), timeout=1)
                        break
                    except queue.Full:
                        continue
                self._feature_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            exit("_core_batch crashed.")

    async def _postprocess_batch(self):
        """collecting forward(.encode) results and put them into the result store"""
        # TODO: the ugly asyncio.sleep() could add to 3-8ms of latency worst case
        # In constrast, at full batch size, sleep releases cruical CPU at time of
        # the forward pass to GPU (after which there is crical time again)
        # and not affecting the latency
        try:
            while not self._shutdown.is_set():
                try:
                    post_batch = self._postprocess_queue.get_nowait()
                    if not self._postprocess_queue.qsize():
                        # queue is not in a hurry
                        # give the CPU some time to focus
                        # on moving the next batch to GPU on the forward pass
                        # before proceeding
                        await asyncio.sleep(1e-3)
                except queue.Empty:
                    # 7 ms, assuming this is below
                    # 3-50ms for inference on avg.
                    await asyncio.sleep(5e-3)
                    continue
                embed, batch = post_batch
                embeddings = self.model.encode_post(embed).tolist()
                for i, item in enumerate(batch):
                    item.embedding = embeddings[i]
                await self._result_store.extend(batch)
                self._postprocess_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            exit("Postprocessor crashed")

    async def spawn(self):
        """set up the resources in batch"""
        logger.info("creating batching engine")
        self._threadpool.submit(self._preprocess_batch)
        self._threadpool.submit(self._core_batch)
        asyncio.create_task(self._postprocess_batch())
