import asyncio
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Dict, List, Optional, Union

from infinity_emb.inference.caching_layer import Cache
from infinity_emb.inference.primitives import (
    EmbeddingResult,
    NpEmbeddingType,
    OverloadStatus,
    PrioritizedQueueItem,
)
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseTransformer
from infinity_emb.transformer.utils import get_lengths_with_tokenize


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
        self, size: int, max_n_batches: int = 32, timeout=0.2, **kwargs
    ) -> Union[List[List[EmbeddingResult]], None]:
        """
        pop batch `up to size` + `continuous (sorted)` from queue

        Args:
            size (int): max size of batch
            timeout (float, optional): timeout until None is returned. Defaults to 0.2.
            latest_first (bool, optional): guarantees processing of oldest item in list.
                As latest first requires getting argmin of created timestamps,
                which is slow.  Defaults to False.

        returns:
            None: if there is not a single item in self._queue after timeout
            else: List[EmbeddingResult] with len(1<=size)
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

        new_items: List[List[EmbeddingResult]] = []
        for i in range(n_batches):
            mini_batch = new_items_l[size * i : size * (i + 1)]
            mini_batch_e: List[EmbeddingResult] = [
                mi.item for mi in mini_batch if not mi.item.future.done()
            ]
            if mini_batch_e:
                new_items.append(mini_batch_e)
        if new_items:
            return new_items
        else:
            return None


class ResultKVStoreFuture:
    def __init__(self, cache: Optional[Cache] = None) -> None:
        self._kv: Dict[str, NpEmbeddingType] = {}
        self._cache = cache

    def __len__(self):
        return len(self._kv)

    async def wait_for_response(self, item: EmbeddingResult) -> NpEmbeddingType:
        """wait for future to return"""
        if self._cache:
            asyncio.create_task(self._cache.aget_complete(item))
        return await item.future

    async def _extend(self, batch: List[EmbeddingResult]):
        for item in batch:
            item.complete()

    async def extend(self, batch: List[EmbeddingResult]) -> None:
        """extend store with results"""
        asyncio.create_task(self._extend(batch))
        if self._cache:
            asyncio.create_task(self._cache.add(batch))


class BatchHandler:
    def __init__(
        self,
        model: BaseTransformer,
        max_batch_size: int,
        max_queue_wait: int = int(os.environ.get("INFINITY_QUEUE_SIZE", 32_000)),
        batch_delay: float = 5e-3,
        vector_disk_cache_path: str = "",
        verbose=False,
    ) -> None:
        """
        performs batching around the model.

        model: BaseTransformer, implements fn (core|pre|post)_encode
        max_batch_size: max batch size of the models
        max_queue_wait: max items to queue in the batch, default 32_000 sentences
        batch_delay: sleep in seconds, wait time for pre/post methods.
            Best result: setting to 1/2 the minimal expected
            time for core_encode method / "gpu inference".
            Dont set it above 1x minimal expected time of interence.
            Should not be 0 to not block Python's GIL.
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_queue_wait = max_queue_wait
        self._verbose = verbose
        self._shutdown = threading.Event()
        self._feature_queue: Queue = Queue(6)
        self._postprocess_queue: Queue = Queue(4)
        self._batch_delay = float(max(1e-4, batch_delay))
        self._threadpool = ThreadPoolExecutor()
        self._queue_prio = CustomFIFOQueue()
        cache = (
            Cache(
                cache_name=str(vector_disk_cache_path),
                shutdown=self._shutdown,
            )
            if vector_disk_cache_path
            else None
        )

        self._result_store = ResultKVStoreFuture(cache)
        self._ready = False
        self._last_inference = time.perf_counter()

        if batch_delay > 0.1:
            logger.warn(f"high batch delay of {self._batch_delay}")
        if max_batch_size > max_queue_wait * 10:
            logger.warn(
                f"queue_size={self.max_queue_wait} to small "
                f"over batch_size={self.max_batch_size}."
                " Consider increasing queue size"
            )

    async def schedule(self, sentences: List[str]) -> tuple[List[NpEmbeddingType], int]:
        """Schedule a sentence to be embedded. Awaits until embedded.

        Args:
            sentences (List[str]): Sentences to be embedded
            prio (List[int]): priority for this embedding

        Returns:
            NpEmbeddingType: embedding as 1darray
        """
        # add an unique identifier

        prios, usage = get_lengths_with_tokenize(sentences)

        prioqueue = []
        for s, p in zip(sentences, prios):
            item = PrioritizedQueueItem(
                priority=p,
                item=EmbeddingResult(sentence=s, future=self.loop.create_future()),
            )
            prioqueue.append(item)
        await self._queue_prio.extend(prioqueue)

        embeddings = await asyncio.gather(
            *[self._result_store.wait_for_response(item.item) for item in prioqueue]
        )
        return embeddings, usage

    def is_overloaded(self) -> bool:
        """checks if more items can be queued."""
        return len(self._queue_prio) > self.max_queue_wait

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
                    time.sleep(self._batch_delay)
                    continue
                # decision to attemp to pop a batch
                # -> will happen if a single datapoint is available

                batches = self._queue_prio.pop_optimal_batches(
                    self.max_batch_size, latest_first=False
                )
                if not batches:
                    # not a single sentence available / len=0, wait for more
                    continue
                # optimal batch has been selected ->
                # lets tokenize it and move tensors to GPU.
                for batch in batches:
                    if self._feature_queue.qsize() > 2:
                        # add some stochastic delay
                        time.sleep(self._batch_delay * 2)

                    sentences = [item.sentence for item in batch]
                    feat = self.model.encode_pre(sentences)
                    if self._verbose:
                        logger.debug(
                            "[📦] batched %s requests, queue remaining:  %s",
                            len(sentences),
                            len(self._queue_prio),
                        )
                    if self._shutdown.is_set():
                        break
                    # while-loop just for shutdown
                    while not self._shutdown.is_set():
                        try:
                            self._feature_queue.put((feat, batch), timeout=1)
                            break
                        except queue.Full:
                            continue
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_preprocess_batch crashed")
        self._ready = False

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
                if self._verbose:
                    logger.debug("[🏃] Inference on batch_size=%s", len(batch))
                self._last_inference = time.perf_counter()
                embed = self.model.encode_core(feat)

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
            raise ValueError("_core_batch crashed.")

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
                except queue.Empty:
                    # instead use async await to get
                    try:
                        post_batch = await to_thread(
                            self._postprocess_queue.get, self._threadpool, timeout=1
                        )
                    except queue.Empty:
                        # in case of timeout start again
                        continue

                if (
                    self._postprocess_queue.empty()
                    and self._last_inference
                    < time.perf_counter() + self._batch_delay * 2
                ):
                    # 5 ms, assuming this is below
                    # 3-50ms for inference on avg.
                    # give the CPU some time to focus
                    # on moving the next batch to GPU on the forward pass
                    # before proceeding
                    await asyncio.sleep(self._batch_delay)
                embed, batch = post_batch
                embeddings = self.model.encode_post(embed).tolist()
                for i, item in enumerate(batch):
                    item.embedding = embeddings[i]
                await self._result_store.extend(batch)
                self._postprocess_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Postprocessor crashed")

    async def _delayed_warmup(self):
        """in case there is no warmup -> perform some warmup."""
        await asyncio.sleep(10)
        logger.debug("Sending a warm up through embedding.")
        await self.schedule(["test"] * self.max_batch_size)

    async def spawn(self):
        """set up the resources in batch"""
        if self._ready:
            raise ValueError("previous threads are still running.")
        logger.info("creating batching engine")
        self.loop = asyncio.get_event_loop()
        self._threadpool.submit(self._preprocess_batch)
        self._threadpool.submit(self._core_batch)
        asyncio.create_task(self._postprocess_batch())
        asyncio.create_task(self._delayed_warmup())

    async def shutdown(self):
        """
        set the shutdown event and close threadpool.
        Blocking event, until shutdown complete.
        """
        self._shutdown.set()
        with ThreadPoolExecutor() as tp_temp:
            await to_thread(self._threadpool.shutdown, tp_temp)
