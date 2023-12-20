import asyncio
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from infinity_emb.inference.caching_layer import Cache
from infinity_emb.inference.queue import CustomFIFOQueue, ResultKVStoreFuture
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    EmbeddingInner,
    EmbeddingReturnType,
    EmbeddingSingle,
    ModelCapabilites,
    ModelNotDeployedError,
    OverloadStatus,
    PipelineItem,
    PredictInner,
    PredictSingle,
    PrioritizedQueueItem,
    ReRankInner,
    ReRankSingle,
)
from infinity_emb.transformer.abstract import BaseTransformer
from infinity_emb.transformer.utils import get_lengths_with_tokenize


class BatchHandler:
    def __init__(
        self,
        model: BaseTransformer,
        max_batch_size: int,
        max_queue_wait: int = int(os.environ.get("INFINITY_QUEUE_SIZE", 32_000)),
        batch_delay: float = 5e-3,
        vector_disk_cache_path: str = "",
        verbose=False,
        lengths_via_tokenize: bool = False,
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
        vector_disk_cache_path: path to cache vectors on disk.
        lengths_via_tokenize: if True, use the tokenizer to get the lengths else len()
        """
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_queue_wait = max_queue_wait
        self._lengths_via_tokenize = lengths_via_tokenize
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

    async def embed(
        self, sentences: List[str]
    ) -> tuple[List[EmbeddingReturnType], int]:
        """Schedule a sentence to be embedded. Awaits until embedded.

        Args:
            sentences (List[str]): Sentences to be embedded

        Returns:
            EmbeddingReturnType: list of embedding as 1darray
        """
        if "embed" not in self.model.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `embed`."
                f"options are {self.model.capabilities} inherited "
                f"from model_class={self.model.__class__}"
            )
        input_sentences = [EmbeddingSingle(s) for s in sentences]

        embeddings, usage = await self._schedule(input_sentences)
        return embeddings, usage

    async def rerank(
        self, query: str, docs: List[str], raw_scores: bool = False
    ) -> tuple[List[float], int]:
        """Schedule a query to be reranked with documents. Awaits until reranked.

        Args:
            query (str): query for reranking
            documents (List[str]): documents to be reranked

        Returns:
            List[float]: list of scores
            int: token usage
        """
        if "rerank" not in self.model.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `rerank`."
                f"options are {self.model.capabilities} inherited "
                f"from model_class={self.model.__class__}"
            )
        rerankables = [ReRankSingle(query=query, document=doc) for doc in docs]
        scores, usage = await self._schedule(rerankables)

        if not raw_scores:
            # perform sigmoid on scores
            scores = (1 / (1 + np.exp(-np.array(scores)))).tolist()

        return scores, usage

    async def classify(
        self, *, sentences: List[str], raw_scores: bool = True
    ) -> Tuple[List[Dict[str, float]], int]:
        """Schedule a query to be classified with documents. Awaits until classified.

        Args:
            sentences (List[str]): sentences to be classified
            raw_scores (bool): if True, return raw scores, else softmax

        Returns:
            EmbeddingReturnType: embedding as 1darray
        """
        if "classify" not in self.model.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `classify`."
                f"options are {self.model.capabilities} inherited "
                f"from model_class={self.model.__class__}"
            )
        items = [PredictSingle(sentence=s) for s in sentences]
        classifications, usage = await self._schedule(items)

        if raw_scores:
            # perform softmax on scores
            pass

        return classifications, usage

    async def _schedule(
        self, list_queueitem: Sequence[PipelineItem]
    ) -> Tuple[List[Any], int]:
        prios, usage = await self._get_prios_usage(list_queueitem)
        new_prioqueue: List[PrioritizedQueueItem] = []

        if isinstance(list_queueitem[0], EmbeddingSingle):
            inner_item = EmbeddingInner  # type: ignore
        elif isinstance(list_queueitem[0], ReRankSingle):
            inner_item = ReRankInner  # type: ignore
        elif isinstance(list_queueitem[0], PredictSingle):
            inner_item = PredictInner  # type: ignore
        else:
            raise ValueError(f"Unknown type of list_queueitem, {list_queueitem[0]}")

        for re, p in zip(list_queueitem, prios):
            item = PrioritizedQueueItem(
                priority=p,
                item=inner_item(
                    content=re, future=self.loop.create_future()  # type: ignore
                ),
            )
            new_prioqueue.append(item)
        await self._queue_prio.extend(new_prioqueue)

        result = await asyncio.gather(
            *[self._result_store.wait_for_response(item.item) for item in new_prioqueue]
        )
        return result, usage

    def get_capabilities(self) -> Set[ModelCapabilites]:
        return self.model.capabilities

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

    async def _get_prios_usage(
        self, items: Sequence[PipelineItem]
    ) -> Tuple[List[int], int]:
        """get priorities and usage

        Args:
            items (List[PipelineItem]): List of items that support a fn with signature
                `.str_repr() -> str` to get the string representation of the item.

        Returns:
            Tuple[List[int], int]: prios, length
        """
        if not self._lengths_via_tokenize:
            return get_lengths_with_tokenize([it.str_repr() for it in items])
        else:
            return await to_thread(
                get_lengths_with_tokenize,
                self._threadpool,
                _sentences=[it.str_repr() for it in items],
                tokenize=self.model.tokenize_lengths,
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

                    items_for_pre = [item.content.to_input() for item in batch]
                    feat = self.model.encode_pre(items_for_pre)
                    if self._verbose:
                        logger.debug(
                            "[📦] batched %s requests, queue remaining:  %s",
                            len(items_for_pre),
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
                results = self.model.encode_post(embed)
                for i, item in enumerate(batch):
                    await item.complete(results[i])

                self._postprocess_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Postprocessor crashed")

    async def _delayed_warmup(self):
        """in case there is no warmup -> perform some warmup."""
        await asyncio.sleep(10)
        logger.debug("Sending a warm up through embedding.")
        if "embed" in self.model.capabilities:
            await self.embed(sentences=["test"] * self.max_batch_size)
        if "rerank" in self.model.capabilities:
            await self.rerank(query="query", docs=["test"] * self.max_batch_size)
        if "classify" in self.model.capabilities:
            await self.classify(sentences=["test"] * self.max_batch_size)

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
