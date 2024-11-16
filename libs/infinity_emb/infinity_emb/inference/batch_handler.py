# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

"""This file contains the dynamic batching logic of multiple requests"""

import asyncio
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np

from infinity_emb.env import MANAGER
from infinity_emb.inference.caching_layer import Cache
from infinity_emb.inference.queue import CustomFIFOQueue, ResultKVStoreFuture
from infinity_emb.inference.threading_asyncio import to_thread
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    AbstractSingle,
    ClassifyReturnType,
    EmbeddingReturnType,
    EmbeddingSingle,
    ImageClassType,
    ModelCapabilites,
    ModelNotDeployedError,
    OverloadStatus,
    PredictSingle,
    PrioritizedQueueItem,
    RerankReturnType,
    ReRankSingle,
    get_inner_item,
)

from infinity_emb.transformer.audio.utils import resolve_audios
from infinity_emb.transformer.utils import get_lengths_with_tokenize
from infinity_emb.transformer.vision.utils import resolve_images

if TYPE_CHECKING:
    from infinity_emb.transformer.abstract import BaseTypeHint


QUEUE_TIMEOUT = 0.5


class ShutdownReadOnly:
    def __init__(self, shutdown: threading.Event) -> None:
        self._shutdown = shutdown

    def is_set(self) -> bool:
        return self._shutdown.is_set()


class ThreadPoolExecutorReadOnly:
    def __init__(self, tp: ThreadPoolExecutor) -> None:
        self._tp = tp

    def submit(self, *args, **kwargs):
        return self._tp.submit(*args, **kwargs)


class BatchHandler:
    def __init__(
        self,
        model_replicas: list["BaseTypeHint"],
        max_batch_size: int,
        max_queue_wait: int = MANAGER.queue_size,
        batch_delay: float = 5e-3,
        vector_disk_cache_path: str = "",
        verbose=False,
        lengths_via_tokenize: bool = False,
    ) -> None:
        """
        performs the scheduling of the dynamic batching around the model.
        Holds the ModelWorker

        Args:
            model (BaseTransformer): the base class of the model to be used
            max_batch_size (int): max batch size of dynamic batch size
            max_queue_wait (int, optional): max items to queue in the batch, default 32_000
            batch_delay (float, optional): sleep in seconds, wait time for pre/post methods.
                Best result: setting to 1/2 the minimal expected
                time for core_encode method / "gpu inference".
                Dont set it above 1x minimal expected time of interence.
                Should not be 0 to not block Python's GIL.
            vector_disk_cache_path (str, optional): path to cache vectors on disk.
            lengths_via_tokenize (bool, optional): if True, use the tokenizer to get the lengths else len()
        """

        self._max_queue_wait = max_queue_wait
        self._lengths_via_tokenize = lengths_via_tokenize

        self._shutdown = threading.Event()
        self._threadpool = ThreadPoolExecutor()
        self._queue_prio = CustomFIFOQueue()
        self._publish_to_model_queue: Queue = Queue(8)
        self._result_queue: Queue = Queue(8)

        self.max_batch_size = max_batch_size
        self._verbose = verbose
        self.batch_delay = batch_delay

        # cache
        cache = (
            Cache(
                cache_name=str(vector_disk_cache_path),
                shutdown=self._shutdown,
            )
            if vector_disk_cache_path
            else None
        )
        self._result_store = ResultKVStoreFuture(cache)

        # model
        self.model_worker = [
            ModelWorker(
                shutdown=ShutdownReadOnly(self._shutdown),
                model=model_replica,
                threadpool=ThreadPoolExecutorReadOnly(self._threadpool),
                input_q=self._publish_to_model_queue,
                output_q=self._result_queue,
                verbose=self.batch_delay,
                batch_delay=batch_delay,
            )
            for model_replica in model_replicas
        ]

        if batch_delay > 0.1:
            logger.warning(f"high batch delay of {batch_delay}")
        if max_batch_size > max_queue_wait * 10:
            logger.warning(
                f"queue_size={self._max_queue_wait} to small "
                f"over batch_size={max_batch_size}."
                " Consider increasing queue size"
            )

    async def embed(self, sentences: list[str]) -> tuple[list["EmbeddingReturnType"], int]:
        """Schedule a sentence to be embedded. Awaits until embedded.

        Args:
            sentences (list[str]): Sentences to be embedded

        Raises:
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list["EmbeddingReturnType"]: list of embedding as 1darray
            int: token usage
        """
        if "embed" not in self.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `embed`. " f"Options are {self.capabilities}."
            )
        input_sentences = [EmbeddingSingle(sentence=s) for s in sentences]

        embeddings, usage = await self._schedule(input_sentences)
        return embeddings, usage

    async def rerank(
        self,
        query: str,
        docs: list[str],
        raw_scores: bool = False,
        top_n: Optional[int] = None,
    ) -> tuple[list[RerankReturnType], int]:
        """Schedule a query to be reranked with documents. Awaits until reranked.

        Args:
            query (str): query for reranking
            docs (list[str]): documents to be reranked
            raw_scores (bool): return raw scores instead of sigmoid
            top_n (Optional[int]): number of top scores to return after reranking
                if top_n is None, <= 0 or out of range, all scores are returned

        Raises:
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list[float]: list of scores
            int: token usage
        """
        if "rerank" not in self.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `rerank`. " f"Options are {self.capabilities}."
            )
        rerankables = [ReRankSingle(query=query, document=doc) for doc in docs]
        scores, usage = await self._schedule(rerankables)

        if not raw_scores:
            # perform sigmoid on scores
            scores = 1 / (1 + np.exp(-np.array(scores)))

        results = [
            RerankReturnType(relevance_score=scores[i], index=i, document=docs[i])
            for i in range(len(scores))
        ]
        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)

        if top_n is not None and top_n > 0:
            results = results[:top_n]

        return results, usage

    async def classify(
        self, *, sentences: list[str], raw_scores: bool = True
    ) -> tuple[list[ClassifyReturnType], int]:
        """Schedule a query to be classified with documents. Awaits until classified.

        Args:
            sentences (list[str]): sentences to be classified
            raw_scores (bool): if True, return raw scores, else softmax

        Raises:
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list[ClassifyReturnType]: list of class encodings
            int: token usage
        """
        if "classify" not in self.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `classify`. " f"Options are {self.capabilities}."
            )
        items = [PredictSingle(sentence=s) for s in sentences]
        classifications, usage = await self._schedule(items)

        if raw_scores:
            # perform softmax on scores
            pass

        return classifications, usage

    async def image_embed(
        self,
        *,
        images: list[Union[str, "ImageClassType", bytes]],
    ) -> tuple[list["EmbeddingReturnType"], int]:
        """Schedule a images and sentences to be embedded. Awaits until embedded.

        Args:
            images (list[Union[str, ImageClassType]]): list of pre-signed urls or ImageClassType objects

        Raises:
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list["EmbeddingReturnType"]: list of embedding as 1darray
            int: token usage
        """

        if "image_embed" not in self.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `image_embed`. "
                f"Options are {self.capabilities}."
            )

        items = await resolve_images(images)
        embeddings, usage = await self._schedule(items)
        return embeddings, usage

    async def audio_embed(
        self,
        *,
        audios: list[Union[str, bytes]],
    ) -> tuple[list["EmbeddingReturnType"], int]:
        """Schedule audios and sentences to be embedded. Awaits until embedded.

        Args:
            audios (list[NDArray]): list of raw wave data

        Raises:
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list["EmbeddingReturnType"]: list of embedding as 1darray
            int: token usage
        """

        if "audio_embed" not in self.capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `audio_embed`. "
                f"Options are {self.capabilities}."
            )

        items = await resolve_audios(
            audios,
            getattr(self.model_worker[0]._model, "sampling_rate", -42),
        )
        embeddings, usage = await self._schedule(items)
        return embeddings, usage

    async def _schedule(self, list_queueitem: Sequence[AbstractSingle]) -> tuple[list[Any], int]:
        """adds list of items to the queue and awaits until these are completed."""
        prios, usage = await self._get_prios_usage(list_queueitem)
        new_prioqueue: list[PrioritizedQueueItem] = []

        inner_item = get_inner_item(type(list_queueitem[0]))

        for re, p in zip(list_queueitem, prios):
            inner = inner_item(content=re, future=self.loop.create_future())  # type: ignore
            item = PrioritizedQueueItem(
                priority=p,
                item=inner,
            )
            new_prioqueue.append(item)
        self._queue_prio.extend(new_prioqueue)

        result = await asyncio.gather(
            *[self._result_store.wait_for_response(item.item) for item in new_prioqueue]
        )
        return result, usage

    @property
    def capabilities(self) -> set[ModelCapabilites]:
        # TODO: try to remove inheritance here and return upon init.
        return self.model_worker[0].capabilities

    def is_overloaded(self) -> bool:
        """checks if more items can be queued.

        Can be used on API level to reject requests if too many are queued and enable better autoscaling.
        """
        return len(self._queue_prio) > self._max_queue_wait

    def overload_status(self) -> OverloadStatus:
        """
        returns info about the queue status
        """
        return OverloadStatus(
            queue_fraction=len(self._queue_prio) / self._max_queue_wait,
            queue_absolute=len(self._queue_prio),
            results_absolute=len(self._result_store),
        )

    async def _get_prios_usage(self, items: Sequence[AbstractSingle]) -> tuple[list[int], int]:
        """get priorities and usage

        Args:
            items (list[AbstractSingle]): list of items that support a fn with signature
                `.str_repr() -> str` to get the string representation of the item.

        Returns:
            tuple[list[int], int]: prios, length
        """
        if not self._lengths_via_tokenize:
            return get_lengths_with_tokenize([it.str_repr() for it in items])
        else:
            return await to_thread(
                get_lengths_with_tokenize,
                self._threadpool,
                _sentences=[it.str_repr() for it in items],
                tokenize=self.model_worker[0].tokenize_lengths,
            )

    def _publish_towards_model(
        self,
        # shutdown: ShutdownReadOnly,
        # queue_prio: "CustomFIFOQueue",
        # publish_to_model_queue: Queue,
        # max_batch_size: int,
        # verbose: bool
    ):
        """background thread for reading  exits only if shutdown.is_set()"""
        max_n_batches = 8
        try:
            while not self._shutdown.is_set():
                if not self._publish_to_model_queue.empty() and (
                    self._publish_to_model_queue.full()
                    or (len(self._queue_prio) < self.max_batch_size * max_n_batches)
                ):
                    # patience:
                    # do not pop a batch if self._publish_to_model_queue still has item(s) left.
                    # - until GPU / _core_batch starts processing the previous item
                    # - or if many items are queued anyhow, so that a good batch
                    #   may be popped already.
                    time.sleep(self.batch_delay)
                    continue
                # decision to attempt to pop a batch
                # -> will happen if a single datapoint is available

                batches = self._queue_prio.pop_optimal_batches(self.max_batch_size, max_n_batches)

                for batch in batches:
                    if self._verbose:
                        logger.debug(
                            "[📦] batched %s requests, queue remaining:  %s",
                            len(batch),
                            len(self._queue_prio),
                        )
                    while not self._shutdown.is_set():
                        try:
                            self._publish_to_model_queue.put(batch, timeout=QUEUE_TIMEOUT)
                            break
                        except queue.Full:
                            continue
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Postprocessor crashed")

    @staticmethod
    async def _subscribe_to_model(
        shutdown: ShutdownReadOnly, result_queue: Queue, tp: ThreadPoolExecutor
    ):
        """background thread for reading  exits only if shutdown.is_set()"""
        schedule_errors = 0
        try:
            while not shutdown.is_set():
                try:
                    post_batch = result_queue.get_nowait()
                except queue.Empty:
                    # instead use async await to get
                    try:
                        post_batch = await to_thread(result_queue.get, tp, timeout=QUEUE_TIMEOUT)
                    except queue.Empty:
                        # in case of timeout start again
                        continue
                    except Exception as e:
                        # exception handing without loop forever.
                        time.sleep(1)
                        schedule_errors += 1
                        if schedule_errors > 10:
                            logger.error("too many schedule errors")
                            raise e
                        continue
                results, batch = post_batch
                for i, item in enumerate(batch):
                    await item.complete(results[i])

                result_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_subscribe_to_model crashed")

    async def spawn(self):
        """spawns the resources"""
        logger.info("creating batching engine")
        self.loop = asyncio.get_event_loop()

        self._threadpool.submit(
            self._publish_towards_model,
        )

        self._push_task = asyncio.create_task(
            self._subscribe_to_model(
                ShutdownReadOnly(self._shutdown), self._result_queue, self._threadpool
            )
        )
        for worker in self.model_worker:
            worker.spawn()

    async def shutdown(self):
        """
        set the shutdown event and close threadpool.
        Blocking event, until shutdown complete.
        reverses .spawn()
        """
        self._shutdown.set()
        await asyncio.to_thread(self._threadpool.shutdown)
        # collect task
        self._push_task.cancel()


class ModelWorker:
    """Model Worker. Handles pre, forward, and post-processing of any model."""

    def __init__(
        self,
        shutdown: ShutdownReadOnly,
        model: "BaseTypeHint",
        threadpool: ThreadPoolExecutorReadOnly,
        input_q: Queue,
        output_q: Queue,
        batch_delay: float = 5e-3,
        verbose=False,
    ) -> None:
        self._shutdown = shutdown
        self._model = model
        self._threadpool = threadpool
        self._feature_queue: Queue = Queue(3)
        self._postprocess_queue: Queue = Queue(5)
        self._batch_delay = float(max(1e-4, batch_delay))
        self._input_q = input_q
        self._output_q = output_q
        self._last_inference = time.perf_counter()
        self._verbose = verbose
        self._ready = False

    def spawn(self):
        if self._ready:
            raise ValueError("already spawned")
        # start the threads
        self._threadpool.submit(self._preprocess_batch)
        self._threadpool.submit(self._core_batch)
        self._threadpool.submit(self._postprocess_batch)

    @property
    def capabilities(self) -> set[ModelCapabilites]:
        return self._model.capabilities

    def tokenize_lengths(self, *args, **kwargs):
        return self._model.tokenize_lengths(*args, **kwargs)

    def _preprocess_batch(self):
        """loops and checks if the _core_batch has worked on all items"""
        logger.info("ready to batch requests.")
        self._ready = True
        try:
            while not self._shutdown.is_set():
                try:
                    batch = self._input_q.get(timeout=QUEUE_TIMEOUT)
                except queue.Empty:
                    continue
                # optimal batch has been selected ->
                # lets tokenize it and move tensors to GPU.

                if self._feature_queue.qsize() > 2:
                    # add some stochastic delay
                    time.sleep(self._batch_delay * 2)

                items_for_pre = [item.content.to_input() for item in batch]
                feat = self._model.encode_pre(items_for_pre)
                if self._verbose:
                    logger.debug(
                        "[🏃->🧠] preprocessed %s requests",
                        len(items_for_pre),
                    )
                if self._shutdown.is_set():
                    break
                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        self._feature_queue.put((feat, batch), timeout=QUEUE_TIMEOUT)
                        break
                    except queue.Full:
                        continue
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_preprocess_batch crashed")
        finally:
            self._ready = False

    def _core_batch(self):
        """waiting for preprocessed batches (on device)
        and do the forward pass / `.encode`
        """
        try:
            while not self._shutdown.is_set():
                try:
                    core_batch = self._feature_queue.get(timeout=QUEUE_TIMEOUT)
                except queue.Empty:
                    continue
                (feat, batch) = core_batch
                if self._verbose:
                    logger.debug("[🧠] Inference on batch_size=%s", len(batch))
                self._last_inference = time.perf_counter()
                embed = self._model.encode_core(feat)

                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        self._postprocess_queue.put((embed, batch), timeout=QUEUE_TIMEOUT)
                        break
                    except queue.Full:
                        continue
                self._feature_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_core_batch crashed.")

    def _postprocess_batch(self):
        """collecting forward(.encode) results and put them into the output queue store"""
        try:
            while not self._shutdown.is_set():
                try:
                    post_batch = self._postprocess_queue.get(timeout=QUEUE_TIMEOUT)
                except queue.Empty:
                    # instead use async await to get
                    continue

                if (
                    self._postprocess_queue.empty()
                    and self._last_inference < time.perf_counter() + self._batch_delay * 2
                ):
                    # 5 ms, assuming this is below
                    # 3-50ms for inference on avg.
                    # give the CPU some time to focus
                    # on moving the next batch to GPU on the forward pass
                    # before proceeding
                    time.sleep(self._batch_delay)
                embed, batch = post_batch
                results = self._model.encode_post(embed)
                if self._verbose:
                    logger.debug("[🧠->🏁] postprocessed %s requests", len(batch))
                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        self._output_q.put((results, batch), timeout=QUEUE_TIMEOUT)
                        break
                    except queue.Full:
                        continue
                self._postprocess_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("Postprocessor crashed")
