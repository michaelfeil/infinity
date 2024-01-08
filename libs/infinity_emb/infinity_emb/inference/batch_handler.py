import asyncio
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process, set_start_method
from queue import Queue
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from infinity_emb.inference.caching_layer import Cache
from infinity_emb.inference.model_worker import ModelWorker
from infinity_emb.inference.queue import (
    CustomFIFOQueue,
    ResultKVStoreFuture,
)
from infinity_emb.inference.select_model import (
    get_engine_type_from_config,
)
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    Device,
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
    QueueSignalMessages,
    ReRankInner,
    ReRankSingle,
)
from infinity_emb.transformer.utils import (
    CapableEngineType,
    InferenceEngine,
    get_lengths_with_tokenize,
)

_mp_manager = None


def get_mp_manager():
    global _mp_manager
    _mp_manager = Manager()
    return _mp_manager


class BatchHandler:
    def __init__(
        self,
        model_name_or_path: str,
        max_batch_size: int,
        engine: InferenceEngine = InferenceEngine.torch,
        model_warmup: bool = True,
        max_queue_wait: int = int(os.environ.get("INFINITY_QUEUE_SIZE", 32_000)),
        vector_disk_cache_path: str = "",
        verbose=False,
        lengths_via_tokenize: bool = False,
        device: Device = Device.auto,
        num_workers: int = 1,
        worker_method: str = "thread",
    ) -> None:
        """
        performs batching around the model.

        max_batch_size: max batch size of the models
        vector_disk_cache_path: path to cache vectors on disk.
        lengths_via_tokenize: if True, use the tokenizer to get the lengths else len()
        """
        self.model_name_or_path = model_name_or_path
        self.max_batch_size = max_batch_size
        self.model_warmup = model_warmup
        self.max_queue_wait = max_queue_wait
        self.lengths_via_tokenize = lengths_via_tokenize
        self.device = device
        self.num_workers = num_workers
        self.worker_method = worker_method

        self._shutdown = threading.Event()

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

        self.capable_engine: CapableEngineType = get_engine_type_from_config(
            model_name_or_path=model_name_or_path,
            engine=engine,
        )
        self.model_capabilities = self.capable_engine.value.capabilities

        if self.worker_method == "thread":
            self._shared_queue_model_out: Queue = Queue()
            self._shared_queue_model_in: Queue = Queue()
        else:
            set_start_method("spawn", force=True)

            mp_manager = get_mp_manager()
            self._shared_queue_model_out = mp_manager.Queue()
            self._shared_queue_model_in = mp_manager.Queue()

        self.worker_args = dict(
            in_queue=self._shared_queue_model_in,
            out_queue=self._shared_queue_model_out,
            max_batch_size=self.max_batch_size,
            model_name_or_path=self.model_name_or_path,
            capable_engine=self.capable_engine,
            model_warmup=self.model_warmup,
            device=self.device,
        )

    def run_model_worker(self):
        if self.worker_method == "thread":
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                tasks = [
                    pool.submit(ModelWorker, **self.worker_args)
                    for _ in range(self.num_workers)
                ]
                for task in tasks:
                    task.result()
        else:
            tasks = [
                Process(target=ModelWorker, kwargs=self.worker_args)
                for _ in range(self.num_workers)
            ]
            for task in tasks:
                task.start()
            for task in tasks:
                task.join()
        logger.info("ModelWorker: all workers finished.")

    async def embed(
        self, sentences: List[str]
    ) -> tuple[List[EmbeddingReturnType], int]:
        """Schedule a sentence to be embedded. Awaits until embedded.

        Args:
            sentences (List[str]): Sentences to be embedded

        Returns:
            EmbeddingReturnType: list of embedding as 1darray
        """
        if "embed" not in self.model_capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `embed`."
                f"options are {self.model_capabilities}"
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
        if "rerank" not in self.model_capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `rerank`."
                f"options are {self.model_capabilities}"
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
        if "classify" not in self.model_capabilities:
            raise ModelNotDeployedError(
                "the loaded moded cannot fullyfill `classify`."
                f"options are {self.model_capabilities}"
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
                item=inner_item(content=re),  # type: ignore
            )
            new_prioqueue.append(item)

        await asyncio.gather(
            *[self._result_store.register_item(item.item) for item in new_prioqueue]
        )

        await self._queue_prio.extend(new_prioqueue)

        result = await asyncio.gather(
            *[self._result_store.wait_for_response(item.item) for item in new_prioqueue]
        )
        return result, usage

    @property
    def capabilities(self) -> Set[ModelCapabilites]:
        return self.model_capabilities

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
        if not self.lengths_via_tokenize:
            return get_lengths_with_tokenize([it.str_repr() for it in items])
        else:
            # TODO: fix lengths_via_tokenize
            return await asyncio.to_thread(
                get_lengths_with_tokenize,
                _sentences=[it.str_repr() for it in items],
                # tokenize=self.model.tokenize_lengths, # TODO: fix
            )

    def _preprocess_batch(self):
        """loops and checks if the _core_batch has worked on all items"""
        self._ready = True
        logger.info("Batch_handler: Ready to batch requests.")
        try:
            while not self._shutdown.is_set():
                # patience:
                # do not pop a batch if self._feature_queue still has an item left
                # - until GPU / _core_batch starts processing the previous item
                # - or if many items are queued anyhow, so that a good batch
                #   may be popped already.
                if not self._shared_queue_model_in.empty() and (
                    self._shared_queue_model_in.full()
                    or (len(self._queue_prio) < self.max_batch_size * 4)
                ):
                    # add some stochastic delay
                    time.sleep(2e-4)
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
                    if self._shared_queue_model_in.qsize() > 2:
                        # add some stochastic delay
                        time.sleep(2e-4)

                    items_for_pre = [item.content.to_input() for item in batch]

                    logger.debug(
                        "[📦] batched %s requests, queue remaining:  %s",
                        len(items_for_pre),
                        self._shared_queue_model_in.qsize(),
                    )

                    # while-loop just for shutdown
                    while not self._shutdown.is_set():
                        try:
                            self._shared_queue_model_in.put(
                                (items_for_pre, batch), timeout=1
                            )
                            break
                        except queue.Full:
                            continue
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_preprocess_batch crashed")
        self._ready = False

    async def _queue_finalizer(self):
        try:
            while True:
                try:
                    batch = self._shared_queue_model_out.get_nowait()
                except queue.Empty:
                    # instead use async await to get
                    batch = await asyncio.to_thread(self._shared_queue_model_out.get)
                if batch == QueueSignalMessages.KILL:
                    break

                for item in batch[1]:
                    await self._result_store.mark_item_ready(item)
        except Exception as ex:
            logger.exception(ex)
            raise ValueError("_queue_finalizer crashed")

    async def _delayed_warmup(self):
        """in case there is no warmup -> perform some warmup."""
        await asyncio.sleep(5)
        if not self._shutdown.is_set():
            try:
                if "embed" in self.model_capabilities:
                    logger.debug("Sending a warm up to `embed`.")
                    await self.embed(sentences=["test"] * self.max_batch_size)
                if "rerank" in self.model_capabilities:
                    logger.debug("Sending a warm up to `rerank`.")
                    await self.rerank(
                        query="query", docs=["test"] * self.max_batch_size
                    )
                if "classify" in self.model_capabilities:
                    logger.debug("Sending a warm up to `classify`.")
                    await self.classify(sentences=["test"] * self.max_batch_size)
            except Exception as ex:
                logger.exception(ex)

    async def astart(self):
        """set up the resources in batch"""
        if self._ready:
            raise ValueError("previous threads are still running.")
        logger.info("creating batching engine")
        self.model_worker = asyncio.create_task(
            asyncio.to_thread(self.run_model_worker)
        )
        asyncio.create_task(self._queue_finalizer())
        asyncio.create_task(asyncio.to_thread(self._preprocess_batch))
        asyncio.create_task(self._delayed_warmup())

    async def astop(self):
        """
        set the shutdown event and close threadpool.
        Blocking event, until shutdown.
        """
        logger.debug("batch handler -> start astop")
        self._shutdown.set()

        self._shared_queue_model_out.put(QueueSignalMessages.KILL)
        # at last, kill the models, which kill the MP Manager.
        self._shared_queue_model_in.put(QueueSignalMessages.KILL)
        time.sleep(1)
        logger.debug("batch handler <- done astop")
