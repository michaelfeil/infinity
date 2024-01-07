import asyncio
import queue
import threading
import time

# from multiprocessing import Process
# from multiprocessing import Queue as MPQueue
from queue import Queue
from typing import Any

from infinity_emb.inference.select_model import (
    select_model,
)
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    Device,
)
from infinity_emb.transformer.utils import (
    CapableEngineType,
)


class ModelWorker:
    def __init__(
        self,
        in_queue: Queue,
        out_queue: Queue,
        shutdown_event: threading.Event,
        model_name_or_path: str,
        max_batch_size: int,
        capable_engine: CapableEngineType,
        model_warmup: bool,
        device: Device,
        verbose: bool,
    ) -> None:
        self.model, batch_delay = select_model(
            model_name_or_path=model_name_or_path,
            batch_size=max_batch_size,
            capable_engine=capable_engine,
            model_warmup=model_warmup,
            device=device,
        )
        self._batch_delay = float(max(1e-4, batch_delay))

        if batch_delay > 0.1:
            logger.warn(f"high batch delay of {self._batch_delay}")
        self._shutdown = shutdown_event
        self.max_batch_size = max_batch_size

        self._shared_in_queue: Queue = in_queue
        self._feature_queue: Queue = Queue(6)
        self._postprocess_queue: Queue = Queue(4)
        self._shared_out_queue = out_queue

        self._verbose = verbose
        self._last_inference = time.perf_counter()

    def _general_batch(
        self,
        alias_name: str,
        in_queue: Queue,
        out_queue: Queue,
        batch_fn: Any,
        is_model_fn: bool = False,
        is_post_fn: bool = False,
    ):
        logger.debug(f"starting {alias_name} in ModelWorker")
        try:
            while not self._shutdown.is_set():
                try:
                    fetched_batch = in_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                feat, meta = fetched_batch

                if self._verbose:
                    logger.debug("[🏃] %s on batch_size=%s", alias_name, len(meta))

                if is_model_fn:
                    self._last_inference = time.perf_counter()
                elif self._last_inference < time.perf_counter() + self._batch_delay:
                    # 5 ms, assuming this is below
                    # 3-50ms for inference on avg.
                    # give the CPU some time to focus
                    # on moving the next batch to GPU on the forward pass
                    # before proceeding
                    time.sleep(self._batch_delay)

                processed = batch_fn(feat)

                if is_post_fn:
                    for i, item in enumerate(meta):
                        item.set_result(processed[i])
                    processed = None

                # while-loop just for shutdown
                while not self._shutdown.is_set():
                    try:
                        out_queue.put((processed, meta), timeout=1)
                        break
                    except queue.Full:
                        continue
                in_queue.task_done()
        except Exception as ex:
            logger.exception(ex)
            self._ready = False
            raise ValueError(f"{alias_name} crashed.")
        self._ready = False

    async def spawn(self):
        """set up the resources in batch"""
        logger.info("creating ModelWorker")
        self.tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._general_batch,
                    "preprocess",
                    self._shared_in_queue,
                    self._feature_queue,
                    self.model.encode_pre,
                )
            ),
            asyncio.create_task(
                asyncio.to_thread(
                    self._general_batch,
                    "forward",
                    self._feature_queue,
                    self._postprocess_queue,
                    self.model.encode_core,
                    is_model_fn=True,
                )
            ),
            asyncio.create_task(
                asyncio.to_thread(
                    self._general_batch,
                    "postprocess",
                    self._postprocess_queue,
                    self._shared_out_queue,
                    self.model.encode_post,
                    is_post_fn=True,
                )
            ),
        ]

    async def shutdown(self):
        """
        set the shutdown event.
        Blocking event, until shutdown.
        """
        self._shutdown.set()

        await asyncio.gather(*self.tasks)
