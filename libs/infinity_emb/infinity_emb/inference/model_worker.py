import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue as MpQueue
from queue import Empty, Full, Queue
from typing import Any, Union

from infinity_emb.inference.select_model import (
    select_model,
)
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, QueueSignalMessages
from infinity_emb.transformer.utils import (
    CapableEngineType,
)

AnyQueue = Union[Queue, MpQueue]


class ModelWorker:
    def __init__(
        self,
        in_queue: AnyQueue,
        out_queue: AnyQueue,
        model_name_or_path: str,
        max_batch_size: int,
        capable_engine: CapableEngineType,
        model_warmup: bool,
        device: Device,
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
        self.max_batch_size = max_batch_size

        self._shared_in_queue = in_queue
        self._feature_queue: Queue = Queue(6)
        self._postprocess_queue: Queue = Queue(4)
        self._shared_out_queue = out_queue

        self._verbose = logger.level <= 5
        self._last_inference = time.perf_counter()

        # spawn threads
        logger.info("creating ModelWorker")

        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [
                pool.submit(
                    self._general_batch,
                    "preprocess",
                    self._shared_in_queue,
                    self._feature_queue,
                    self.model.encode_pre,
                ),
                pool.submit(
                    self._general_batch,
                    "forward",
                    self._feature_queue,
                    self._postprocess_queue,
                    self.model.encode_core,
                    is_model_fn=True,
                ),
                pool.submit(
                    self._general_batch,
                    "postprocess",
                    self._postprocess_queue,
                    self._shared_out_queue,
                    self.model.encode_post,
                    is_post_fn=True,
                ),
            ]
            # block until all tasks are done
            for future in tasks:
                future.result()

            logger.info("stopped ModelWorker")

    def _general_batch(
        self,
        alias_name: str,
        in_queue: AnyQueue,
        out_queue: AnyQueue,
        batch_fn: Any,
        is_model_fn: bool = False,
        is_post_fn: bool = False,
    ):
        try:
            while True:
                fetched_batch = in_queue.get()
                if type(in_queue) == Queue:
                    in_queue.task_done()
                if fetched_batch == QueueSignalMessages.KILL:
                    self.destruct()
                    break

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

                out_queue.put((processed, meta))

        except Exception as ex:
            logger.exception(ex)
            self.destruct()
            raise ValueError(f"{alias_name} crashed.")

    def destruct(self):
        """kill all tasks"""
        for q in [
            self._shared_in_queue,
            self._postprocess_queue,
            self._feature_queue,
            self._shared_out_queue,
        ]:
            while not q.empty():
                try:
                    res = q.get_nowait()
                    if res == QueueSignalMessages.KILL:
                        break
                    if type(q) == Queue:
                        q.task_done()
                except Empty:
                    pass
            for _ in range(10):
                try:
                    q.put_nowait(QueueSignalMessages.KILL)
                except Full:
                    pass
