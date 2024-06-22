import asyncio
import threading
import time
from concurrent.futures import Future
from typing import Iterator

from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray, EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import ClassifyReturnType, EmbeddingDtype, ReRankReturnType


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def threaded_asyncio_executor():
    def decorator(fn):
        funcname = fn.__name__  # e.g. `embed`

        def wrapper(self: "SyncEngineArray", **kwargs) -> "Future":
            future: Future = Future()

            assert self.is_running, "SyncEngineArray is not running"

            def execute():
                async_function = getattr(self.async_engine_array, funcname)
                try:
                    # async_function is e.g. `self.async_engine_array.embed`
                    # get async future object
                    result = asyncio.run_coroutine_threadsafe(
                        async_function(**kwargs), self._loop
                    )
                    # block until the result is available
                    future.set_result(result.result())
                except Exception as e:
                    future.set_exception(e)

            threading.Thread(target=execute).start()
            return future  # return the future object immediately

        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator


@add_start_docstrings(AsyncEngineArray.__doc__)
class SyncEngineArray:
    def __init__(self, engine_args: list[EngineArgs]):
        self._start_event = threading.Event()
        self._stop_event = threading.Event()
        self.async_engine_array = AsyncEngineArray.from_args(engine_args)
        threading.Thread(target=self._lifetime).start()
        self._start_event.wait()  # wait until the event loop has started

    @classmethod
    def from_args(cls, engine_args: list[EngineArgs]) -> "SyncEngineArray":
        return cls(engine_args)

    @property
    def is_running(self):
        return (
            not self._stop_event.is_set()
            and self._loop.is_running()
            and self.async_engine_array.is_running
        )

    def __iter__(self) -> Iterator["AsyncEmbeddingEngine"]:
        return iter(self.async_engine_array)

    def stop(self):
        """blocks until the engine is stopped"""
        self._stop_event.set()
        while self._loop.is_running():
            time.sleep(0.05)

    def _lifetime(self):
        """takes care of starting, stopping (engine and event loop)"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def block_until_engine_stop():
            logger.info("Started SyncEngineArray Background Event Loop")
            self._start_event.set()  # signal that the event loop has started
            try:
                await self.async_engine_array.astart()
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.2)
            finally:
                await self.async_engine_array.astop()
            # additional delay to ensure that the engine is stopped
            await asyncio.sleep(2.0)

        self._loop.run_until_complete(block_until_engine_stop())
        self._loop.close()
        logger.info("Closed SyncEngineArray Background Event Loop")

    @add_start_docstrings(AsyncEngineArray.embed.__doc__)
    @threaded_asyncio_executor()
    def embed(self, *, model: str, sentences: list[str]) -> Future[EmbeddingDtype]:
        """sync interface of AsyncEngineArray"""
        return None  # type: ignore

    @add_start_docstrings(AsyncEngineArray.rerank.__doc__)
    @threaded_asyncio_executor()
    def rerank(
        self, *, model: str, query: str, docs: list[str]
    ) -> Future[ReRankReturnType]:
        """sync interface of AsyncEngineArray"""
        return None  # type: ignore

    @add_start_docstrings(AsyncEngineArray.classify.__doc__)
    @threaded_asyncio_executor()
    def classify(self, *, model: str, text: str) -> Future[ClassifyReturnType]:
        """sync interface of AsyncEngineArray"""
        return None  # type: ignore

    @add_start_docstrings(AsyncEngineArray.image_embed.__doc__)
    @threaded_asyncio_executor()
    def image_embed(self, *, model: str, images: list[str]) -> Future[EmbeddingDtype]:
        """sync interface of AsyncEngineArray"""
        return None  # type: ignore
