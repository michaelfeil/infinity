import asyncio
import threading
from concurrent.futures import Future
from functools import partial
from typing import TYPE_CHECKING, Awaitable, Callable, Iterator, TypeVar

from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray, EngineArgs
from infinity_emb.log_handler import logger

if TYPE_CHECKING:
    from infinity_emb import AsyncEmbeddingEngine


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


T = TypeVar("T")


class AsyncLifeMixin:
    def __init__(self) -> None:
        self.__lock = threading.Lock()
        self.__stop_signal = threading.Event()
        self.__loop: asyncio.AbstractEventLoop = None  # type: ignore
        # init
        self.__is_closed: Future = Future()
        self.__is_closed.set_result(None)
        self.async_start_loop()

    def __async_lifetime(self, start_event: Future):
        """private function, takes care of starting, stopping event loop"""
        self.__loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.__loop)

        async def block_until_engine_stop():
            logger.info("Started Background Event Loop")
            start_event.set_result(None)  # signal that the event loop has started
            while not self.__stop_signal.is_set():
                await asyncio.sleep(0.2)

        self.__loop.run_until_complete(block_until_engine_stop())
        self.__loop.close()
        self.__is_closed.set_result(None)
        logger.info("Closed Background Event Loop")

    def is_async_loop_running(self):
        return (
            (not self.__stop_signal.is_set())
            and self.__loop is not None
            and self.__loop.is_running()
        )

    def async_start_loop(self):
        self.async_close_loop()
        with self.__lock:
            start_event = Future()
            self.__stop_signal.clear()
            self.__is_closed = Future()
            threading.Thread(
                target=partial(self.__async_lifetime, start_event=start_event),
                daemon=True,
            ).start()
            start_event.result()

    def async_close_loop(self):
        """closes the event loop. This is a blocking call"""
        with self.__lock:
            self.__stop_signal.set()
            self.__is_closed.result()

    def async_run(
        self,
        async_function: Callable[..., Awaitable[T]],
        *funcion_args,
        **function_kwargs
    ) -> Future[T]:
        """run an async function in the background event loop.

        Args:
            async_function: the async function to run
            funcion_args: args to pass to the async function
            function_kwargs: kwargs to pass to the async function

        Returns:
            concurrent.futures.Future returning the result of async_function.
        """
        if not self.is_async_loop_running():
            raise RuntimeError("Event loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            async_function(*funcion_args, **function_kwargs), self.__loop
        )
        return future


@add_start_docstrings(AsyncEngineArray.__doc__)
class SyncEngineArray(AsyncLifeMixin):
    def __init__(self, _engine_args_array: list[EngineArgs]):
        super().__init__()
        self.async_engine_array = AsyncEngineArray.from_args(_engine_args_array)
        self.async_run(self.async_engine_array.astart).result()

    @classmethod
    def from_args(cls, engine_args_array: list[EngineArgs]) -> "SyncEngineArray":
        return cls(_engine_args_array=engine_args_array)

    @property
    def is_running(self):
        return self.async_engine_array.is_running

    def __iter__(self) -> Iterator["AsyncEmbeddingEngine"]:
        return iter(self.async_engine_array)

    def stop(self):
        """blocks until the engine is stopped"""
        self.async_run(self.async_engine_array.astop).result()
        self.async_close_loop()

    @add_start_docstrings(AsyncEngineArray.embed.__doc__)
    def embed(self, *, model: str, sentences: list[str]):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.embed, model=model, sentences=sentences
        )

    @add_start_docstrings(AsyncEngineArray.rerank.__doc__)
    def rerank(
        self, *, model: str, query: str, docs: list[str], raw_scores: bool = False
    ):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.rerank,
            model=model,
            query=query,
            docs=docs,
            raw_scores=raw_scores,
        )

    @add_start_docstrings(AsyncEngineArray.classify.__doc__)
    def classify(self, *, model: str, sentences: list[str], raw_scores: bool = False):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.classify,
            model=model,
            sentences=sentences,
            raw_scores=raw_scores,
        )

    @add_start_docstrings(AsyncEngineArray.image_embed.__doc__)
    def image_embed(self, *, model: str, images: list[str]):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.image_embed, model=model, images=images
        )
