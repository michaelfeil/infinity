import asyncio
import threading
from concurrent.futures import Future
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
        self.__start_event: Future = Future()
        self.__stop_event = threading.Event()
        self.__is_closed: Future = Future()
        threading.Thread(target=self.__async_lifetime, daemon=True).start()
        self.__start_event.result()

    def __async_lifetime(self):
        """takes care of starting, stopping event loop"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def block_until_engine_stop():
            logger.info("Started Background Event Loop")
            self.__start_event.set_result(
                None
            )  # signal that the event loop has started
            while not self.__stop_event.is_set():
                await asyncio.sleep(0.2)

        self._loop.run_until_complete(block_until_engine_stop())
        self._loop.close()
        self.__is_closed.set_result(None)
        logger.info("Closed Background Event Loop")

    def async_close_loop(self):
        """closes the event loop forever. This is a blocking call"""
        self.__stop_event.set()
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
        if not self._loop.is_running() or self.__stop_event.is_set():
            raise RuntimeError("Loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            async_function(*funcion_args, **function_kwargs), self._loop
        )
        return future


@add_start_docstrings(AsyncEngineArray.__doc__)
class SyncEngineArray(AsyncLifeMixin):
    def __init__(self, engine_args: list[EngineArgs]):
        super().__init__()
        self.async_engine_array = AsyncEngineArray.from_args(engine_args)
        self.async_run(self.async_engine_array.astart).result()

    @classmethod
    def from_args(cls, engine_args: list[EngineArgs]) -> "SyncEngineArray":
        return cls(engine_args)

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
    def rerank(self, *, model: str, query: str, docs: list[str]):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.rerank, model=model, query=query, docs=docs
        )

    @add_start_docstrings(AsyncEngineArray.classify.__doc__)
    def classify(self, *, model: str, text: str):
        """sync interface of AsyncEngineArray"""
        return self.async_run(self.async_engine_array.classify, model=model, text=text)

    @add_start_docstrings(AsyncEngineArray.image_embed.__doc__)
    def image_embed(self, *, model: str, images: list[str]):
        """sync interface of AsyncEngineArray"""
        return self.async_run(
            self.async_engine_array.image_embed, model=model, images=images
        )
