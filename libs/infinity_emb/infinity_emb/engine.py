from asyncio import Semaphore
from typing import Iterable, Iterator, Optional, Set, Union

from infinity_emb.args import EngineArgs

# prometheus
from infinity_emb.inference import (
    BatchHandler,
    select_model,
)
from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    ClassifyReturnType,
    EmbeddingReturnType,
    ModelCapabilites,
)


class AsyncEmbeddingEngine:
    """
    An LLM engine that receives requests and embeds them asynchronously.

    This is the main worker of the infinity-emb library. It is responsible for
    handling the requests and embedding them asynchronously.

    Initialize via `from_args` method.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        _show_deprecation_warning=True,
        **kwargs,
    ) -> None:
        """Creating a Async EmbeddingEngine object.
        preferred way to create an engine is via `from_args` method.
        """
        # TODO: remove _show_deprecation_warning and __init__ option.
        if _show_deprecation_warning:
            logger.warning(
                "AsyncEmbeddingEngine() is deprecated since 0.0.25. "
                "Use `AsyncEmbeddingEngine.from_args()` instead"
            )
        if model_name_or_path is not None:
            kwargs["model_name_or_path"] = model_name_or_path
        self._engine_args = EngineArgs(**kwargs)

        self.running = False
        self._running_sepamore = Semaphore(1)
        self._model, self._min_inference_t, self._max_inference_t = select_model(
            self._engine_args
        )

    @classmethod
    def from_args(
        cls,
        engine_args: EngineArgs,
    ) -> "AsyncEmbeddingEngine":
        """create an engine from EngineArgs

        Args:
            engine_args (EngineArgs): EngineArgs object
        """
        engine = cls(**engine_args.to_dict(), _show_deprecation_warning=False)

        return engine

    def __str__(self) -> str:
        return (
            f"AsyncEmbeddingEngine(running={self.running}, "
            f"inference_time={[self._min_inference_t, self._max_inference_t]}, "
            f"{self._engine_args})"
        )

    async def astart(self):
        """startup engine"""
        async with self._running_sepamore:
            if not self.running:
                self.running = True
                self._batch_handler = BatchHandler(
                    max_batch_size=self._engine_args.batch_size,
                    model=self._model,
                    batch_delay=self._min_inference_t / 2,
                    vector_disk_cache_path=self._engine_args.vector_disk_cache_path,
                    verbose=logger.level <= 10,
                    lengths_via_tokenize=self._engine_args.lengths_via_tokenize,
                )
                await self._batch_handler.spawn()

    async def astop(self):
        """stop engine"""
        async with self._running_sepamore:
            if self.running:
                self.running = False
                await self._batch_handler.shutdown()

    async def __aenter__(self):
        await self.astart()

    async def __aexit__(self, *args):
        await self.astop()

    def overload_status(self):
        self._assert_running()
        return self._batch_handler.overload_status()

    def is_overloaded(self) -> bool:
        self._assert_running()
        return self._batch_handler.is_overloaded()

    @property
    def is_running(self) -> bool:
        return self.running

    @property
    def capabilities(self) -> Set[ModelCapabilites]:
        return self._model.capabilities

    @property
    def engine_args(self) -> EngineArgs:
        return self._engine_args

    async def embed(
        self, sentences: list[str]
    ) -> tuple[list[EmbeddingReturnType], int]:
        """embed multiple sentences

        Args:
            sentences (list[str]): sentences to be embedded

        Raises:
            ValueError: raised if engine is not started yet
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list[EmbeddingReturnType]: embeddings
                2D list-array of shape( len(sentences),embed_dim )
            int: token usage
        """

        self._assert_running()
        embeddings, usage = await self._batch_handler.embed(sentences)
        return embeddings, usage

    async def rerank(
        self, *, query: str, docs: list[str], raw_scores: bool = False
    ) -> tuple[list[float], int]:
        """rerank multiple sentences

        Args:
            query (str): query to be reranked
            docs (list[str]): docs to be reranked
            raw_scores (bool): return raw scores instead of sigmoid

        Raises:
            ValueError: raised if engine is not started yet
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list[float]: list of scores
            int: token usage
        """
        self._assert_running()
        scores, usage = await self._batch_handler.rerank(
            query=query, docs=docs, raw_scores=raw_scores
        )

        return scores, usage

    async def classify(
        self, *, sentences: list[str], raw_scores: bool = False
    ) -> tuple[list[ClassifyReturnType], int]:
        """classify multiple sentences

        Args:
            sentences (list[str]): sentences to be classified
            raw_scores (bool): if True, return raw scores, else softmax

        Raises:
            ValueError: raised if engine is not started yet
            ModelNotDeployedError: If loaded model does not expose `embed`
                capabilities

        Returns:
            list[ClassifyReturnType]: list of class encodings
            int: token usage
        """
        self._assert_running()
        scores, usage = await self._batch_handler.classify(
            sentences=sentences, raw_scores=raw_scores
        )

        return scores, usage

    def _assert_running(self):
        if not self.running:
            raise ValueError(
                "didn't start `AsyncEmbeddingEngine` "
                " recommended use is via AsyncContextManager"
                " `async with engine: ..`"
            )


class AsyncEngineArray:
    """EngineArray is a collection of AsyncEmbeddingEngine objects."""

    def __init__(self, engines: Iterable["AsyncEmbeddingEngine"]):
        if not engines:
            raise ValueError("Engines cannot be empty")
        if len(list(engines)) != len(
            set(engine.engine_args.served_model_name for engine in engines)
        ):
            raise ValueError("Engines must have unique model names")
        self.engines_dict = {
            engine.engine_args.served_model_name: engine for engine in engines
        }

    @classmethod
    def from_args(cls, engine_args_array: Iterable[EngineArgs]) -> "AsyncEngineArray":
        """create an engine from EngineArgs

        Args:
            engine_args_array (list[EngineArgs]): EngineArgs object
        """
        return cls(
            engines=tuple(
                AsyncEmbeddingEngine.from_args(engine_args)
                for engine_args in engine_args_array
            )
        )

    def __iter__(self) -> Iterator["AsyncEmbeddingEngine"]:
        return iter(self.engines_dict.values())

    async def astart(self):
        """startup engines"""
        for engine in self.engines_dict.values():
            await engine.astart()

    async def astop(self):
        """stop engines"""
        for engine in self.engines_dict.values():
            await engine.astop()

    def __getitem__(self, index_or_name: Union[str, int]) -> "AsyncEmbeddingEngine":
        """resolve engine by model name -> Auto resolve if only one engine is present

        Args:
            model_name (str): model name to be used
        """
        if len(self.engines_dict) == 1:
            return list(self.engines_dict.values())[0]
        if isinstance(index_or_name, int):
            return list(self.engines_dict.values())[index_or_name]
        if isinstance(index_or_name, str) and index_or_name in self.engines_dict:
            return self.engines_dict[index_or_name]
        raise IndexError(
            f"Engine for model name `{index_or_name}` not found. "
            f"Available model names are {list(self.engines_dict.keys())}"
        )
