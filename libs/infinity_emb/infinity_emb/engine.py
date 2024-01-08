from typing import Dict, List, Set, Tuple, Union

# prometheus
from infinity_emb.inference import (
    BatchHandler,
    Device,
)
from infinity_emb.log_handler import logger
from infinity_emb.primitives import EmbeddingReturnType, ModelCapabilites
from infinity_emb.transformer.utils import InferenceEngine


class AsyncEmbeddingEngine:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        engine: Union[InferenceEngine, str] = InferenceEngine.torch,
        model_warmup: bool = False,
        vector_disk_cache_path: str = "",
        device: Union[Device, str] = Device.auto,
        lengths_via_tokenize: bool = False,
    ) -> None:
        """Creating a Async EmbeddingEngine object.

        Args:
            model_name_or_path, str:  Defaults to "BAAI/bge-small-en-v1.5".
            batch_size, int: Defaults to 64.
            engine, InferenceEngine: backend for inference.
                Defaults to InferenceEngine.torch.
            model_warmup, bool: decide if warmup with max batch size . Defaults to True.
            vector_disk_cache_path, str: file path to folder of cache.
                Defaults to "" - default no caching.
            device, Device: device to use for inference. Defaults to Device.auto,
            lengths_via_tokenize, bool: schedule by token usage. Defaults to False

        Example:
            ```python
            from infinity_emb import AsyncEmbeddingEngine, transformer
            sentences = ["Embedded this via Infinity.", "Paris is in France."]
            engine = AsyncEmbeddingEngine(engine="torch")
            async with engine: # engine starts with engine.astart()
                embeddings = np.array(await engine.embed(sentences))
            # engine stops with engine.astop().
            # For frequent restarts, handle start/stop yourself.
            ```
        """
        self.batch_size = batch_size
        self.running = False
        self._vector_disk_cache_path = vector_disk_cache_path
        self._model_name_or_path = model_name_or_path
        self._model_name_or_pathengine = engine
        self._model_warmup = model_warmup
        self._lengths_via_tokenize = lengths_via_tokenize

        if isinstance(engine, str):
            self._engine_type = InferenceEngine[engine]
        else:
            self._engine_type = engine
        if isinstance(device, str):
            self.device = Device[device]
        else:
            self.device = device

    async def astart(self):
        """startup engine"""
        if self.running:
            raise ValueError(
                "DoubleSpawn: already started `AsyncEmbeddingEngine`. "
                " recommended use is via AsyncContextManager"
                " `async with engine: ..`"
            )
        self.running = True
        self._batch_handler = BatchHandler(
            model_name_or_path=self._model_name_or_path,
            engine=self._engine_type,
            max_batch_size=self.batch_size,
            model_warmup=self._model_warmup,
            vector_disk_cache_path=self._vector_disk_cache_path,
            verbose=logger.level <= 10,
            lengths_via_tokenize=self._lengths_via_tokenize,
            device=self.device,
        )
        await self._batch_handler.astart()

    async def astop(self):
        """stop engine"""
        self._assert_running()
        self.running = False
        await self._batch_handler.astop()
        self._batch_handler = None

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
    def capabilities(self) -> Set[ModelCapabilites]:
        self._assert_running()
        return self._batch_handler.capabilities

    async def embed(
        self, sentences: List[str]
    ) -> Tuple[List[EmbeddingReturnType], int]:
        """embed multiple sentences

        Args:
            sentences (List[str]): sentences to be embedded

        Raises:
            ValueError: raised if engine is not started yet"

        Returns:
            List[numpy.ndarray]: embeddings
                2D list-array of shape( len(sentences),embed_dim )
            Usage:
        """

        self._assert_running()
        embeddings, usage = await self._batch_handler.embed(sentences)
        return embeddings, usage

    async def rerank(
        self, *, query: str, docs: List[str], raw_scores: bool = False
    ) -> Tuple[List[float], int]:
        """rerank multiple sentences

        Args:
            query (str): query to be reranked
            docs (List[str]): docs to be reranked
            raw_scores (bool): return raw scores instead of sigmoid
        """
        self._assert_running()
        scores, usage = await self._batch_handler.rerank(
            query=query, docs=docs, raw_scores=raw_scores
        )

        return scores, usage

    async def classify(
        self, *, sentences: List[str], raw_scores: bool = False
    ) -> Tuple[List[Dict[str, float]], int]:
        """rerank multiple sentences

        Args:
            query (str): query to be reranked
            docs (List[str]): docs to be reranked
            raw_scores (bool): return raw scores instead of sigmoid
        """
        self._assert_running()
        scores, usage = await self._batch_handler.classify(sentences=sentences)

        return scores, usage

    def _assert_running(self):
        if not self.running:
            raise ValueError(
                "didn't start `AsyncEmbeddingEngine` "
                " recommended use is via AsyncContextManager"
                " `async with engine: ..`"
            )
