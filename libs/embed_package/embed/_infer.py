from concurrent.futures import Future
from typing import Collection, Literal, Union

from infinity_emb import EngineArgs, SyncEngineArray  # type: ignore
from infinity_emb.infinity_server import AutoPadding

__all__ = ["BatchedInference"]

Device = Literal["cpu", "cuda"]
ModelID = str
Engine = Literal["torch", "optimum"]
EmbeddingDtype = Literal["float32", "int8", "binary"]
ModelIndex = Union[int, str]


class BatchedInference:
    def __init__(
        self,
        *,
        model_id: Union[ModelID, Collection[ModelID]],
        engine: Union[Engine, Collection[Engine]] = "optimum",
        device: Union[Device, Collection[Device]] = "cpu",
        embedding_dtype: Union[EmbeddingDtype, Collection[EmbeddingDtype]] = "float32",
    ):
        """An easy interface to infer with multiple models.
        >>> ei = BatchedInference(model_id=['michaelfeil/bge-small-en-v1.5','mixedbread-ai/mxbai-rerank-xsmall-v1'])
        >>> ei
        BatchedInference(['michaelfeil/bge-small-en-v1.5', 'mixedbread-ai/mxbai-rerank-xsmall-v1'])
        >>> ei.stop() # always stop when you are done
        """
        if isinstance(model_id, str):
            model_id = [model_id]
        if isinstance(engine, str):
            engine = [engine]
        if isinstance(device, str):
            device = [device]
        if isinstance(embedding_dtype, str):
            embedding_dtype = [embedding_dtype]
        EngineArgs()
        pad = AutoPadding(
            length=len(model_id),
            # pass through arguments
            model_name_or_path=model_id,
            engine=engine,
            device=device,
            embedding_dtype=embedding_dtype,
            # optinionated defaults
            lengths_via_tokenize=True,
            model_warmup=False,
            trust_remote_code=True,
        )
        self._engine_args = [EngineArgs(**kwargs) for kwargs in pad]
        self._engine_array = SyncEngineArray.from_args(self._engine_args)

    def stop(self):
        self._engine_array.stop()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[a.model_name_or_path for a in self._engine_args]})"

    def embed(
        self,
        *,
        sentences: list[str],
        model_id: ModelIndex = 0,
    ) -> Future[tuple[list[list[float]], int]]:
        """Embed sentences with a model.

        >>> ei = BatchedInference(model_id="michaelfeil/bge-small-en-v1.5", engine="torch")
        >>> embed_result = ei.embed(model_id="michaelfeil/bge-small-en-v1.5", sentences=["Hello, world!"])
        >>> type(embed_result)
        <class 'concurrent.futures._base.Future'>
        >>> embed_result.result()[0][0].shape # embedding
        (384,)
        >>> embed_result.result()[1] # embedding and usage of 6 tokens
        4
        >>> ei.stop()
        """
        return self._engine_array.embed(model=model_id, sentences=sentences)

    def image_embed(
        self,
        *,
        images: list[str],
        model_id: ModelIndex = 0,
    ) -> Future[tuple[list[list[float]], int]]:
        """Embed images with a model.

        >>> ei = BatchedInference(model_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", engine="torch")
        >>> image_embed_result = ei.image_embed(model_id="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M", images=["http://images.cocodataset.org/val2017/000000039769.jpg"])
        >>> type(image_embed_result)
        <class 'concurrent.futures._base.Future'>
        >>> image_embed_result.result()[0][0].shape
        (512,)
        >>> ei.stop()
        """
        return self._engine_array.image_embed(model=model_id, images=images)

    def classify(
        self,
        *,
        sentences: list[str],
        model_id: ModelIndex = 0,
    ) -> Future[tuple[list[list[dict[str, float]]], int]]:
        """Classify sentences with a model.

        >>> ei = BatchedInference(model_id="philschmid/tiny-bert-sst2-distilled", engine="torch")
        >>> classify_result = ei.classify(model_id="philschmid/tiny-bert-sst2-distilled", sentences=["I love this movie"])
        >>> type(classify_result)
        <class 'concurrent.futures._base.Future'>
        >>> label_0 = classify_result.result()[0][0][0]
        >>> label_0["label"], round(label_0["score"], 4)
        ('positive', 0.9996)
        >>> ei.stop()
        """
        return self._engine_array.classify(model=model_id, sentences=sentences)

    def rerank(
        self,
        *,
        query: str,
        docs: list[str],
        model_id: ModelIndex = 0,
    ) -> Future[list[str]]:
        """

        >>> ei = BatchedInference(model_id="mixedbread-ai/mxbai-rerank-xsmall-v1")
        >>> docs = ["Paris is nice", "Paris is in France", "In Germany"]
        >>> rerank_result = ei.rerank(model_id="mixedbread-ai/mxbai-rerank-xsmall-v1", query="Where is Paris?", docs=docs)
        >>> type(rerank_result)
        <class 'concurrent.futures._base.Future'>
        >>> [round(score, 3) for score in rerank_result.result()[0]]
        [0.288, 0.742, 0.022]
        >>> ei.stop()
        """
        return self._engine_array.rerank(model=model_id, query=query, docs=docs)
