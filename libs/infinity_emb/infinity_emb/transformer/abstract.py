from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Dict, List, Set, Tuple

from infinity_emb.primitives import (
    EmbeddingInner,
    EmbeddingReturnType,
    EmbeddingSingle,
    ModelCapabilites,
    PredictInner,
    PredictSingle,
    ReRankInner,
    ReRankSingle,
)

INPUT_FEATURE = Any
OUT_FEATURES = Any


class BaseTransformer(ABC):  # Inherit from ABC(Abstract base class)
    capabilities: Set[ModelCapabilites] = set()

    @abstractmethod
    def encode_core(self, features: INPUT_FEATURE) -> OUT_FEATURES:
        """runs plain inference, on cpu/gpu"""

    @abstractmethod
    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""

    @abstractmethod
    def warmup(self, batch_size: int = 64, n_tokens=1) -> Tuple[float, float, str]:
        """warmup the model

        returns embeddings per second, inference time, and a log message"""


class BaseEmbedder(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"embed"}

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: List[str]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> EmbeddingReturnType:
        """runs post encoding such as normlization"""

    def warmup(self, batch_size: int = 64, n_tokens=1) -> Tuple[float, float, str]:
        sample = ["warm" * n_tokens] * batch_size
        inp = [
            EmbeddingInner(content=EmbeddingSingle(s), future=None)  # type: ignore
            for s in sample
        ]
        return run_warmup(self, inp)


class BaseClassifer(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"classify"}

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: List[str]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> Dict[str, float]:
        """runs post encoding such as normlization"""

    def warmup(self, batch_size: int = 64, n_tokens=1) -> Tuple[float, float, str]:
        sample = ["warm" * n_tokens] * batch_size
        inp = [
            PredictInner(content=PredictSingle(s), future=None)  # type: ignore
            for s in sample
        ]
        return run_warmup(self, inp)


class BaseCrossEncoder(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"rerank"}

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, queries_docs: List[Tuple[str, str]]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> List[float]:
        """runs post encoding such as normlization"""

    def warmup(self, batch_size: int = 64, n_tokens=1) -> Tuple[float, float, str]:
        sample = ["warm" * n_tokens] * batch_size
        inp = [
            ReRankInner(
                content=ReRankSingle(query=s, document=s), future=None  # type: ignore
            )
            for s in sample
        ]
        return run_warmup(self, inp)


def run_warmup(model, inputs) -> Tuple[float, float, str]:
    inputs_formated = [i.content.to_input() for i in inputs]
    start = perf_counter()
    feat = model.encode_pre(inputs_formated)
    tokenization_time = perf_counter()
    embed = model.encode_core(feat)
    inference_time = perf_counter()
    model.encode_post(embed)
    post_time = perf_counter()

    analytics_string = (
        f"Getting timings for batch_size={len(inputs)}"
        " and avg tokens per sentence="
        f"{model.tokenize_lengths([i.content.str_repr() for i in inputs])[0]}\n"
        f"\t{(tokenization_time - start)*1000:.2f} \t ms tokenization\n"
        f"\t{(inference_time-tokenization_time)*1000:.2f} \t ms inference\n"
        f"\t{(post_time-inference_time)*1000:.2f} \t ms post-processing\n"
        f"\t{(post_time - start)*1000:.2f} \t ms total\n"
        f"embeddings/sec: {len(inputs) / (post_time - start):.2f}"
    )

    emb_per_sec = len(inputs) / (post_time - start)
    return emb_per_sec, inference_time - tokenization_time, analytics_string
