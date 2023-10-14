from abc import ABC, abstractmethod
from typing import Any, List

from infinity_emb.inference.primitives import NpEmbeddingType

INPUT_FEATURE = Any
OUT_FEATURES = Any


class BaseTransformer(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: List[str]) -> INPUT_FEATURE:
        pass

    @abstractmethod
    def encode_core(self, features: INPUT_FEATURE) -> OUT_FEATURES:
        pass

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> NpEmbeddingType:
        pass

    @abstractmethod
    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        pass
