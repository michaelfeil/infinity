from abc import ABC, abstractmethod
from typing import Any, List

from infinity_emb.inference.primitives import NpEmbeddingType

INPUT_FEATURE = Any
OUT_FEATURES = Any


class BaseTransformer(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: List[str]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_core(self, features: INPUT_FEATURE) -> OUT_FEATURES:
        """runs plain inference, on cpu/gpu"""

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> NpEmbeddingType:
        """runs post encoding such as normlization"""

    @abstractmethod
    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
