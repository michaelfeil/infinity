from typing import List

import numpy as np

from infinity_emb.inference.primitives import NpEmbeddingType
from infinity_emb.transformer.abstract import BaseTransformer


class DummyTransformer(BaseTransformer):
    """fix-13 dimension embedding, filled with length of sentence"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode_pre(self, sentences: List[str]) -> np.ndarray:
        return np.asarray(sentences)

    def encode_core(self, features: np.ndarray) -> NpEmbeddingType:
        lengths = np.array([[len(s) for s in features]])
        # embedding of size 13
        return np.ones([len(features), 13]) * lengths.T

    def encode_post(self, embedding: NpEmbeddingType):
        return embedding

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        return [len(s) for s in sentences]
