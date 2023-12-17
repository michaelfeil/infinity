from typing import List

import numpy as np

from infinity_emb.primitives import EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder


class DummyTransformer(BaseEmbedder):
    """fix-13 dimension embedding, filled with length of sentence"""

    def __init__(self, *args, **kwargs) -> None:
        print(f"running DummyTransformer.__init__ with args={args} and kwargs={kwargs}")

    def encode_pre(self, sentences: List[str]) -> np.ndarray:
        return np.asarray(sentences)

    def encode_core(self, features: np.ndarray) -> EmbeddingReturnType:
        lengths = np.array([[len(s) for s in features]])
        # embedding of size 13
        return np.ones([len(features), 13]) * lengths.T

    def encode_post(self, embedding: EmbeddingReturnType):
        return embedding

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        return [len(s) for s in sentences]
