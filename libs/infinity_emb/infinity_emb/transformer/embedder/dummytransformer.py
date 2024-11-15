# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import numpy as np

from infinity_emb.args import EngineArgs
from infinity_emb.primitives import EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.quantization.interface import quant_embedding_decorator


class DummyTransformer(BaseEmbedder):
    """fix-13 dimension embedding, filled with length of sentence"""

    def __init__(self, *, engine_args: EngineArgs) -> None:
        print(f"running DummyTransformer.__init__ with engine_args={engine_args}")
        self.engine_args = engine_args

    def encode_pre(self, sentences: list[str]) -> np.ndarray:
        return np.asarray(sentences)

    def encode_core(self, features: np.ndarray) -> EmbeddingReturnType:
        lengths = np.array([[len(s) for s in features]])
        # embedding of size 13
        return np.ones([len(features), 13]) * lengths.T

    @quant_embedding_decorator()
    def encode_post(self, embedding: EmbeddingReturnType):
        return [e for e in embedding]

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        return [len(s) for s in sentences]
