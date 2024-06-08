from enum import Enum
from typing import Callable

from infinity_emb.primitives import InferenceEngine
from infinity_emb.transformer.classifier.torch import SentenceClassifier
from infinity_emb.transformer.crossencoder.optimum import OptimumCrossEncoder
from infinity_emb.transformer.crossencoder.torch import (
    CrossEncoderPatched as CrossEncoderTorch,
)
from infinity_emb.transformer.embedder.ct2 import CT2SentenceTransformer
from infinity_emb.transformer.embedder.dummytransformer import DummyTransformer
from infinity_emb.transformer.embedder.neuron import NeuronOptimumEmbedder
from infinity_emb.transformer.embedder.optimum import OptimumEmbedder
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)
from infinity_emb.transformer.vision.torch_vision import ClipLikeModel

__all__ = [
    "length_tokenizer",
    "get_lengths_with_tokenize",
]


class EmbedderEngine(Enum):
    torch = SentenceTransformerPatched
    ctranslate2 = CT2SentenceTransformer
    debugengine = DummyTransformer
    optimum = OptimumEmbedder
    neuron = NeuronOptimumEmbedder

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return EmbedderEngine.torch
        elif engine == InferenceEngine.ctranslate2:
            return EmbedderEngine.ctranslate2
        elif engine == InferenceEngine.debugengine:
            return EmbedderEngine.debugengine
        elif engine == InferenceEngine.optimum:
            return EmbedderEngine.optimum
        elif engine == InferenceEngine.neuron:
            return EmbedderEngine.neuron
        else:
            raise NotImplementedError(f"EmbedderEngine for {engine} not implemented")


class RerankEngine(Enum):
    torch = CrossEncoderTorch
    optimum = OptimumCrossEncoder

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return RerankEngine.torch
        elif engine == InferenceEngine.optimum:
            return RerankEngine.optimum
        else:
            raise NotImplementedError(f"RerankEngine for {engine} not implemented")


class ClipLikeEngine(Enum):
    torch = ClipLikeModel

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return ClipLikeEngine.torch
        else:
            raise NotImplementedError(f"ClipLikeEngine for {engine} not implemented")


class PredictEngine(Enum):
    torch = SentenceClassifier

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return PredictEngine.torch
        else:
            raise NotImplementedError(f"PredictEngine for {engine} not implemented")


def length_tokenizer(
    _sentences: list[str],
) -> list[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: list[str], tokenize: Callable = length_tokenizer
) -> tuple[list[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)
