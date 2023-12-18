import os
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from infinity_emb.transformer.classifier.torch import SentenceClassifier
from infinity_emb.transformer.crossencoder.torch import (
    CrossEncoderPatched as CrossEncoderTorch,
)
from infinity_emb.transformer.embedder.dummytransformer import DummyTransformer
from infinity_emb.transformer.embedder.fastembed import Fastembed
from infinity_emb.transformer.embedder.sentence_transformer import (
    CT2SentenceTransformer,
    SentenceTransformerPatched,
)

# from infinity_emb.transformer.fastembed import FastEmbed
__all__ = [
    "InferenceEngine",
    "InferenceEngineTypeHint",
    "length_tokenizer",
    "get_lengths_with_tokenize",
    "infinity_cache_dir",
]


class InferenceEngine(Enum):
    torch = "torch"
    ctranslate2 = "ctranslate2"
    fastembed = "fastembed"
    debugengine = "dummytransformer"


class EmbedderEngine(Enum):
    torch = SentenceTransformerPatched
    ctranslate2 = CT2SentenceTransformer
    fastembed = Fastembed
    debugengine = DummyTransformer

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return EmbedderEngine.torch
        elif engine == InferenceEngine.ctranslate2:
            return EmbedderEngine.ctranslate2
        elif engine == InferenceEngine.fastembed:
            return EmbedderEngine.fastembed
        elif engine == InferenceEngine.debugengine:
            return EmbedderEngine.debugengine
        else:
            raise NotImplementedError(f"EmbedderEngine for {engine} not implemented")


class RerankEngine(Enum):
    torch = CrossEncoderTorch

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return RerankEngine.torch
        else:
            raise NotImplementedError(f"RerankEngine for {engine} not implemented")


class PredictEngine(Enum):
    torch = SentenceClassifier

    @staticmethod
    def from_inference_engine(engine: InferenceEngine):
        if engine == InferenceEngine.torch:
            return PredictEngine.torch
        else:
            raise NotImplementedError(f"PredictEngine for {engine} not implemented")


_types: Dict[str, str] = {e.name: e.name for e in InferenceEngine}
InferenceEngineTypeHint = Enum("InferenceEngineTypeHint", _types)  # type: ignore


def length_tokenizer(
    _sentences: List[str],
) -> List[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: List[str], tokenize: Callable = length_tokenizer
) -> Tuple[List[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)


def infinity_cache_dir(overwrite=False):
    """gets the cache dir. If

    Args:
        overwrite (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    cache_dir = None
    inf_home = os.environ.get("INFINITY_HOME")
    st_home = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    hf_home = os.environ.get("HF_HOME")
    if inf_home:
        cache_dir = inf_home
    elif st_home:
        cache_dir = st_home
    elif hf_home:
        cache_dir = hf_home
    else:
        cache_dir = str(Path(".").resolve() / ".infinity_cache")

    if overwrite:
        os.environ.setdefault("INFINITY_HOME", cache_dir)
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
        os.environ.setdefault("HF_HOME", cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    return cache_dir
