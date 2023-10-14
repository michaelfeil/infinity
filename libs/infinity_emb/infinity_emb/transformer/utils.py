from enum import Enum
from typing import Callable, Dict, List, Tuple

from infinity_emb.transformer.dummytransformer import DummyTransformer
from infinity_emb.transformer.sentence_transformer import (
    CT2SentenceTransformer,
    SentenceTransformerPatched,
)

# from infinity_emb.transformer.fastembed import FastEmbed
__all__ = [
    "InferenceEngine",
    "InferenceEngineTypeHint",
    "length_tokenizer",
    "get_lengths_with_tokenize",
]


class InferenceEngine(Enum):
    torch = SentenceTransformerPatched
    ctranslate2 = CT2SentenceTransformer
    debugengine = DummyTransformer


types: Dict[str, str] = {e.name: e.name for e in InferenceEngine}
InferenceEngineTypeHint = Enum("InferenceEngineTypeHint", types)  # type: ignore


def length_tokenizer(
    _sentences: List[str],
) -> List[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: List[str], tokenize: Callable = length_tokenizer
) -> Tuple[List[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)
