# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Annotated, Any, Iterable, Literal, Optional, Union
from uuid import uuid4

import numpy as np

if TYPE_CHECKING:
    from infinity_emb.args import EngineArgs
    from infinity_emb.primitives import ClassifyReturnType, EmbeddingReturnType

from infinity_emb._optional_imports import CHECK_PYDANTIC
from infinity_emb.primitives import EmbeddingEncodingFormat

# potential backwards compatibility to pydantic 1.X
# pydantic 2.x is preferred by not strictly needed
if CHECK_PYDANTIC.is_available:
    from pydantic import BaseModel, Field, conlist

    try:
        from pydantic import AnyUrl, HttpUrl, StringConstraints

        # Note: adding artificial limit, this might reveal splitting
        # issues on the client side
        #      and is not a hard limit on the server side.
        INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
        ITEMS_LIMIT = {
            "min_length": 1,
            "max_length": 2048,
        }
    except ImportError:
        from pydantic import constr

        INPUT_STRING = constr(max_length=8192 * 15, strip_whitespace=True)  # type: ignore
        ITEMS_LIMIT = {
            "min_items": 1,
            "max_items": 2048,
        }
        HttpUrl, AnyUrl = str, str  # type: ignore
else:

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(*args, **kwargs):  # type: ignore
        pass

    def conlist():  # type: ignore
        pass


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingInput(BaseModel):
    input: Union[  # type: ignore
        conlist(  # type: ignore
            Annotated[str, INPUT_STRING],
            **ITEMS_LIMIT,
        ),
        Annotated[str, INPUT_STRING],
    ]
    model: str = "default/not-specified"
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float
    user: Optional[str] = None


class ImageEmbeddingInput(BaseModel):
    input: Union[  # type: ignore
        conlist(  # type: ignore
            Annotated[AnyUrl, HttpUrl],
            **ITEMS_LIMIT,
        ),
        Annotated[AnyUrl, HttpUrl],
    ]
    model: str = "default/not-specified"
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float
    user: Optional[str] = None


class _EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[list[float], bytes]
    index: int


class OpenAIEmbeddingResult(BaseModel):
    object: Literal["embedding"] = "embedding"
    data: list[_EmbeddingObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_embeddings_response(
        embeddings: Iterable["EmbeddingReturnType"],
        engine_args: "EngineArgs",
        usage: int,
        encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float,
    ) -> dict[str, Union[str, list[dict], dict]]:
        if encoding_format == EmbeddingEncodingFormat.base64:
            assert (
                not engine_args.embedding_dtype.uses_bitpacking()
            ), f"model {engine_args.served_model_name} does not support base64 encoding, as it uses uint8-bitpacking with {engine_args.embedding_dtype}"
            embeddings = [base64.b64encode(np.frombuffer(emb.astype(np.float32), dtype=np.float32)) for emb in embeddings]  # type: ignore

        return dict(
            model=engine_args.served_model_name,
            data=[
                dict(
                    object="embedding",
                    embedding=emb,
                    index=count,
                )
                for count, emb in enumerate(embeddings)
            ],
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )


class ClassifyInput(BaseModel):
    input: conlist(  # type: ignore
        Annotated[str, INPUT_STRING],
        **ITEMS_LIMIT,
    )
    model: str = "default/not-specified"
    raw_scores: bool = False


class _ClassifyObject(BaseModel):
    score: float
    label: str


class ClassifyResult(BaseModel):
    """Result of classification."""

    object: Literal["classify"] = "classify"
    data: list[list[_ClassifyObject]]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_classify_response(
        scores_labels: list[ClassifyReturnType],
        model: str,
        usage: int,
    ) -> dict[str, Union[str, list[ClassifyReturnType], dict]]:
        return dict(
            model=model,
            data=scores_labels,
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )


class RerankInput(BaseModel):
    """Input for reranking"""

    query: Annotated[str, INPUT_STRING]
    documents: conlist(  # type: ignore
        Annotated[str, INPUT_STRING],
        **ITEMS_LIMIT,
    )
    return_documents: bool = False
    model: str = "default/not-specified"


class _ReRankObject(BaseModel):
    relevance_score: float
    index: int
    document: Optional[str] = None


class ReRankResult(BaseModel):
    """Following the Cohere protocol for Rerankers."""

    object: Literal["rerank"] = "rerank"
    results: list[_ReRankObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_rerank_response(
        scores: list[float],
        model=str,
        usage=int,
        documents: Optional[list[str]] = None,
    ) -> dict:
        if documents is None:
            return dict(
                model=model,
                results=[
                    dict(relevance_score=score, index=count)
                    for count, score in enumerate(scores)
                ],
                usage=dict(prompt_tokens=usage, total_tokens=usage),
            )
        else:
            return dict(
                model=model,
                results=[
                    dict(relevance_score=score, index=count, document=doc)
                    for count, (score, doc) in enumerate(zip(scores, documents))
                ],
                usage=dict(prompt_tokens=usage, total_tokens=usage),
            )


class ModelInfo(BaseModel):
    id: str
    stats: dict[str, Any]
    object: Literal["model"] = "model"
    owned_by: Literal["infinity"] = "infinity"
    created: int = Field(default_factory=lambda: int(time.time()))
    backend: str = ""
    capabilities: set[str] = set()


class OpenAIModelInfo(BaseModel):
    data: list[ModelInfo]
    object: str = "list"
