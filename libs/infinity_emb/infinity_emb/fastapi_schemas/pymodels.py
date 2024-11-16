# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil
# IMPORT of this file requires pydantic 2.x

from __future__ import annotations

import base64
import time
from typing import TYPE_CHECKING, Annotated, Any, Iterable, Literal, Optional, Union
from uuid import uuid4

import numpy as np


from infinity_emb._optional_imports import CHECK_PYDANTIC
from infinity_emb.primitives import EmbeddingEncodingFormat, Modality

CHECK_PYDANTIC.mark_required()
# pydantic 2.x is strictly needed starting v0.0.70
from pydantic import (  # noqa
    BaseModel,
    Discriminator,
    Field,
    RootModel,
    Tag,
    conlist,
)

from .data_uri import DataURI  # noqa
from .pydantic_v2 import (  # noqa
    INPUT_STRING,
    ITEMS_LIMIT,
    ITEMS_LIMIT_SMALL,
    HttpUrl,
)

if TYPE_CHECKING:
    from infinity_emb.args import EngineArgs
    from infinity_emb.primitives import (
        ClassifyReturnType,
        EmbeddingReturnType,
        RerankReturnType,
    )

DataURIorURL = Union[Annotated[DataURI, str], HttpUrl]


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class _OpenAIEmbeddingInput(BaseModel):
    model: str = "default/not-specified"
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float
    user: Optional[str] = None


class _OpenAIEmbeddingInput_Text(_OpenAIEmbeddingInput):
    """helper"""

    input: Union[  # type: ignore
        conlist(  # type: ignore
            Annotated[str, INPUT_STRING],
            **ITEMS_LIMIT,
        ),
        Annotated[str, INPUT_STRING],
    ]
    modality: Literal[Modality.text] = Modality.text  # type: ignore


class _OpenAIEmbeddingInput_URI(_OpenAIEmbeddingInput):
    """helper"""

    input: Union[  # type: ignore
        conlist(  # type: ignore
            DataURIorURL,
            **ITEMS_LIMIT_SMALL,
        ),
        DataURIorURL,
    ]


class OpenAIEmbeddingInput_Audio(_OpenAIEmbeddingInput_URI):
    modality: Literal[Modality.audio] = Modality.audio  # type: ignore


class OpenAIEmbeddingInput_Image(_OpenAIEmbeddingInput_URI):
    modality: Literal[Modality.image] = Modality.image  # type: ignore


def get_modality(obj: dict) -> str:
    """resolve the modality of the extra_body.
    If not present, default to text

    Function name is used to return error message, keep it explicit
    """
    try:
        return obj.get("modality", Modality.text.value)
    except AttributeError:
        # in case a very weird request is sent, validate it against the default
        return Modality.text.value


class MultiModalOpenAIEmbedding(RootModel):
    root: Annotated[
        Union[
            Annotated[_OpenAIEmbeddingInput_Text, Tag(Modality.text.value)],
            Annotated[OpenAIEmbeddingInput_Audio, Tag(Modality.audio.value)],
            Annotated[OpenAIEmbeddingInput_Image, Tag(Modality.image.value)],
        ],
        Discriminator(get_modality),
    ]


class ImageEmbeddingInput(BaseModel):
    """LEGACY, DO NO LONGER UPDATE"""

    input: Union[  # type: ignore
        conlist(  # type: ignore
            DataURIorURL,
            **ITEMS_LIMIT_SMALL,
        ),
        DataURIorURL,
    ]
    model: str = "default/not-specified"
    encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float
    user: Optional[str] = None


class AudioEmbeddingInput(ImageEmbeddingInput):
    """LEGACY, DO NO LONGER UPDATE"""

    pass


class _EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: Union[list[float], bytes, list[list[float]]]
    index: int


class OpenAIEmbeddingResult(BaseModel):
    object: Literal["list"] = "list"
    data: list[_EmbeddingObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_embeddings_response(
        embeddings: Union[Iterable["EmbeddingReturnType"], np.ndarray],
        engine_args: "EngineArgs",
        usage: int,
        encoding_format: EmbeddingEncodingFormat = EmbeddingEncodingFormat.float,
    ) -> dict[str, Union[str, list[dict], dict]]:
        if encoding_format == EmbeddingEncodingFormat.base64:
            if engine_args.embedding_dtype.uses_bitpacking():
                raise ValueError(
                    f"model {engine_args.served_model_name} does not support base64 encoding, as it uses uint8-bitpacking with {engine_args.embedding_dtype}"
                )
            embeddings = [
                base64.b64encode(np.frombuffer(emb.astype(np.float32), dtype=np.float32))  # type: ignore
                for emb in embeddings
            ]  # type: ignore
        else:
            embeddings = [e.tolist() for e in embeddings]
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
    raw_scores: bool = False
    model: str = "default/not-specified"
    top_n: Optional[int] = Field(default=None, gt=0)


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
        scores: list["RerankReturnType"],
        model: str,
        usage: int,
        return_documents: bool,
    ) -> dict:
        if not return_documents:
            return dict(
                model=model,
                results=[
                    dict(relevance_score=entry.relevance_score, index=entry.index)
                    for entry in scores
                ],
                usage=dict(prompt_tokens=usage, total_tokens=usage),
            )
        else:
            return dict(
                model=model,
                results=[
                    dict(
                        relevance_score=entry.relevance_score,
                        index=entry.index,
                        document=entry.document,
                    )
                    for entry in scores
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
