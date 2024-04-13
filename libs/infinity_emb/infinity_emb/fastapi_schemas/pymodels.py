from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Optional, Union
from uuid import uuid4

from infinity_emb._optional_imports import CHECK_PYDANTIC

# potential backwards compatibility to pydantic 1.X
# pydantic 2.x is preferred by not strictly needed
if CHECK_PYDANTIC.is_available:
    from pydantic import BaseModel, Field, conlist

    try:
        from pydantic import StringConstraints

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


else:

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(*args, **kwargs):  # type: ignore
        pass

    def conlist():  # type: ignore
        pass


class OpenAIEmbeddingInput(BaseModel):
    input: Union[  # type: ignore
        conlist(  # type: ignore
            Annotated[str, INPUT_STRING],
            **ITEMS_LIMIT,
        ),
        Annotated[str, INPUT_STRING],
    ]
    model: str = "default/not-specified"
    user: Optional[str] = None


class _EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingResult(BaseModel):
    object: Literal["embedding"] = "embedding"
    data: list[_EmbeddingObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class RerankInput(BaseModel):
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
    object: Literal["rerank"] = "rerank"
    data: list[_ReRankObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))


class ModelInfo(BaseModel):
    id: str
    stats: dict[str, Any]
    object: Literal["model"] = "model"
    owned_by: Literal["infinity"] = "infinity"
    created: int = Field(default_factory=lambda: int(time.time()))
    backend: str = ""


class OpenAIModelInfo(BaseModel):
    data: list[ModelInfo]
    object: str = "list"
