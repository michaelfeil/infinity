from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

if TYPE_CHECKING:
    from infinity_emb.primitives import EmbeddingReturnType


def list_embeddings_to_response(
    embeddings: Iterable[EmbeddingReturnType],
    model: str,
    usage: int,
) -> dict[str, Union[str, list[dict], dict]]:
    return dict(
        model=model,
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


def to_rerank_response(
    scores: list[float],
    model=str,
    usage=int,
    documents: Optional[list[str]] = None,
) -> dict[str, Any]:
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
