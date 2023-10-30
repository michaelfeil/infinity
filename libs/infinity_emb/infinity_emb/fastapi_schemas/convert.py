from typing import Any, Dict, Iterable, Union

from infinity_emb.inference.primitives import NpEmbeddingType


def list_embeddings_to_response(
    embeddings: Union[NpEmbeddingType, Iterable[NpEmbeddingType]],
    model: str,
    usage: int,
) -> Dict[str, Any]:
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
