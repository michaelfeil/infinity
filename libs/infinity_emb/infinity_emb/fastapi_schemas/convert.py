from infinity_emb.fastapi_schemas.pymodels import OpenAIEmbeddingResult
from infinity_emb.inference.primitives import NpEmbeddingType


def list_embeddings_to_response(
    embeddings: NpEmbeddingType, model: str, usage: int
) -> OpenAIEmbeddingResult:
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
