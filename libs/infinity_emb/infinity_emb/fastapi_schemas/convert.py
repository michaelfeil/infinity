from ..inference.primitives import NpEmbeddingType
from .pymodels import OpenAIEmbeddingResult, _EmbeddingObject, _Usage


def list_embeddings_to_response(
    embeddings: NpEmbeddingType, model: str, usage: int
) -> OpenAIEmbeddingResult:
    return OpenAIEmbeddingResult(
        model=model,
        data=[
            _EmbeddingObject(
                object="embedding",
                embedding=emb,
                index=count,
            )
            for count, emb in enumerate(embeddings)
        ],
        usage=_Usage(prompt_tokens=usage, total_tokens=usage),
    )

    # return {
    #     "model": model,
    #     "data": [
    #         dict(
    #             object="embedding",
    #             embedding=emb,
    #             index=count,
    #         )
    #         for count, emb in enumerate(embeddings)
    #     ],
    #     "usage": {"prompt_tokens": usage, "total_tokens": usage},
    # }
