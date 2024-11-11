from infinity_emb.fastapi_schemas.pymodels import OpenAIEmbeddingResult


def test_embedding_response():
    res = OpenAIEmbeddingResult(data=[], model="hi", usage={"prompt_tokens": 5, "total_tokens": 10})
    assert res.model_dump()["object"] == "list"
