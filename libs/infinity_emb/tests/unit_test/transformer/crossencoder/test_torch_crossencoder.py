from infinity_emb.transformer.crossencoder.torch import CrossEncoderPatched


def test_crossencoder():
    model = CrossEncoderPatched("BAAI/bge-reranker-base")

    query = "Where is Paris?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish",
    ]

    query_docs = [(query, doc) for doc in documents]

    encode_pre = model.encode_pre(query_docs)
    encode_core = model.encode_core(encode_pre)
    rankings = model.encode_post(encode_core)

    assert len(rankings) == 3
    assert rankings[0] > rankings[1] > rankings[2]
