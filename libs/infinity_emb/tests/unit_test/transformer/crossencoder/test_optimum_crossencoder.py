import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.crossencoder.optimum import OptimumCrossEncoder


def test_crossencoder():
    model = OptimumCrossEncoder(
        engine_args=EngineArgs(
            model_name_or_path="Xenova/bge-reranker-base",
            device="cpu",
        )
    )

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


def test_patched_crossencoder_vs_sentence_transformers():
    model = OptimumCrossEncoder(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            device="cpu",
        )
    )
    model_unpatched = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

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
    rankings_sigmoid = 1 / (1 + np.exp(-np.array(rankings)))

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(
        rankings_sigmoid, rankings_unpatched, rtol=0.04, atol=0.04
    )
