import sys

import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.crossencoder.torch import CrossEncoderPatched

SHOULD_TORCH_COMPILE = sys.platform == "linux" and sys.version_info < (3, 12)


def test_crossencoder():
    model = CrossEncoderPatched(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            compile=SHOULD_TORCH_COMPILE,
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
    model = CrossEncoderPatched(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1", compile=True
        )
    )
    model_unpatched = CrossEncoder(
        "mixedbread-ai/mxbai-rerank-xsmall-v1", trust_remote_code=True
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
    rankings_sigmoid = 1 / (1 + np.exp(-rankings))

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(
        rankings_sigmoid, rankings_unpatched, rtol=1e-2, atol=1e-2
    )
