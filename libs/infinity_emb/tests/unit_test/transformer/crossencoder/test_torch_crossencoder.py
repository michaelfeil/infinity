import sys

import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.crossencoder.torch import CrossEncoderPatched
from infinity_emb.primitives import Device

import torch

device = Device.cpu if torch.backends.mps.is_available() else Device.auto

SHOULD_TORCH_COMPILE = (
    sys.platform == "linux" and sys.version_info < (3, 12) and torch.cuda.is_available()
)


def test_crossencoder():
    model = CrossEncoderPatched(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            compile=SHOULD_TORCH_COMPILE,
            device=device,
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


def test_crossencoder_rerank_limits():
    from infinity_emb.primitives import RerankLimits

    model = CrossEncoderPatched(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            compile=SHOULD_TORCH_COMPILE,
            device=device,
        )
    )

    query = "Where is Paris? " * 100
    long_doc = "Paris is the capital of France. " * 200

    model_max = (
        getattr(model.model.config, "max_position_embeddings", None)
        or model.tokenizer.model_max_length
    )

    no_limits = RerankLimits(None, None, None)
    uncapped = model.encode_pre([(query, long_doc, no_limits)])
    doc_capped = model.encode_pre([(query, long_doc, RerankLimits(None, 32, None))])
    pair_capped = model.encode_pre([(query, long_doc, RerankLimits(None, None, 48))])
    all_capped = model.encode_pre([(query, long_doc, RerankLimits(16, 32, 40))])
    over_model_max = model.encode_pre([(query, long_doc, RerankLimits(None, None, model_max * 10))])
    legacy_pair = model.encode_pre([(query, long_doc)])  # 2-tuple -> RerankLimits() (no caps)

    # capping the document shortens the scored sequence vs no limits at all.
    assert doc_capped["input_ids"].shape[1] < uncapped["input_ids"].shape[1]
    # the pair cap is a hard ceiling on the joined sequence (plus no extra).
    assert pair_capped["input_ids"].shape[1] <= 48
    assert all_capped["input_ids"].shape[1] <= 40
    # no limit and an over-large pair cap are both clamped to the model's positional limit,
    # so neither ever exceeds what the model can process in encode_core.
    assert uncapped["input_ids"].shape[1] <= model_max
    assert over_model_max["input_ids"].shape[1] <= model_max
    # a 2-tuple still works (RerankLimits() now applies no truncation).
    assert legacy_pair["input_ids"].shape[1] > 0


def test_patched_crossencoder_vs_sentence_transformers():
    model = CrossEncoderPatched(
        engine_args=EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            compile=SHOULD_TORCH_COMPILE,
            device=device,
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
    rankings_sigmoid = 1 / (1 + np.exp(-rankings))

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(rankings_sigmoid, rankings_unpatched, rtol=1e-2, atol=1e-2)
