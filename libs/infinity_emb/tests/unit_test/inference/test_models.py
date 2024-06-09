"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

import copy
import sys
from typing import Union

import pytest
import torch
from sentence_transformers import InputExample  # type: ignore[import-untyped]
from sentence_transformers.evaluation import (  # type: ignore[import-untyped]
    EmbeddingSimilarityEvaluator,
)

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.embedder.ct2 import CT2SentenceTransformer
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)


def _pretrained_model_score(
    dataset: list[InputExample],
    model_name,
    expected_score,
    ct2_compute_type: str = "",
):
    test_samples = dataset[::3]
    model: Union[CT2SentenceTransformer, SentenceTransformerPatched]

    if ct2_compute_type:
        model = CT2SentenceTransformer(
            engine_args=EngineArgs(model_name_or_path=model_name),
            ct2_compute_type=ct2_compute_type,
        )

    else:
        model = SentenceTransformerPatched(
            engine_args=EngineArgs(
                model_name_or_path=model_name,
                bettertransformer=not torch.backends.mps.is_available(),
            )
        )
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="sts-test"
    )

    score = model.evaluate(evaluator)["sts-test_spearman_cosine"] * 100  # type: ignore
    print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.01


@pytest.mark.parametrize(
    "model,score,compute_type",
    [
        ("sentence-transformers/bert-base-nli-mean-tokens", 76.37, "int8"),
        ("sentence-transformers/bert-base-nli-mean-tokens", 76.46, None),
        ("sentence-transformers/all-MiniLM-L6-v2", 81.03, None),
        ("sentence-transformers/all-MiniLM-L6-v2", 81.03, "default"),
        ("sentence-transformers/all-MiniLM-L6-v2", 80.73, "int8"),
        ("sentence-transformers/all-MiniLM-L6-v2", 81.03, "default"),
        ("michaelfeil/bge-small-en-v1.5", 84.90, None),
        ("michaelfeil/bge-small-en-v1.5", 84.90, "int8"),
    ],
)
@pytest.mark.skipif(sys.platform == "darwin", reason="does not run on mac")
def test_bert(get_sts_bechmark_dataset, model, score, compute_type):
    samples = copy.deepcopy(get_sts_bechmark_dataset[2])
    _pretrained_model_score(samples, model, score, ct2_compute_type=compute_type)
