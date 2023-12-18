"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""
import copy
from typing import List

import pytest
from sentence_transformers import InputExample  # type: ignore
from sentence_transformers.evaluation import (  # type: ignore
    EmbeddingSimilarityEvaluator,  # type: ignore
)

from infinity_emb.transformer.embedder.sentence_transformer import (
    CT2SentenceTransformer,
    SentenceTransformerPatched,
)


def _pretrained_model_score(
    dataset: List[InputExample],
    model_name,
    expected_score,
    ct2_compute_type: str = "",
):
    test_samples = dataset[::3]

    if ct2_compute_type:
        model = CT2SentenceTransformer(model_name, compute_type=ct2_compute_type)

    else:
        model = SentenceTransformerPatched(model_name)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name="sts-test"
    )

    score = model.evaluate(evaluator) * 100
    print(model_name, "{:.2f} vs. exp: {:.2f}".format(score, expected_score))
    assert score > expected_score or abs(score - expected_score) < 0.01


@pytest.mark.parametrize(
    "model,score,compute_type",
    [
        ("sentence-transformers/bert-base-nli-mean-tokens", 76.76, "int8"),
        ("sentence-transformers/bert-base-nli-mean-tokens", 76.86, None),
        ("sentence-transformers/all-MiniLM-L6-v2", 82.03, None),
        ("sentence-transformers/all-MiniLM-L6-v2", 82.03, "default"),
        ("sentence-transformers/all-MiniLM-L6-v2", 81.73, "int8"),
        ("sentence-transformers/all-MiniLM-L6-v2", 82.03, "default"),
        ("BAAI/bge-small-en-v1.5", 86.03, None),
        ("BAAI/bge-small-en-v1.5", 86.03, "int8"),
    ],
)
def test_bert(get_sts_bechmark_dataset, model, score, compute_type):
    samples = copy.deepcopy(get_sts_bechmark_dataset[2])
    _pretrained_model_score(samples, model, score, ct2_compute_type=compute_type)
