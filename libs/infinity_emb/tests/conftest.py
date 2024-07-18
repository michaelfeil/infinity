import csv
import gzip
import os

import pytest
from sentence_transformers import InputExample, util  # type: ignore

pytest.DEFAULT_BERT_MODEL = "michaelfeil/bge-small-en-v1.5"
pytest.DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
pytest.DEFAULT_CLASSIFIER_MODEL = "SamLowe/roberta-base-go_emotions"

pytest.ENGINE_METHODS = ["embed", "image_embed", "classify", "rerank"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def get_sts_bechmark_dataset() -> (
    tuple[list[InputExample], list[InputExample], list[InputExample]]
):
    sts_dataset_path = os.path.join(
        os.path.dirname(__file__), "data", "datasets", "stsbenchmark.tsv.gz"
    )

    if not os.path.exists(sts_dataset_path):
        util.http_get(
            "https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path
        )

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(
                texts=[row["sentence1"], row["sentence2"]], label=score
            )

            if row["split"] == "dev":
                dev_samples.append(inp_example)
            elif row["split"] == "test":
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples
