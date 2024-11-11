import csv
import gzip
import os
import socket

import pytest
import requests
from sentence_transformers import InputExample, util  # type: ignore

pytest.DEFAULT_BERT_MODEL = "michaelfeil/bge-small-en-v1.5"
pytest.DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
pytest.DEFAULT_CLASSIFIER_MODEL = "SamLowe/roberta-base-go_emotions"
pytest.DEFAULT_AUDIO_MODEL = "laion/clap-htsat-unfused"
pytest.DEFAULT_IMAGE_MODEL = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
pytest.DEFAULT_IMAGE_COLPALI_MODEL = "michaelfeil/colpali-v12-random-testing"
pytest.DEFAULT_COLBERT_MODEL = "michaelfeil/colbert-tiny-random"

pytest.IMAGE_SAMPLE_URL = "https://github.com/michaelfeil/infinity/raw/06fd1f4d8f0a869f4482fc1c78b62a75ccbb66a1/docs/assets/cats_coco_sample.jpg"
pytest.AUDIO_SAMPLE_URL = "https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/infinity_emb/tests/data/audio/beep.wav"

pytest.ENGINE_METHODS = ["embed", "image_embed", "classify", "rerank", "audio_embed"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _download(url: str, **kwargs) -> requests.Response:
    for i in range(5):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code == 200:
                return response
        except Exception:
            pass
    else:
        raise Exception(f"Failed to download {url}")


@pytest.fixture(scope="function")
def audio_sample() -> tuple[requests.Response, str]:
    return (_download(pytest.AUDIO_SAMPLE_URL)), pytest.AUDIO_SAMPLE_URL  # type: ignore


@pytest.fixture(scope="function")
def image_sample() -> tuple[requests.Response, str]:
    return (_download(pytest.IMAGE_SAMPLE_URL, stream=True)), pytest.IMAGE_SAMPLE_URL  # type: ignore


def internet_available():
    try:
        # Attempt to connect to a well-known public DNS server (Google's)
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        return True
    except OSError:
        return False


# pytest hook to dynamically add skip marker for tests that require internet
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Apply skipif marker to tests based on internet availability
    # adds pytest.mark.requires_internet to tests that require internet
    if "requires_internet" in item.keywords and internet_available():
        pytest.skip("Test skipped due to no internet connection")


@pytest.fixture(scope="session")
def get_sts_bechmark_dataset() -> tuple[list[InputExample], list[InputExample], list[InputExample]]:
    sts_dataset_path = os.path.join(
        os.path.dirname(__file__), "data", "datasets", "stsbenchmark.tsv.gz"
    )

    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "dev":
                dev_samples.append(inp_example)
            elif row["split"] == "test":
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)
    return train_samples, dev_samples, test_samples
