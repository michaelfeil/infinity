from types import SimpleNamespace

import torch
from transformers import pipeline  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.classifier.torch import (
    SentenceClassifier,
    _set_pad_token_id_if_missing,
)


def test_classifier(model_name: str = "SamLowe/roberta-base-go_emotions"):
    model = SentenceClassifier(
        engine_args=EngineArgs(
            model_name_or_path=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )  # type: ignore
    )
    pipe = pipeline(model=model_name, task="text-classification")

    sentences = ["This is awesome.", "I am depressed."]

    encode_pre = model.encode_pre(sentences)
    encode_core = model.encode_core(encode_pre)
    preds = model.encode_post(encode_core)

    assert len(preds) == len(sentences)
    assert isinstance(preds, list)
    assert isinstance(preds[0], list)
    assert isinstance(preds[0][0], dict)
    assert isinstance(preds[0][0]["label"], str)
    assert isinstance(preds[0][0]["score"], float)
    assert preds[0][0]["label"] == "admiration"
    assert 0.98 > preds[0][0]["score"] > 0.93

    preds_orig = pipe(sentences, top_k=None, truncation=True)

    assert len(preds_orig) == len(preds)

    for pred_orig, pred in zip(preds_orig, preds):
        assert len(pred_orig) == len(pred)
        for pred_orig_i, pred_i in zip(pred_orig[:5], pred[:5]):
            assert abs(pred_orig_i["score"] - pred_i["score"]) < 0.05

            if pred_orig_i["score"] > 0.005:
                assert pred_orig_i["label"] == pred_i["label"]


def test_set_pad_token_id_if_missing_adopts_tokenizer_value():
    """decoder-only seq-cls checkpoints often ship without config.pad_token_id.

    transformers then raises `Cannot handle batch sizes > 1 if no padding token is
    defined.`, which infinity hits during warmup (batch_size 32) before serving a
    single request. michaelfeil/Qwen3-Reranker-0.6B-seq is such a checkpoint.
    """
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))
    tokenizer = SimpleNamespace(pad_token_id=151643)

    _set_pad_token_id_if_missing(model, tokenizer)

    assert model.config.pad_token_id == 151643


def test_set_pad_token_id_if_missing_is_noop_when_already_set():
    """strict no-op for every model that works today (BERT rerankers, mxbai-*-seq)."""
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=0))
    tokenizer = SimpleNamespace(pad_token_id=151643)

    _set_pad_token_id_if_missing(model, tokenizer)

    assert model.config.pad_token_id == 0


def test_set_pad_token_id_if_missing_tolerates_tokenizer_without_pad():
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=None))
    tokenizer = SimpleNamespace(pad_token_id=None)

    _set_pad_token_id_if_missing(model, tokenizer)

    assert model.config.pad_token_id is None
