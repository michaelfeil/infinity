import torch
from transformers import pipeline  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.classifier.torch import SentenceClassifier


def test_classifier(model_name: str = "SamLowe/roberta-base-go_emotions"):
    model = SentenceClassifier(
        engine_args=EngineArgs(model_name_or_path=model_name, device="cuda" if torch.cuda.is_available() else "cpu")  # type: ignore
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
