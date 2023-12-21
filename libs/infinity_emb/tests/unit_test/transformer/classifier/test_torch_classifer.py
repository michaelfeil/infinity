from transformers import pipeline  # type: ignore

from infinity_emb.transformer.classifier.torch import SentenceClassifier


def test_classifier():
    model = SentenceClassifier("SamLowe/roberta-base-go_emotions")
    pipe = pipeline(
        model="SamLowe/roberta-base-go_emotions", task="text-classification"
    )

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
        for pred_orig_i, pred_i in zip(pred_orig, pred):
            assert pred_orig_i["label"] == pred_i["label"]
            assert abs(pred_orig_i["score"] - pred_i["score"]) < 0.1
