from transformers import AutoModel  # type: ignore

from infinity_emb.transformer.quantization.quant import quantize


def test_quantize_bert(mode="int8"):
    model = AutoModel.from_pretrained("bert-base-uncased")

    quantize(
        model=model,
        mode=mode,
    )
