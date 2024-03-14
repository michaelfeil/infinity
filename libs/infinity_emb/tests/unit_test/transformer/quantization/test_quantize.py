import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

from infinity_emb.transformer.quantization.quant import quantize


def get_model():
    name = "michaelfeil/bge-small-en-v1.5"
    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name)


def test_quantize_bert(mode="int8"):
    model, tokenizer = get_model()
    model_unquantized, _ = get_model()
    quant_handler, _ = quantize(model=model, mode=mode, device="cuda")
    model = quant_handler.convert_for_runtime()
    model.to("cuda")
    model_unquantized.to("cuda")
    tokens_encoded = tokenizer.batch_encode_plus(
        ["This is an english text to be encoded."], return_tensors="pt"
    )
    tokens_encoded = {k: v.to("cuda") for k, v in tokens_encoded.items()}
    with torch.no_grad():
        out_default = model_unquantized.forward(**tokens_encoded)
        out_quant = model.forward(**tokens_encoded)

    # TODO: distance between the two outputs is to big - need to add bias to quant
    out_default["last_hidden_state"] - out_quant["last_hidden_state"].float()
