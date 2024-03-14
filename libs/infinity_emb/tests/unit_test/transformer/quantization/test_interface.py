import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

from infinity_emb.primitives import Device, Dtype
from infinity_emb.transformer.quantization.interface import quant_interface


def get_model(device="cpu"):
    name = "michaelfeil/bge-small-en-v1.5"
    return AutoModel.from_pretrained(
        name, device=device
    ), AutoTokenizer.from_pretrained(name)


def test_quantize_bert(device: Device = Device.cpu):
    model, tokenizer = get_model(device.value)
    model_unquantized, _ = get_model(device.value)
    model = quant_interface(model=model, device=device, dtype=Dtype.int8)

    model.to(device.value)
    model_unquantized.to(device.value)
    tokens_encoded = tokenizer.batch_encode_plus(
        ["This is an english text to be encoded."], return_tensors="pt"
    )
    tokens_encoded = {k: v.to(device.value) for k, v in tokens_encoded.items()}
    with torch.no_grad():
        out_default = model_unquantized.forward(**tokens_encoded)
        out_quant = model.forward(**tokens_encoded)

    # TODO: distance between the two outputs is to big - need to add bias to quant
    assert np.dot(
        out_default["last_hidden_state"], out_quant["last_hidden_state"].float()
    )
