from typing import Optional

import pytest
import torch
from transformers import AutoTokenizer, BertModel  # type: ignore

from infinity_emb.primitives import Device, Dtype
from infinity_emb.transformer.quantization.interface import quant_interface

devices = [Device.cpu]
# TODO: add support for cuda
if torch.cuda.is_available():
    devices.append(Device.cuda)


def get_model(device: Optional[str] = "cpu"):
    name = "michaelfeil/bge-small-en-v1.5"
    model = BertModel.from_pretrained(
        name,
    )
    model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dtype", [Dtype.int8])
def test_quantize_bert(device: Device, dtype: Dtype):
    """Test if the quantized model is close to the unquantized model.

    Args:
        device (Device): device to use for inference.
        dtype (Dtype): data type for quantization
    """
    model, tokenizer = get_model(device.resolve())
    model_unquantized, _ = get_model(device.resolve())
    model = quant_interface(model=model, device=device, dtype=dtype)

    model.to(device.resolve())
    model_unquantized.to(device.resolve())
    tokens_encoded = tokenizer.batch_encode_plus(
        ["This is an english text to be encoded."], return_tensors="pt"
    )
    tokens_encoded = {k: v.to(device.resolve()) for k, v in tokens_encoded.items()}
    with torch.no_grad():
        out_default = model_unquantized.forward(**tokens_encoded)[
            "last_hidden_state"
        ].mean(dim=1)
        out_quant = model.forward(**tokens_encoded)["last_hidden_state"].mean(dim=1)

    assert torch.cosine_similarity(out_default, out_quant) > 0.95
