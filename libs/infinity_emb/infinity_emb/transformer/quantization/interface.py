from typing import Any

from infinity_emb._optional_imports import CHECK_TORCH
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, Dtype
from infinity_emb.transformer.quantization.quant import quantize

if CHECK_TORCH.is_available:
    import torch


def quant_interface(model: Any, dtype: Dtype = Dtype.int8, device: Device = Device.cpu):
    """Quantize a model to a specific dtype and device.

    Args:
        model (Any): The model (torch state dict) to quantize.
        dtype (Dtype, optional): The dtype to quantize to. Defaults to Dtype.int8.
        device (Device, optional): The device of the model. Do not use Device.auto, needs to be a resolved device.
            Defaults to Device.cpu.
    """
    if device == Device.cpu and dtype in [Dtype.int8, Dtype.auto]:
        logger.info("using torch.quantization.quantize_dynamic()")
        # TODO: verify if cpu requires quantization with torch.quantization.quantize_dynamic()
        model = torch.quantization.quantize_dynamic(
            model.to("cpu"),  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    elif device == Device.cuda and dtype in [Dtype.int8, Dtype.auto]:
        quant_handler, state_dict = quantize(model, mode=dtype.value)
        model = quant_handler.convert_for_runtime()
        model.load_state_dict(state_dict)
        model.to(device.value)
        # features1 = self.tokenize(["hello world"])
        # features1 = util.batch_to_device(features1, self.device)
        # model.forward(**features1)
    else:
        raise ValueError(
            f"Quantization is not supported on {device} with dtype {dtype}."
        )
    return model
