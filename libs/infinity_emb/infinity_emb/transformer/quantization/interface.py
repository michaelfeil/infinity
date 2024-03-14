from typing import Any, TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_TORCH
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, Dtype
from infinity_emb.transformer.quantization.quant import quantize

if CHECK_TORCH.is_available:
    import torch



def quant_interface(model: Any, dtype: Dtype = Dtype.int8, device: Device = Device.cpu):
    if device == Device.cpu and dtype in [Dtype.int8, Dtype.auto]:
        logger.info("using torch.quantization.quantize_dynamic()")
        model = torch.quantization.quantize_dynamic(
            model.to("cpu"),  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    elif device == Device.cuda and dtype in [Dtype.int8, Dtype.auto]:
        logger.warning(
            "Quantization is only supported on device=cpu,"
            f" but you are using device={device} with dtype={dtype}."
        )
        quant_handler, _ = quantize(model, mode=dtype.value)
        model = quant_handler.convert_for_runtime()
        model.to(device.value)
        # features1 = self.tokenize(["hello world"])
        # features1 = util.batch_to_device(features1, self.device)
        # model.forward(**features1)
    else:
        raise ValueError(
            f"Quantization is not supported on {device} with dtype {dtype}."
        )
    return model