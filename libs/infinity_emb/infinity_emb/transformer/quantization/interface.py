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
    device_orig = model.device
    if device == Device.cpu and dtype in [Dtype.int8, Dtype.auto]:
        logger.info("using torch.quantization.quantize_dynamic()")
        # TODO: verify if cpu requires quantization with torch.quantization.quantize_dynamic()
        model = torch.quantization.quantize_dynamic(
            model.to("cpu"),  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    elif device == Device.cuda and dtype in [Dtype.int8, Dtype.auto]:
        logger.info(f"using quantize() for {dtype.value}")
        quant_handler, state_dict = quantize(model, mode=dtype.value)
        model = quant_handler.convert_for_runtime()
        model.load_state_dict(state_dict)
        model.to(device_orig)
    elif device == Device.cuda and dtype == Dtype.fp8:
        try:
            from float8_experimental.float8_dynamic_linear import (  # type: ignore
                Float8DynamicLinear,
            )
            from float8_experimental.float8_linear_utils import (  # type: ignore
                swap_linear_with_float8_linear,
            )
        except ImportError:
            raise ImportError(
                "float8_experimental is not installed."
                "https://github.com/pytorch-labs/float8_experimental "
                "with commit `88e9e507c56e59c5f17edf513ecbf621b46fc67d`"
            )
        logger.info("using dtype=fp8")
        swap_linear_with_float8_linear(model, Float8DynamicLinear)
    else:
        raise ValueError(
            f"Quantization is not supported on {device} with dtype {dtype}."
        )
    return model
