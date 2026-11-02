# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from functools import cache, wraps
from hashlib import md5
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import requests  # type: ignore

from infinity_emb._optional_imports import CHECK_SENTENCE_TRANSFORMERS, CHECK_TORCH
from infinity_emb.env import MANAGER
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, Dtype, EmbeddingDtype
from infinity_emb.transformer.quantization.quant import quantize

if TYPE_CHECKING:
    from infinity_emb.transformer.abstract import BaseEmbedder
    import torch

if CHECK_TORCH.is_available:
    import torch

if CHECK_SENTENCE_TRANSFORMERS.is_available:
    from sentence_transformers.quantization import quantize_embeddings  # type: ignore


def quant_interface(model: Any, dtype: Union[Dtype] = Dtype.int8, device: Device = Device.cpu):
    """Quantize a model to a specific dtype and device.

    Args:
        model (Any): The model (torch state dict) to quantize.
        dtype (Dtype, optional): The dtype to quantize to. Defaults to Dtype.int8.
        device (Device, optional): The device of the model. Do not use Device.auto, needs to be a resolved device.
            Defaults to Device.cpu.
    """
    device_orig = model.device
    if device == Device.cpu and dtype in [Dtype.int8, Dtype.auto, torch.int8]:
        logger.info("using torch.quantization.quantize_dynamic()")
        # TODO: verify if cpu requires quantization with torch.quantization.quantize_dynamic()
        model = torch.quantization.quantize_dynamic(
            model.to("cpu"),  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.qint8,
        )
    elif device == Device.cuda and dtype in [Dtype.int8, Dtype.auto, torch.int8]:
        logger.info(f"using quantize() for {dtype.value}")
        quant_handler, state_dict = quantize(model, mode=dtype.value)
        model = quant_handler.convert_for_runtime()
        model.load_state_dict(state_dict)
        model.to(device_orig)
    elif device == Device.cuda and dtype in [Dtype.fp8, torch.float8_e5m2]:
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
        raise ValueError(f"Quantization is not supported on {device} with dtype {dtype}.")
    return model


@cache
def _get_text_calibration_dataset() -> list[str]:
    url = MANAGER.calibration_dataset_url

    cache_file = (
        MANAGER.cache_dir
        / "calibration_dataset"
        / md5(url.encode()).hexdigest()
        / "calibration_dataset.txt"
    )
    if cache_file.exists():
        text = cache_file.read_text()
    else:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text)

    return [line.strip() for line in text.splitlines()]


@cache
def _create_statistics_embedding(model: "BaseEmbedder", percentile=100) -> np.ndarray:
    """returns `ranges`, the min and max values of the embeddings for quantization."""

    def _encode(model, dataset, batch_size=8):
        """batched encoding of the dataset"""
        for i in range(0, len(dataset), batch_size):
            yield model.encode_post(
                model.encode_core(model.encode_pre(dataset[i : i + batch_size])),
                # _internal_skip_quanitzation is a hack to skip quantization
                # and avoid infinite recursion
                _internal_skip_quanitzation=True,
            )

    if "image_embed" in model.capabilities:
        # TODO: implement calibration for vision models
        logger.error(
            "quantization requires a calibrating dataset, "
            f"which is implemented for Text models only. You are using {model.__class__.__name__}"
        )
    dataset = _get_text_calibration_dataset()

    logger.info(f"Creating calibration dataset for model using {len(dataset)} sentences.")

    calibration_embeddings = np.concatenate(list(_encode(model, dataset)))
    assert percentile > 50 and percentile <= 100, "percentile should be between 50 and 100"
    return np.percentile(calibration_embeddings, [100 - percentile, percentile], axis=0)


def quant_embedding_decorator():
    def decorator(func):
        @wraps(func)
        def wrapper(self: "BaseEmbedder", *args, **kwargs):
            """
            wraps a func called via func(self, *args, **kwargs) -> EmbeddingDtype(similar)

            Special:
                self has embedding_dtype: EmbeddingDtype
                _internal_skip_quanitzation=True skips quantization
            """
            skip_quanitzation = kwargs.pop("_internal_skip_quanitzation", False)
            embeddings = func(self, *args, **kwargs)
            if self.embedding_dtype == EmbeddingDtype.float32 or skip_quanitzation:
                return embeddings
            elif (
                self.embedding_dtype == EmbeddingDtype.int8
                or self.embedding_dtype == EmbeddingDtype.uint8
            ):
                calibration_ranges = _create_statistics_embedding(self)
            else:
                calibration_ranges = None
            return quantize_embeddings(
                embeddings,
                precision=self.embedding_dtype.value,
                ranges=calibration_ranges,
            )

        return wrapper

    return decorator
