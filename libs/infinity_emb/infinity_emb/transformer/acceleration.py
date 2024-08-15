# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import os
from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_OPTIMUM, CHECK_TORCH
from infinity_emb.primitives import Device

if CHECK_OPTIMUM.is_available:
    from optimum.bettertransformer import (  # type: ignore[import-untyped]
        BetterTransformer,
    )

if CHECK_TORCH.is_available:
    import torch

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends, "cudnn"):
        # allow TF32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

if TYPE_CHECKING:
    from logging import Logger

    from transformers import PreTrainedModel  # type: ignore[import-untyped]

    from infinity_emb.args import EngineArgs


def to_bettertransformer(
    model: "PreTrainedModel", engine_args: "EngineArgs", logger: "Logger"
):
    if not engine_args.bettertransformer:
        return model

    if engine_args.device == Device.mps or (
        hasattr(model, "device") and model.device.type == "mps"
    ):
        logger.warning(
            "BetterTransformer is not available for MPS device. Continue without bettertransformer modeling code."
        )
        return model

    if os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
        # TODO: remove this code path, it just prints this warning
        logger.error(
            "DEPRECATED the `INFINITY_DISABLE_OPTIMUM` - setting optimizations via BetterTransformer,"
            "INFINITY_DISABLE_OPTIMUM is no longer supported, please use the CLI / ENV for that."
        )

    if (
        hasattr(model.config, "_attn_implementation")
        and model.config._attn_implementation != "eager"
    ):
        raise ValueError("BetterTransformer overwrite requires eager attention.")
    CHECK_OPTIMUM.mark_required()
    logger.info("Adding optimizations via Huggingface optimum. ")
    try:
        model = BetterTransformer.transform(model)
    except Exception as ex:
        # if level is debug then show the exception
        if logger.level <= 10:
            logger.exception(
                f"BetterTransformer is not available for model: {model.__class__} {ex}."
                " Continue without bettertransformer modeling code."
            )
        else:
            logger.warning(
                f"BetterTransformer is not available for model: {model.__class__}"
                " Continue without bettertransformer modeling code."
            )

    return model
