import os
from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_OPTIMUM

if CHECK_OPTIMUM.is_available:
    from optimum.bettertransformer import (  # type: ignore[import-untyped]
        BetterTransformer,
    )

if TYPE_CHECKING:
    from logging import Logger

    from transformers import PreTrainedModel  # type: ignore[import-untyped]


def to_bettertransformer(model: "PreTrainedModel", logger: "Logger"):
    if os.environ.get("INFINITY_DISABLE_OPTIMUM", False):  # OLD VAR
        logger.warning(
            "DEPRECATED `INFINITY_DISABLE_OPTIMUM` - setting optimizations via BetterTransformer,"
            "INFINITY_DISABLE_OPTIMUM is no longer supported, please use the CLI / ENV for that."
        )
        return model
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
