import os
from typing import TYPE_CHECKING

from packaging.version import Version

from infinity_emb._optional_imports import CHECK_OPTIMUM, CHECK_TRANSFORMERS

if CHECK_TRANSFORMERS.is_available:
    from transformers import __version__ as transformers_version  # type: ignore

if CHECK_OPTIMUM.is_available:
    from optimum.bettertransformer import (  # type: ignore[import-untyped]
        BetterTransformer,
    )

if TYPE_CHECKING:
    from logging import Logger

    from transformers import PreTrainedModel  # type: ignore[import-untyped]


def to_bettertransformer(model: "PreTrainedModel", logger: "Logger"):
    if os.environ.get("INFINITY_DISABLE_OPTIMUM", False): # OLD VAR
        logger.warning(
            "No optimizations via BetterTransformer,"
            " it is disabled via env `INFINITY_DISABLE_OPTIMUM` "
            "INFINITY_DISABLE_OPTIMUM is no longer supported, please use the CLI / ENV for that."
        )
        return model
    CHECK_TRANSFORMERS.mark_required()
    if Version(transformers_version) >= Version("4.40.3"):
        logger.info(
            "Disable optimizations via BetterTransformer, as torch.sdpa ships with transformers >= 4.41.0"
        )
        return model
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
