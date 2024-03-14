import os

from infinity_emb._optional_imports import CHECK_OPTIMUM

if CHECK_OPTIMUM.is_available:
    from optimum.bettertransformer import BetterTransformer  # type: ignore


def to_bettertransformer(model, logger, force_usage=False):
    if not force_usage:
        logger.info("No optimizations via Huggingface optimum. ")
        return model
    if os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
        logger.info(
            "No optimizations via Huggingface optimum,"
            " it is disabled via env INFINITY_DISABLE_OPTIMUM "
        )
        return model
    if CHECK_OPTIMUM.mark_required():
        logger.info("Adding optimizations via Huggingface optimum. ")
        try:
            model = BetterTransformer.transform(model)
        except Exception as ex:
            logger.exception(
                f"BetterTransformer is not available for model. {ex}."
                " Continue without bettertransformer modeling code."
            )
    return model
