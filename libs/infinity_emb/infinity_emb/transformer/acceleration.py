import os

try:
    from optimum.bettertransformer import BetterTransformer  # type: ignore

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


def to_bettertransformer(model, logger):
    if OPTIMUM_AVAILABLE and not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
        logger.info(
            "Adding optimizations via Huggingface optimum. "
            "Disable by setting the env var `INFINITY_DISABLE_OPTIMUM`"
        )
        try:
            model = BetterTransformer.transform(model)
        except Exception as ex:
            logger.exception(f"BetterTransformer failed with {ex}")
            exit(1)
    elif not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
        logger.info(
            "No optimizations via Huggingface optimum,"
            " it is disabled via env INFINITY_DISABLE_OPTIMUM "
        )
    else:
        logger.info(
            "No optimizations via Huggingface optimum, "
            "install `pip install infinity-emb[optimum]`"
        )
    return model
