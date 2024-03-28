__all__ = ["InferenceEngine"]
from infinity_emb._optional_imports import CHECK_HF_TRANSFER
from infinity_emb.transformer.utils import InferenceEngine

# place the enabling of hf hub transfer here
if CHECK_HF_TRANSFER.is_available:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore[import-untyped]

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
