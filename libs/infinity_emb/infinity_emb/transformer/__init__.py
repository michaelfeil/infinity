__all__ = ["InferenceEngine"]
from infinity_emb.transformer.utils import InferenceEngine

# place the enabling of hf hub transfer here
try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
