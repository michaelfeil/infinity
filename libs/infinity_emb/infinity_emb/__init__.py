import importlib.metadata
import os

import huggingface_hub.constants  # type: ignore

### Check if HF_HUB_ENABLE_HF_TRANSFER is set, if not try to enable it
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa

        # Needs to be at the top of the file / before other
        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
    except ImportError:
        pass
huggingface_hub.constants.HF_HUB_DISABLE_PROGRESS_BARS = True


from infinity_emb import fastapi_schemas, inference, transformer  # noqa: E402
from infinity_emb.args import EngineArgs  # noqa: E402
from infinity_emb.engine import AsyncEmbeddingEngine  # noqa: E402

# reexports
from infinity_emb.infinity_server import create_server  # noqa: E402
from infinity_emb.log_handler import logger  # noqa: E402

__version__ = importlib.metadata.version("infinity_emb")

__all__ = [
    "transformer",
    "inference",
    "fastapi_schemas",
    "logger",
    "create_server",
    "AsyncEmbeddingEngine",
    "EngineArgs",
    "__version__",
]
