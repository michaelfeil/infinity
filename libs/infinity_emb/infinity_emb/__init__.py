# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import importlib.metadata
import os

### Check if HF_HUB_ENABLE_HF_TRANSFER is set, if not try to enable it
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa

        # Needs to be at the top of the file / before other
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        import huggingface_hub.constants  # type: ignore

        huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
    except ImportError:
        pass
import huggingface_hub.constants  # type: ignore

huggingface_hub.constants.HF_HUB_DISABLE_PROGRESS_BARS = True


from infinity_emb.args import EngineArgs  # noqa: E402
from infinity_emb.engine import AsyncEmbeddingEngine, AsyncEngineArray  # noqa: E402
from infinity_emb.env import MANAGER  # noqa: E402

# reexports
from infinity_emb.infinity_server import create_server  # noqa: E402
from infinity_emb.log_handler import logger  # noqa: E402
from infinity_emb.sync_engine import SyncEngineArray  # noqa: E402

__version__: str = importlib.metadata.version("infinity_emb")

__all__ = [
    "__version__",
    "AsyncEmbeddingEngine",
    "AsyncEngineArray",
    "create_server",
    "EngineArgs",
    "logger",
    "MANAGER",
    "SyncEngineArray",
]
