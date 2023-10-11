__all__ = ["logger", "create_server", "inference", "fastapi_schemas", "__version__"]
import importlib.metadata

from . import fastapi_schemas, inference
from .infinity_server import create_server
from .log_handler import logger

__version__ = importlib.metadata.version("infinity_emb")
