__all__ = ["logger", "create_server", "inference", "fastapi_schemas", "__version__"]
import importlib.metadata

from infinity_emb import fastapi_schemas, inference
from infinity_emb.infinity_server import create_server
from infinity_emb.log_handler import logger

__version__ = importlib.metadata.version("infinity_emb")
