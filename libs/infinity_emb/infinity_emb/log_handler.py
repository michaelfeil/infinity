import logging
from enum import Enum
from typing import Dict

logging.getLogger().handlers.clear()

handlers = []
try:
    from rich.console import Console
    from rich.logging import RichHandler

    handlers.append(RichHandler(console=Console(stderr=True), show_time=False))
except ImportError:
    pass

LOG_LEVELS: Dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": 5,
}

FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    handlers=handlers,
)

logger = logging.getLogger("infinity_emb")


class UVICORN_LOG_LEVELS(Enum):
    critical = "critical"
    error = "error"
    warning = "warning"
    info = "info"
    debug = "debug"
    trace = "trace"

    def to_int(self) -> int:
        return LOG_LEVELS[self.name]
