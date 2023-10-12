import logging
from enum import Enum

from rich.console import Console
from rich.logging import RichHandler
from uvicorn.config import LOG_LEVELS

logging.getLogger().handlers.clear()

FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    handlers=[RichHandler(console=Console(stderr=True), show_time=False)],
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
