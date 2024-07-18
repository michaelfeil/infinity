import logging
import sys
from enum import Enum
from typing import Any

handlers: list[Any] = []
try:
    from rich.console import Console
    from rich.logging import RichHandler

    handlers.append(RichHandler(console=Console(stderr=True), show_time=False))
except ImportError:
    handlers.append(logging.StreamHandler(sys.stderr))

LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": 5,
}

FORMAT = """{
        "ts": "%(timestamp_ms)d",
        "type": "app",
        "svc": "embedding-service",
        "lvl": "%(levelname)s",
        "act": "%(pathname)s:%(funcName)s:%(lineno)d",
        "a_id": "%(account_id)s",
        "r_id": "%(request_id)s",
        "p": "freddy-freshservice",
        "tp": "%(trace_parent)s",
        "d": "%(time_elapsed)f",
        "thread_id": "%(thread)s",
        "trace_id": "%(otelTraceID)s",
        "dur": "%(time_elapsed)f",
        "msg": "%(message)s",
    }"""

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
