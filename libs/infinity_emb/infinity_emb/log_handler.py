# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import logging
import os
import sys
from enum import Enum
from typing import Any

from infinity_emb.env import MANAGER

logging.getLogger().handlers.clear()

def configure_log_handlers() -> list[Any]:
    """Configure and return logging handlers based on environment settings."""
    handlers = []

    # Determine the appropriate handler
    if not MANAGER.disable_rich_handler:
        try:
            from rich.console import Console
            from rich.logging import RichHandler
            handlers.append(RichHandler(
                console=Console(stderr=True),
                show_time=False,
                markup=True
            ))
            return handlers
        except ImportError:
            pass  # Fall through to default handler

    # Default handler (used when Rich is disabled or not available)
    handlers.append(logging.StreamHandler(sys.stderr))
    return handlers


handlers = configure_log_handlers()

LOG_LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": 5,
}

FORMAT = MANAGER.log_format
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    handlers=handlers,
)

logger = logging.getLogger("infinity_emb")


class UVICORN_LOG_LEVELS(Enum):
    """Re-exports the uvicorn log levels for type hinting and usage."""

    critical = "critical"
    error = "error"
    warning = "warning"
    info = "info"
    debug = "debug"
    trace = "trace"

    def to_int(self) -> int:
        return LOG_LEVELS[self.name]
