"""Stable Python backend protocol for Infinity v3.

A Python backend supplies a factory that returns an object with three methods:
``submit``, ``healthy``, and ``close``. The Infinity Rust backend owns request
scheduling, batching, result packing, and the HTTP API. The plugin owns model
loading and execution.

Factories receive the effective per-model ``max_batch_tokens`` and
``max_sequence_length`` keywords in addition to the model/configuration
arguments. Factories should accept ``**config`` or declare both keywords.
"""

from __future__ import annotations

from typing import Any, Protocol

PROTOCOL_VERSION = 1

ENTRY_POINT_GROUP = "infinity_v3.backends"


class Response(Protocol):
    """Completion handle owned by the Rust side."""

    def set_result(self, result: Any) -> None:
        """Complete the request with a result object."""

    def set_exception(self, error: BaseException) -> None:
        """Complete the request with an exception."""


class ModelExecutor(Protocol):
    """Request executor owned by one Python worker thread."""

    def submit(self, route: str, response: Response, payload: dict[str, Any]) -> None:
        """Accept one request without blocking the Rust backend worker."""

    def healthy(self) -> bool:
        """Return whether the executor can accept work."""

    def close(self) -> None:
        """Stop the executor and release model resources."""


class BackendFactory(Protocol):
    """Factory loaded from an entry point or ``module:attribute``."""

    def __call__(self, **config: Any) -> ModelExecutor:
        """Create an executor for one model."""
