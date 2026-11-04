"""Stable Python backend protocol for Infinity v3.

A Python backend supplies a factory that returns an object with three methods:
``submit``, ``healthy``, and ``close``. The Infinity Rust backend owns request
scheduling, batching, result packing, and the HTTP API. The plugin owns model
loading and execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

PROTOCOL_VERSION = 1

DEFAULT_VISIBLE_DEVICE_COUNT = 1
DEFAULT_SUPPORTS_REPLICATION = False

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
        """Execute on the persistent worker or retain the response for later completion."""

    def healthy(self) -> bool:
        """Return whether the executor can accept work."""

    def close(self) -> None:
        """Stop the executor and release model resources."""


class BackendFactory(Protocol):
    """Factory loaded from an entry point or ``module:attribute``."""

    def __call__(self, **config: Any) -> ModelExecutor:
        """Create an executor for one model."""


@dataclass(frozen=True)
class BackendCapabilities:
    """Optional replica capabilities reported before backend startup.

    Protocol 1 plugins may omit the ``capabilities`` hook. Such plugins keep
    the single-device, non-replicated defaults. A plugin that exports
    ``capabilities(config_json: str)`` can report how many devices are visible
    and whether the frontend may place one replica per device. The hook must be
    cheap and side-effect free: discovery can run before model weights are
    downloaded and may run more than once.
    """

    visible_device_count: int = DEFAULT_VISIBLE_DEVICE_COUNT
    supports_replication: bool = DEFAULT_SUPPORTS_REPLICATION


class CapabilityProvider(Protocol):
    """Optional module-level hook used for replica discovery."""

    def __call__(self, config_json: str) -> BackendCapabilities | Mapping[str, Any]:
        """Return backend device capabilities for one ``--python-config`` value."""
