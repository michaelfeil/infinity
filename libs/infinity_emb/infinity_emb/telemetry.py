import hashlib
import logging
import platform
import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any

from infinity_emb._optional_imports import CHECK_POSTHOG
from infinity_emb.args import EngineArgs
from infinity_emb.env import MANAGER
from infinity_emb.log_handler import logger

if CHECK_POSTHOG.is_available:
    from posthog import Posthog


@dataclass
class ProductTelemetryEvent:
    @abstractmethod
    def render(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


@dataclass
class StartupTelemetry(ProductTelemetryEvent):
    engine_args: list[EngineArgs]

    def render(self):
        return asdict(self)

    def name(self):
        return "startup"


@cache
def get_system_anonymous_name():
    attributes = []

    # Kernel version
    attributes.append(platform.uname().release)

    # OS information
    attributes.append(platform.uname().version)

    # Machine hardware name
    attributes.append(platform.uname().machine)

    # Combine attributes and hash them
    fingerprint_str = "|".join(attributes)
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode()).hexdigest()
    return fingerprint_hash


class _PostHogCapture:
    def __init__(self):
        self._posthog = None

        if (
            (not CHECK_POSTHOG.is_available)
            or "pytest" in sys.modules
            or (not MANAGER.anonymous_usage_stats)
        ):
            return
        try:
            # self.anonymous_user_id = hashlib.md5(
            #     socket.gethostname().encode("utf-8", errors="ignore")
            # ).hexdigest()
            self.anonymous_user_id = get_system_anonymous_name()
        except Exception:
            self.anonymous_user_id = "UNKNOWN"

        try:
            logger.debug(
                "Anonymized telemetry enabled. See \
                    https://michaelfeil.github.io/infinity for more information."
            )
            self._posthog = Posthog(
                (
                    "ph"  # split
                    "c_1l0OUnO8H0dUHjc"  # to avoid spam.
                    "AURlAGQUuyDLhncR8mFeP6LLO4DJ"
                ),
                host="https://eu.i.posthog.com",
            )

            posthog_logger = logging.getLogger("posthog")
            # Silence posthog's logging
            posthog_logger.disabled = True

        except Exception:
            logger.debug("Failed to startup posthog")

    def capture(self, event: ProductTelemetryEvent) -> None:
        if self._posthog is None:
            return
        try:
            self._posthog.capture(
                distinct_id=self.anonymous_user_id,
                event=event.name(),
                properties=event.render(),
            )
        except Exception as e:
            logger.debug(f"Failed to send telemetry event {event}: {e}")


PostHog = _PostHogCapture()
