import hashlib
import logging
import os
import platform
import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from infinity_emb._optional_imports import CHECK_POSTHOG, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.env import MANAGER
from infinity_emb.log_handler import logger
from infinity_emb.primitives import ModelCapabilites

if CHECK_POSTHOG.is_available:
    import posthog
    from posthog import Posthog
if CHECK_TORCH.is_available:
    import torch


@dataclass
class ProductTelemetryEvent:
    @abstractmethod
    def render(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


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


@cache
def infinity_version():
    from infinity_emb import __version__

    return __version__


@cache
def get_system_properties():
    gpu_count = 0
    gpu_type = ""
    gpu_memory_per_device_mb = 0
    if CHECK_TORCH.is_available:
        if torch.cuda.is_available():
            device_property = torch.cuda.get_device_properties(0)
            gpu_count = torch.cuda.device_count()
            gpu_type = str(device_property.name)
            gpu_memory_per_device_mb = (
                int(device_property.total_memory) * 1000000 / 1024**2
            )

    return {
        "gpu_count": gpu_count,
        "gpu_type": gpu_type,
        "gpu_memory_per_device_mb": gpu_memory_per_device_mb,
    }


@cache
def _detect_cloud_provider() -> str:
    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    # Try detecting through vendor file
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    return "UNKNOWN"


@cache
def _get_cpu_info():
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
    except Exception:
        info = {}
    return {
        "count": info.get("count", -1),
        "cpu_type": info.get("brand_raw", ""),
        "cpu_family_model_stepping": ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        ),
    }


@cache
def _get_os_info():
    try:
        import psutil  # type: ignore

        memory = psutil.virtual_memory().total // (1024**2)
    except Exception:
        memory = -1

    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "machine": platform.processor(),
        "total_memory": memory,
    }


@dataclass
class StartupTelemetry(ProductTelemetryEvent):
    engine_args: "EngineArgs"
    num_engines: int
    capabilities: set["ModelCapabilites"]
    session_id: str

    # auto populated fields
    cloud_provider: str = field(default_factory=_detect_cloud_provider)
    os: str = field(default_factory=lambda: _get_os_info()["os"])
    architecture: str = field(default_factory=lambda: _get_os_info()["architecture"])
    machine: str = field(default_factory=lambda: _get_os_info()["machine"])
    cpu_count: int = field(default_factory=lambda: _get_cpu_info()["count"])
    cpu_type: str = field(default_factory=lambda: _get_cpu_info()["cpu_type"])
    cpu_family_model_stepping: str = field(
        default_factory=lambda: _get_cpu_info()["cpu_family_model_stepping"]
    )
    total_memory: int = field(default_factory=lambda: _get_os_info()["total_memory"])
    gpu_count: int = field(default_factory=lambda: get_system_properties()["gpu_count"])
    gpu_type: str = field(default_factory=lambda: get_system_properties()["gpu_type"])
    gpu_memory_per_device_mb: int = field(
        default_factory=lambda: get_system_properties()["gpu_memory_per_device_mb"]
    )
    version: str = field(default_factory=infinity_version)

    def render(self):
        """defines the message to be sent to posthog"""
        return {
            **asdict(self.engine_args),
            "session_id": self.session_id,
            "num_engines": self.num_engines,
            "capabilities": self.capabilities,
            "cloud_provider": self.cloud_provider,
            "architecture": self.architecture,
            "os": self.os,
            "machine": self.machine,
            "cpu_count": self.cpu_count,
            "cpu_type": self.cpu_type,
            "cpu_family_model_stepping": self.cpu_family_model_stepping,
            "total_memory": self.total_memory,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "gpu_memory_per_device": self.gpu_memory_per_device_mb,
            "version": self.version,
        }

    def name(self):
        return "startup_v1"


class _PostHogCapture:
    def __init__(self):
        self._posthog = None
        disabled = False
        if not CHECK_POSTHOG.is_available or (not MANAGER.anonymous_usage_stats):
            return
        if "pytest" in sys.modules:
            # disable posthog
            disabled = True
            posthog.disabled = True

        try:
            logger.debug(
                "Anonymized telemetry enabled. See \
                    https://michaelfeil.github.io/infinity for more information."
            )
            k = (
                "ph"  # split
                "c_IOq"  # to avoid spam on project
                "2AjB200yaxV2qtYTmhAacFE4x42RKOW0K0G5v5uh"
            )
            self._posthog = Posthog(
                project_api_key=k,
                host="https://eu.i.posthog.com",
                disabled=disabled,
            )

            posthog_logger = logging.getLogger("posthog")
            # Silence posthog's logging
            posthog_logger.disabled = True

        except Exception:
            logger.debug("Failed to startup posthog")

    @property
    @cache
    def anonymous_user_id(self):
        return get_system_anonymous_name()

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
