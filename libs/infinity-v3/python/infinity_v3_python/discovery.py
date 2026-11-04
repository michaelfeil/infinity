"""Discovery for installed Python backend plugins."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points
from types import ModuleType
from typing import Any, Callable

from .api import ENTRY_POINT_GROUP, PROTOCOL_VERSION


class PluginError(RuntimeError):
    """Raised when a backend plugin cannot be loaded or is incompatible."""


def load_factory(selector: str | None = None) -> Callable[..., Any]:
    """Load a factory from ``module:attribute`` or an entry-point name.

    With no selector, use the sole installed entry point and fail with the
    available names when the choice is ambiguous.
    """

    if selector is None:
        return _load_default()
    selector = selector.strip()
    if not selector:
        return _load_default()
    if ":" in selector:
        module_name, attribute = selector.split(":", 1)
        module = _import(module_name, selector)
        factory = getattr(module, attribute, None)
        if not callable(factory):
            raise PluginError(
                f"Python backend selector {selector!r} does not resolve to a callable"
            )
        _check_module(module, selector)
        return factory
    if selector.startswith("infinity_v3_python."):
        module = _import(selector, selector)
        factory = getattr(module, "create_executor", None)
        if not callable(factory):
            raise PluginError(
                f"Python backend module {selector!r} has no create_executor"
            )
        _check_module(module, selector)
        return factory
    matches = [
        entry_point
        for entry_point in entry_points(group=ENTRY_POINT_GROUP)
        if entry_point.name == selector
    ]
    if not matches:
        raise PluginError(
            f"no Python backend entry point named {selector!r} in group "
            f"{ENTRY_POINT_GROUP!r}"
        )
    if len(matches) > 1:
        raise PluginError(
            f"multiple Python backend entry points named {selector!r}; "
            "remove the conflicting distributions"
        )
    try:
        factory = matches[0].load()
    except Exception as error:
        raise PluginError(
            f"could not load Python backend entry point {selector!r}: {error}"
        ) from error
    if not callable(factory):
        raise PluginError(
            f"Python backend entry point {selector!r} does not resolve to a callable"
        )
    module_name = getattr(factory, "__module__", None)
    if module_name:
        module = _import(module_name, selector)
        _check_module(module, selector)
    return factory


def load_capabilities(
    selector: str | None = None,
    config_json: str = "{}",
) -> tuple[int, bool] | None:
    """Return optional ``(visible_device_count, supports_replication)``.

    A plugin reports this by exporting a module-level ``capabilities`` hook
    that accepts the factory's ``config_json`` and returns a dict with exactly
    ``visible_device_count`` and ``supports_replication``. Plugins without the
    hook keep the protocol-1 single-device defaults.
    """

    factory = load_factory(selector)
    module_name = getattr(factory, "__module__", None)
    if not module_name:
        return None
    module = _import(module_name, selector or repr(factory))
    hook = getattr(module, "capabilities", None)
    if hook is None:
        return None
    try:
        value = hook(config_json)
    except Exception as error:
        raise PluginError(
            f"Python backend {selector or module_name!r} capabilities hook "
            f"failed: {error}"
        ) from error
    if not isinstance(value, dict) or set(value) != {
        "visible_device_count",
        "supports_replication",
    }:
        raise PluginError(
            f"Python backend {selector or module_name!r} capabilities must be "
            "a dict with visible_device_count and supports_replication"
        )
    count = value["visible_device_count"]
    replication = value["supports_replication"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise PluginError(
            f"Python backend {selector or module_name!r} capabilities "
            f"visible_device_count must be a positive integer, got {count!r}"
        )
    if not isinstance(replication, bool):
        raise PluginError(
            f"Python backend {selector or module_name!r} capabilities "
            f"supports_replication must be a boolean, got {replication!r}"
        )
    return count, replication


def list_plugins() -> list[str]:
    """Return the installed Python backend entry-point names."""

    return sorted(
        entry_point.name
        for entry_point in entry_points(group=ENTRY_POINT_GROUP)
    )


def _load_default() -> Callable[..., Any]:
    names = list_plugins()
    if len(names) == 1:
        return load_factory(names[0])
    if not names:
        raise PluginError(
            "no Python backend entry point is installed; install a backend "
            "plugin or set \"backend\" in --python-config"
        )
    raise PluginError(
        "multiple Python backend entry points are installed "
        f"({', '.join(names)}); set \"backend\" in --python-config"
    )


def _import(module_name: str, selector: str) -> ModuleType:
    try:
        return import_module(module_name)
    except Exception as error:
        raise PluginError(
            f"could not import Python backend module {module_name!r} for "
            f"selector {selector!r}: {error}"
        ) from error


def _check_module(module: ModuleType, selector: str) -> None:
    version = getattr(module, "PROTOCOL_VERSION", None)
    if version is None:
        raise PluginError(
            f"Python backend {selector!r} must export PROTOCOL_VERSION"
        )
    if version != PROTOCOL_VERSION:
        raise PluginError(
            f"Python backend {selector!r} uses protocol {version}, "
            f"expected {PROTOCOL_VERSION}"
        )
