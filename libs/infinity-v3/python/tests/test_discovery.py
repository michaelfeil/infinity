import json
import sys
import types
import unittest
from importlib.metadata import EntryPoint
from unittest.mock import patch

from infinity_v3_python.discovery import (
    factory_accepts_serving_limits,
    load_capabilities,
    load_factory,
)


class DiscoveryTests(unittest.TestCase):
    def test_serving_limits_opt_in_requires_an_explicit_parameter(self):
        def keyword_only(*, serving_limits=None, **kwargs):
            pass

        def positional_or_keyword(serving_limits=None, **kwargs):
            pass

        def legacy(**kwargs):
            pass

        self.assertTrue(factory_accepts_serving_limits(keyword_only))
        self.assertTrue(factory_accepts_serving_limits(positional_or_keyword))
        self.assertFalse(factory_accepts_serving_limits(legacy))
        self.assertFalse(factory_accepts_serving_limits(lambda: None))

    def test_reexported_factory_uses_selected_plugin_hooks(self):
        implementation = types.ModuleType("_discovery_implementation")
        exec(
            "PROTOCOL_VERSION = 1\ndef create_executor(**kwargs): pass",
            implementation.__dict__,
        )
        plugin = types.ModuleType("_discovery_plugin")
        plugin.PROTOCOL_VERSION = 1
        plugin.create_executor = implementation.create_executor
        plugin.capabilities = json.loads
        entry = EntryPoint(
            "example", "_discovery_plugin:create_executor", "infinity_v3.backends"
        )
        config = json.dumps({"visible_device_count": 2, "supports_replication": True})
        modules = {plugin.__name__: plugin, implementation.__name__: implementation}
        with patch.dict(sys.modules, modules):
            with patch("infinity_v3_python.discovery.entry_points", return_value=[entry]):
                for selector in ["_discovery_plugin:create_executor", "example", None]:
                    with self.subTest(selector=selector):
                        self.assertIs(load_factory(selector), implementation.create_executor)
                        self.assertEqual(load_capabilities(selector, config), (2, True))
                del plugin.capabilities
                self.assertIsNone(load_capabilities("example", config))
