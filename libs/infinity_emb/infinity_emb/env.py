# cache
from __future__ import annotations

import os
from functools import cached_property

class __Infinity_EnvManager:
    def __init__(self):
        self._debug(f"Loading Infinity ENV variables.\nCONFIG:\n{'-'*10}")
        for f_name in dir(self):
            if isinstance(getattr(type(self), f_name, None), cached_property):
                getattr(self, f_name)  # pre-cache
        self._debug(f"{'-'*10}\nENV variables loaded.")
        
    def _debug(self, message: str):
        if "API_KEY" in message:
            print("INFINITY_API_KEY=not_shown")
            print(f"INFINITY_LOG_LEVEL={self.log_level}")
        elif "LOG_LEVEL" in message:
            return # recursion
        elif self.log_level in {"debug", "trace"}:
            print(message)
            
    @staticmethod
    def _to_name(name: str) -> str:
        return "INFINITY_" + name.upper().replace("-", "_")
    
    def _optional_infinity_var(self, name: str, default: str = ""):
        name = self._to_name(name)
        value = os.getenv(name)
        if value is None:
            self._debug(f"{name}=`{default}`(default)")
            return default
        self._debug(f"{name}=`{value}`")
        return value

    def _optional_infinity_var_multiple(self, name: str, default: list[str]) -> list[str]:
        name = self._to_name(name)
        value = os.getenv(name)
        if value is None:
            self._debug(f"{name}=`{';'.join(default)}`(default)")
            return default
        if value.endswith(";"):
            value = value[:-1]
        value = value.split(";")
        self._debug(f"{name}=`{';'.join(value)}`")
        return value

    @staticmethod
    def _to_bool(value: str) -> bool:
        return value.lower() in {"true", "1"}

    @staticmethod
    def _to_bool_multiple(value: list[str]) -> list[bool]:
        return [v.lower() in {"true", "1"} for v in value]

    @staticmethod
    def _to_int_multiple(value: list[str]) -> list[int]:
        return [int(v) for v in value]

    @cached_property
    def api_key(self):
        return self._optional_infinity_var("api_key", default="")

    @cached_property
    def model_id(self):
        return self._optional_infinity_var_multiple(
            "model_id", default=["michaelfeil/bge-small-en-v1.5"]
        )

    @cached_property
    def served_model_name(self):
        return self._optional_infinity_var_multiple("served_model_name", default=[""])

    @cached_property
    def batch_size(self):
        return self._to_int_multiple(
            self._optional_infinity_var_multiple("batch_size", default=["32"])
        )

    @cached_property
    def revision(self):
        return self._optional_infinity_var_multiple("revision", default=[""])

    @cached_property
    def trust_remote_code(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple("trust_remote_code", default=["true"])
        )

    @cached_property
    def model_warmup(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple("model_warmup", default=["true"])
        )

    @cached_property
    def vector_disk_cache(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple("vector_disk_cache", default=["false"])
        )

    @cached_property
    def lengths_via_tokenize(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple(
                "lengths_via_tokenize", default=["false"]
            )
        )

    @cached_property
    def compile(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple("compile", default=["false"])
        )

    @cached_property
    def bettertransformer(self):
        return self._to_bool_multiple(
            self._optional_infinity_var_multiple("bettertransformer", default=["true"])
        )

    @cached_property
    def preload_only(self):
        return self._to_bool(
            self._optional_infinity_var("preload_only", default="false")
        )

    @cached_property
    def permissive_cors(self):
        return self._to_bool(
            self._optional_infinity_var("permissive_cors", default="false")
        )

    @cached_property
    def url_prefix(self):
        return self._optional_infinity_var("url_prefix", default="")

    @cached_property
    def port(self):
        port = self._optional_infinity_var("port", default="7997")
        assert port.isdigit(), "INFINITY_PORT must be a number"
        return int(port)

    @cached_property
    def host(self):
        return self._optional_infinity_var("host", default="0.0.0.0")

    @cached_property
    def redirect_slash(self):
        route = self._optional_infinity_var("redirect_slash", default="/docs")
        assert not route or route.startswith(
            "/"
        ), "INFINITY_REDIRECT_SLASH must start with /"
        return route

    @cached_property
    def log_level(self):
        return self._optional_infinity_var("log_level", default="info")


MANAGER = __Infinity_EnvManager()
