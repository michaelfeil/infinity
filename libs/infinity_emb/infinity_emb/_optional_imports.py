# files just imports external dependencies. Do not import any infinity_emb modules here.

from __future__ import annotations

import importlib.util
from functools import cached_property
from typing import Iterable, Optional


class OptionalImports:
    def __init__(
        self, lib: str, extra_install: str, dependencies: Optional[Iterable[str]] = None
    ) -> None:
        self.lib = lib
        self.extra_install = extra_install
        self._marked_as_dirty: Optional[Exception] = None
        self.dependencies = dependencies

    @cached_property
    def is_available(self) -> bool:
        if self.dependencies is not None:
            for dep in self.dependencies:
                if importlib.util.find_spec(dep) is None:
                    return False
        if "." in self.lib:
            # check module recursively
            lib = self.lib.split(".")
            for i in range(len(lib)):
                module = ".".join(lib[: i + 1])
                if importlib.util.find_spec(module) is None:
                    return False

        return importlib.util.find_spec(self.lib) is not None

    def mark_dirty(self, exception: Exception) -> None:
        """marking the import as dirty, e.g. when runtimeerror occurs."""
        self._marked_as_dirty = exception

    def mark_required(self) -> bool:
        if not self.is_available or self._marked_as_dirty:
            self._raise_error()
        return True

    def _raise_error(self) -> None:
        """raise ImportError if the library is not available."""
        msg = (
            f"{self.lib} is not available. "
            f"install via `pip install infinity-emb[{self.extra_install}]`"
        )
        if self._marked_as_dirty:
            raise ImportError(msg) from self._marked_as_dirty
        raise ImportError(msg)


CHECK_DISKCACHE = OptionalImports("diskcache", "cache")
CHECK_CTRANSLATE2 = OptionalImports("ctranslate2", "ctranslate2")
CHECK_FASTAPI = OptionalImports("fastapi", "server")
CHECK_HF_TRANSFER = OptionalImports("hf_transfer", "hf_transfer")
CHECK_ONNXRUNTIME = OptionalImports("optimum.onnxruntime", "optimum")
CHECK_OPTIMUM = OptionalImports("optimum", "optimum")
CHECK_OPTIMUM_NEURON = OptionalImports("optimum.neuron", "neuronx")
CHECK_SENTENCE_TRANSFORMERS = OptionalImports("sentence_transformers", "torch")
CHECK_TRANSFORMERS = OptionalImports("transformers", "torch")
CHECK_TORCH = OptionalImports("torch.nn", "torch")
CHECK_PYDANTIC = OptionalImports("pydantic", "server")
CHECK_TYPER = OptionalImports("typer", "server")
CHECK_UVICORN = OptionalImports("uvicorn", "server")
