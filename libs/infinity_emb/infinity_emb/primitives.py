"""
Definition of enums and dataclasses used in the library.

Do not import infinity_emb from this file, as it will cause a circular import.
"""
# if python>=3.10 use kw_only
import asyncio
import enum
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

dataclass_args = {"kw_only": True} if sys.version_info >= (3, 10) else {}

EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]


class InferenceEngine(enum.Enum):
    torch = "torch"
    ctranslate2 = "ctranslate2"
    optimum = "optimum"
    fastembed = "fastembed"
    debugengine = "dummytransformer"


class Device(enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    tensorrt = "tensorrt"
    auto = None


class Dtype(enum.Enum):
    float16: str = "float16"
    auto: str = "auto"


class PoolingMethod(enum.Enum):
    mean: str = "mean"
    cls: str = "cls"
    auto: str = "auto"


_devices: Dict[str, str] = {e.name: e.name for e in Device}
DeviceTypeHint = enum.Enum("DeviceTypeHint", _devices)  # type: ignore


@dataclass
class AbstractSingle(ABC):
    @abstractmethod
    def str_repr(self) -> str:
        pass

    @abstractmethod
    def to_input(self) -> Union[str, Tuple[str, str]]:
        pass


@dataclass
class EmbeddingSingle(AbstractSingle):
    sentence: str

    def str_repr(self) -> str:
        return self.sentence

    def to_input(self) -> str:
        return self.sentence


@dataclass
class ReRankSingle(AbstractSingle):
    query: str
    document: str

    def str_repr(self) -> str:
        return self.query + self.document

    def to_input(self) -> Tuple[str, str]:
        return self.query, self.document


@dataclass
class PredictSingle(EmbeddingSingle):
    pass


PipelineItem = Union[EmbeddingSingle, ReRankSingle, PredictSingle]


@dataclass(order=True)
class AbstractInner(ABC):
    future: asyncio.Future

    @abstractmethod
    async def complete(self, result: Any) -> None:
        pass

    @abstractmethod
    async def get_result(self) -> Any:
        pass


@dataclass(order=True)
class EmbeddingInner(AbstractInner):
    content: EmbeddingSingle
    embedding: Optional[EmbeddingReturnType] = None

    async def complete(self, result: EmbeddingReturnType) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.embedding = result

        if self.embedding is None:
            raise ValueError("embedding is None")
        try:
            self.future.set_result(self.embedding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> EmbeddingReturnType:
        """waits for future to complete and returns result"""
        await self.future
        return self.embedding  # type: ignore


@dataclass(order=True)
class ReRankInner(AbstractInner):
    content: ReRankSingle
    score: Optional[float] = field(default=None, compare=False)

    async def complete(self, result: float) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.score = result

        if self.score is None:
            raise ValueError("score is None")
        try:
            self.future.set_result(self.score)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> float:
        """waits for future to complete and returns result"""
        await self.future
        return self.score  # type: ignore


@dataclass(order=True)
class PredictInner(AbstractInner):
    content: PredictSingle
    class_encoding: Optional[EmbeddingReturnType] = None

    async def complete(self, result: EmbeddingReturnType) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.class_encoding = result

        if self.class_encoding is None:
            raise ValueError("class_encoding is None")
        try:
            self.future.set_result(self.class_encoding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> EmbeddingReturnType:
        """waits for future to complete and returns result"""
        await self.future
        return self.class_encoding  # type: ignore


QueueItemInner = Union[EmbeddingInner, ReRankInner, PredictInner]


@dataclass(order=True)
class PrioritizedQueueItem:
    priority: int
    item: QueueItemInner = field(compare=False)


@dataclass
class OverloadStatus:
    queue_fraction: float
    queue_absolute: int
    results_absolute: int


class ModelNotDeployedError(Exception):
    pass


ModelCapabilites = Literal["embed", "rerank", "classify"]

_types: Dict[str, str] = {e.name: e.name for e in InferenceEngine}
InferenceEngineTypeHint = enum.Enum("InferenceEngineTypeHint", _types)  # type: ignore
