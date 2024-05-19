"""
Definition of enums and dataclasses used in the library.

Do not import infinity_emb from this file, as it will cause a circular import.
"""
from __future__ import annotations

import asyncio
import enum
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# cached_porperty
from functools import lru_cache
from typing import Generic, Literal, Optional, Type, TypedDict, TypeVar, Union

import numpy as np
import numpy.typing as npt

# if python>=3.10 use kw_only

dataclass_args = {"kw_only": True} if sys.version_info >= (3, 10) else {}

EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]


class ClassifyReturnType(TypedDict):
    label: str
    score: float


ReRankReturnType = float

UnionReturnType = Union[EmbeddingReturnType, ReRankReturnType, ClassifyReturnType]


class EnumType(enum.Enum):
    @classmethod
    @lru_cache
    def names_enum(cls) -> enum.Enum:
        """returns an enum with the same names as the class.

        Allows for type hinting of the enum names.
        """
        return enum.Enum(
            cls.__name__ + "__names", {k: k for k in cls.__members__.keys()}
        )

    @abstractmethod
    def default_value(self) -> str:
        ...


class InferenceEngine(EnumType):
    torch = "torch"
    ctranslate2 = "ctranslate2"
    optimum = "optimum"
    debugengine = "debugengine"

    @classmethod
    def default_value(self):
        return InferenceEngine.torch.value


class Device(EnumType):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    tensorrt = "tensorrt"
    auto = "auto"

    @classmethod
    def default_value(self):
        return Device.auto.value

    def resolve(self) -> Optional[str]:
        if self == Device.auto:
            return None
        return self.value


class Dtype(EnumType):
    float32: str = "float32"
    float16: str = "float16"
    int8: str = "int8"
    fp8: str = "fp8"
    auto: str = "auto"

    @classmethod
    def default_value(self):
        return Dtype.auto.value


class EmbeddingDtype(EnumType):
    float32: str = "float32"
    # int8: str = "int8"
    # binary: str = "binary"

    @classmethod
    def default_value(self):
        return EmbeddingDtype.float32.value


class PoolingMethod(EnumType):
    mean: str = "mean"
    cls: str = "cls"
    auto: str = "auto"

    @classmethod
    def default_value(self):
        return PoolingMethod.auto.value


@dataclass(**dataclass_args)
class AbstractSingle(ABC):
    @abstractmethod
    def str_repr(self) -> str:
        pass

    @abstractmethod
    def to_input(self) -> Union[str, tuple[str, str]]:
        pass


@dataclass(**dataclass_args)
class EmbeddingSingle(AbstractSingle):
    sentence: str

    def str_repr(self) -> str:
        return self.sentence

    def to_input(self) -> str:
        return self.sentence


@dataclass(**dataclass_args)
class ReRankSingle(AbstractSingle):
    query: str
    document: str

    def str_repr(self) -> str:
        return self.query + self.document

    def to_input(self) -> tuple[str, str]:
        return self.query, self.document


@dataclass(**dataclass_args)
class PredictSingle(EmbeddingSingle):
    pass


AbstractInnerType = TypeVar("AbstractInnerType")


@dataclass(order=True, **dataclass_args)
class AbstractInner(ABC, Generic[AbstractInnerType]):
    content: AbstractSingle
    future: asyncio.Future

    @abstractmethod
    async def complete(self, result: AbstractInnerType) -> None:
        pass

    @abstractmethod
    async def get_result(self) -> AbstractInnerType:
        pass


@dataclass(order=True, **dataclass_args)
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
        assert self.embedding is not None
        return self.embedding


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
        assert self.score is not None
        return self.score


@dataclass(order=True)
class PredictInner(AbstractInner):
    content: PredictSingle
    class_encoding: Optional[ClassifyReturnType] = None

    async def complete(self, result: ClassifyReturnType) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.class_encoding = result

        if self.class_encoding is None:
            raise ValueError("class_encoding is None")
        try:
            self.future.set_result(self.class_encoding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> ClassifyReturnType:
        """waits for future to complete and returns result"""
        await self.future
        assert self.class_encoding is not None
        return self.class_encoding


QueueItemInner = Union[EmbeddingInner, ReRankInner, PredictInner]

_type_to_inner_item_map = {
    EmbeddingSingle: EmbeddingInner,
    ReRankSingle: ReRankInner,
    PredictSingle: PredictInner,
}


def get_inner_item(single_type: Type[AbstractSingle]) -> Type[QueueItemInner]:
    if single_type not in _type_to_inner_item_map:
        raise ValueError(f"Unknown type of input_single_item, {single_type}")

    return _type_to_inner_item_map[single_type]  # type: ignore


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
