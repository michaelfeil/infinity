import asyncio
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

EmbeddingReturnType = np.ndarray


class Device(enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    auto = None


_devices: Dict[str, str] = {e.name: e.name for e in Device}
DeviceTypeHint = enum.Enum("DeviceTypeHint", _devices)  # type: ignore


@dataclass
class AbstractSingle(ABC):
    @abstractmethod
    def str_repr(self) -> str:
        pass

    @abstractmethod
    def to_input(self) -> str:
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
    def complete(self, *args) -> None:
        pass

    @abstractmethod
    def get_result(self) -> Any:
        pass


@dataclass(order=True)
class EmbeddingInner(AbstractInner):
    content: EmbeddingSingle
    embedding: Optional[EmbeddingReturnType] = None

    async def complete(self, embedding: EmbeddingReturnType) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.embedding = embedding

        if self.embedding is None:
            raise ValueError("embedding is None")
        try:
            self.future.set_result(self.embedding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> EmbeddingReturnType:
        """waits for future to complete and returns result"""
        await self.future
        return self.embedding


@dataclass(order=True)
class ReRankInner(AbstractInner):
    content: ReRankSingle
    score: Optional[float] = field(default=None, compare=False)

    async def complete(self, score: float) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.score = score

        if self.score is None:
            raise ValueError("score is None")
        try:
            self.future.set_result(self.score)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> float:
        """waits for future to complete and returns result"""
        await self.future
        return self.score


@dataclass(order=True)
class PredictInner(AbstractInner):
    content: PredictSingle
    class_encoding: Optional[EmbeddingReturnType] = None

    async def complete(self, class_encoding: EmbeddingReturnType) -> None:
        """marks the future for completion.
        only call from the same thread as created future."""
        self.embeclass_encodingdding = class_encoding

        if self.class_encoding is None:
            raise ValueError("embedding is None")
        try:
            self.future.set_result(self.class_encoding)
        except asyncio.exceptions.InvalidStateError:
            pass

    async def get_result(self) -> EmbeddingReturnType:
        """waits for future to complete and returns result"""
        await self.future
        return self.class_encoding


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
