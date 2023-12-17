import asyncio
import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

EmbeddingReturnType = np.ndarray


class Device(enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    auto = None


_devices: Dict[str, str] = {e.name: e.name for e in Device}
DeviceTypeHint = enum.Enum("DeviceTypeHint", _devices)  # type: ignore


@dataclass
class EmbeddingSingle:
    sentence: str

    def str_repr(self) -> str:
        return self.sentence

    def to_input(self) -> str:
        return self.sentence


@dataclass
class ReRankSingle:
    query: str
    document: str

    def str_repr(self) -> str:
        return self.query + self.document

    def to_input(self) -> Tuple[str, str]:
        return self.query, self.document


PipelineItem = Union[EmbeddingSingle, ReRankSingle]


@dataclass(order=True)
class EmbeddingInner:
    content: EmbeddingSingle
    future: asyncio.Future
    embedding: Optional[EmbeddingReturnType] = None

    def complete(self):
        """marks the future for completion.
        only call from the same thread as created future."""
        if self.embedding is None:
            raise ValueError("calling complete on an uncompleted embedding")
        try:
            self.future.set_result(self.embedding)
        except asyncio.exceptions.InvalidStateError:
            pass


@dataclass(order=True)
class ReRankInner:
    content: ReRankSingle
    future: asyncio.Future
    score: Optional[float] = field(default=None, compare=False)

    def complete(self):
        """marks the future for completion.
        only call from the same thread as created future."""
        if self.score is None:
            raise ValueError("calling complete on an uncompleted embedding")
        try:
            self.future.set_result(self.score)
        except asyncio.exceptions.InvalidStateError:
            pass


QueueItemInner = Union[EmbeddingInner, ReRankInner]


@dataclass(order=True)
class PrioritizedQueueItem:
    priority: int
    item: QueueItemInner = field(compare=False)


@dataclass
class OverloadStatus:
    queue_fraction: float
    queue_absolute: int
    results_absolute: int
