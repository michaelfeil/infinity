import asyncio
import enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, List

import numpy as np

NpEmbeddingType = np.ndarray


class Device(enum.Enum):
    cpu = "cpu"
    cuda = "cuda"
    auto = None


_devices: Dict[str, str] = {e.name: e.name for e in Device}
DeviceTypeHint = enum.Enum("DeviceTypeHint", _devices)  # type: ignore


@dataclass(order=True)
class EmbeddingResult:
    sentence: str 
    future: asyncio.Future 
    embedding: Optional[NpEmbeddingType] = None

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
class ReRankResult:
    query: str 
    documents: List[str] 
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


@dataclass(order=True)
class PrioritizedQueueItem:
    priority: int
    item: Union[EmbeddingResult, ReRankResult] = field(compare=False)


@dataclass
class OverloadStatus:
    queue_fraction: float
    queue_absolute: int
    results_absolute: int
