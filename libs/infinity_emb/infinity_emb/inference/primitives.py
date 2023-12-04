import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import uuid4

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
    sentence: str = field(compare=False)
    future: asyncio.Future = field(compare=False)
    uuid: str = field(default_factory=lambda: str(uuid4()), compare=False)
    embedding: Optional[NpEmbeddingType] = field(default=None, compare=False)

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
class PrioritizedQueueItem:
    priority: int
    item: EmbeddingResult = field(compare=False)
    timestamp: float = field(default_factory=time.time, compare=False)


@dataclass
class OverloadStatus:
    queue_fraction: float
    queue_absolute: int
    results_absolute: int
