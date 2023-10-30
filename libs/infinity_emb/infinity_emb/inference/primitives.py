import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import numpy as np

NpEmbeddingType = np.ndarray


@dataclass(order=True)
class EmbeddingResult:
    sentence: str = field(compare=False)
    future: asyncio.Future = field(compare=False)
    uuid: str = field(default_factory=lambda: str(uuid4()), compare=False)
    embedding: Optional[NpEmbeddingType] = field(default=None, compare=False)


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
