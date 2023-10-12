import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import numpy as np

from infinity_emb.inference.threading_asyncio import EventTS

NpEmbeddingType = np.ndarray


@dataclass
class EmbeddingResult:
    sentence: str
    event: EventTS
    uuid: str = field(default_factory=lambda: str(uuid4()))
    created: float = field(default_factory=time.time)
    embedding: Optional[NpEmbeddingType] = None


@dataclass(order=True)
class PrioritizedQueueItem:
    priority: int
    item: EmbeddingResult = field(compare=False)


@dataclass
class OverloadStatus:
    queue_fraction: float
    queue_absolute: int
    results_absolute: int
