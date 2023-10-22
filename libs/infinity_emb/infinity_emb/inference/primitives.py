import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import numpy as np

# from infinity_emb.inference.threading_asyncio import EventTS

NpEmbeddingType = np.ndarray


@dataclass(order=True)
class EmbeddingResult:
    sentence: str = field(compare=False)
    uuid: str = field(default_factory=lambda: str(uuid4()), compare=False)
    embedding: Optional[NpEmbeddingType] = field(default=None, compare=False)
    future: Optional[asyncio.Future] = field(default=None, compare=False)
    # event: Optional[EventTS] = field(default=None, compare=False)


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


if __name__ == "__main__":
    import bisect
    from concurrent.futures import ThreadPoolExecutor

    tp = ThreadPoolExecutor()
    r1 = EmbeddingResult(5, "hello")
    r2 = EmbeddingResult(6, "hello_")
    r3 = EmbeddingResult(6, "hello_")
    r1 < r2
    l1 = []
    bisect.insort(l1, r1)
    bisect.insort(l1, r2)
