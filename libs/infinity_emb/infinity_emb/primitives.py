import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import numpy.typing as npt

EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]


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


@dataclass
class AbstractInner(ABC):
    _result: Any
    _uuid: str = field(default_factory=lambda: str(uuid4()))

    def get_id(self) -> str:
        return self._uuid

    def get_result(self) -> Any:
        """returns result"""
        return self._result  # type: ignore

    def set_result(self, result: Any) -> None:
        if result is None:
            raise ValueError("result is None")
        self._result = result


@dataclass
class EmbeddingInner(AbstractInner):
    content: EmbeddingSingle = None  # type: ignore
    _result: Optional[EmbeddingReturnType] = None


@dataclass
class ReRankInner(AbstractInner):
    content: ReRankSingle = None  # type: ignore
    _result: Optional[float] = field(default=None, compare=False)


@dataclass
class PredictInner(AbstractInner):
    content: PredictSingle = None  # type: ignore
    _result: Optional[List[Dict[str, Any]]] = None


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
