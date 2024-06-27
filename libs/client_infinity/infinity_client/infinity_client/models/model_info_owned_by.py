from enum import Enum


class ModelInfoOwnedBy(str, Enum):
    INFINITY = "infinity"

    def __str__(self) -> str:
        return str(self.value)
