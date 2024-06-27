from enum import Enum


class ClassifyResultObject(str, Enum):
    CLASSIFY = "classify"

    def __str__(self) -> str:
        return str(self.value)
