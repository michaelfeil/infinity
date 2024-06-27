from enum import Enum


class ModelInfoObject(str, Enum):
    MODEL = "model"

    def __str__(self) -> str:
        return str(self.value)
