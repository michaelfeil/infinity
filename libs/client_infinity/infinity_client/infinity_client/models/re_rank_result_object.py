from enum import Enum


class ReRankResultObject(str, Enum):
    RERANK = "rerank"

    def __str__(self) -> str:
        return str(self.value)
