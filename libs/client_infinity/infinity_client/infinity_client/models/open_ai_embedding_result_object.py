from enum import Enum


class OpenAIEmbeddingResultObject(str, Enum):
    EMBEDDING = "embedding"

    def __str__(self) -> str:
        return str(self.value)
