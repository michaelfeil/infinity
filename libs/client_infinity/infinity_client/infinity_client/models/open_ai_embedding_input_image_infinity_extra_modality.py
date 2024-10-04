from enum import Enum


class OpenAIEmbeddingInputImageInfinityExtraModality(str, Enum):
    IMAGE = "image"

    def __str__(self) -> str:
        return str(self.value)
