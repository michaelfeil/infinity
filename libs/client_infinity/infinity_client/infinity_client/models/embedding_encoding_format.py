from enum import Enum


class EmbeddingEncodingFormat(str, Enum):
    BASE64 = "base64"
    FLOAT = "float"

    def __str__(self) -> str:
        return str(self.value)
