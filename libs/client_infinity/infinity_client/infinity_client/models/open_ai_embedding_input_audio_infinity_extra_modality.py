from enum import Enum


class OpenAIEmbeddingInputAudioInfinityExtraModality(str, Enum):
    AUDIO = "audio"

    def __str__(self) -> str:
        return str(self.value)
