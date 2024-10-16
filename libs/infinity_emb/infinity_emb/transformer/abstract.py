# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import random
from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, Any, Union

from infinity_emb._optional_imports import CHECK_PIL  # , CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioInner,
    AudioInputType,
    # AudioSingle,
    EmbeddingDtype,
    EmbeddingInner,
    EmbeddingReturnType,
    EmbeddingSingle,
    ImageInner,
    ImageSingle,
    ModelCapabilites,
    PredictInner,
    PredictSingle,
    ReRankInner,
    ReRankSingle,
)
from infinity_emb.transformer.quantization.interface import quant_embedding_decorator

INPUT_FEATURE = Any
OUT_FEATURES = Any


if TYPE_CHECKING:
    from PIL.Image import Image as ImageClass

    from infinity_emb.args import EngineArgs

if CHECK_PIL.is_available:
    from PIL import Image

# if CHECK_SOUNDFILE:
#     import soundfile as sf


class BaseTransformer(ABC):  # Inherit from ABC(Abstract base class)
    capabilities: set[ModelCapabilites] = set()
    engine_args: "EngineArgs"

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, *args, **kwargs) -> Any:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_core(self, features: INPUT_FEATURE) -> OUT_FEATURES:
        """runs plain inference, on cpu/gpu"""

    @abstractmethod
    def encode_post(self, *args, **kwargs) -> Any:
        """postprocessing of the inference"""

    @abstractmethod
    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""

    @abstractmethod
    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        """warmup the model

        returns embeddings per second, inference time, and a log message"""


class BaseEmbedder(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"embed"}

    @property
    def embedding_dtype(self) -> EmbeddingDtype:
        """returns the dtype of the embeddings"""
        return self.engine_args.embedding_dtype

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: list[Union[str, Any]]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_post(
        self, embedding: OUT_FEATURES, skip_quanitzation=True
    ) -> EmbeddingReturnType:
        """runs post encoding such as normalization"""

    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        sample = ["warm " * n_tokens] * batch_size
        inp = [
            EmbeddingInner(content=EmbeddingSingle(sentence=s), future=None)  # type: ignore
            for s in sample
        ]
        return run_warmup(self, inp)


class BaseTIMM(BaseEmbedder):  # Inherit from ABC(Abstract base class)
    capabilities = {"embed", "image_embed"}

    @property
    def embedding_dtype(self) -> EmbeddingDtype:
        """returns the dtype of the embeddings"""
        return self.engine_args.embedding_dtype

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(
        self, sentences_or_images: list[Union[str, "ImageClass"]]
    ) -> INPUT_FEATURE:
        """
        takes a list of sentences, or a list of images.
        Images could be url or numpy arrays/pil
        """

    @abstractmethod
    def encode_post(
        self, embedding: OUT_FEATURES, skip_quanitzation=True
    ) -> EmbeddingReturnType:
        """runs post encoding such as normalization"""

    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        sample_text = ["warm " * n_tokens] * max(1, batch_size // 2)
        sample_image = [Image.new("RGB", (128, 128), (255, 255, 255))] * max(1, batch_size // 2)  # type: ignore
        inp = [
            # TODO: warmup for images
            ImageInner(content=ImageSingle(image=img), future=None)  # type: ignore
            for img in sample_image
        ] + [
            EmbeddingInner(
                content=EmbeddingSingle(sentence=s), future=None  # type: ignore
            )
            for s in sample_text
        ]
        random.shuffle(inp)

        return run_warmup(self, inp)


class BaseAudioEmbedModel(BaseEmbedder):  # Inherit from ABC(Abstract base class)
    capabilities = {"embed", "audio_embed"}

    @property
    def embedding_dtype(self) -> EmbeddingDtype:
        """returns the dtype of the embeddings"""
        return self.engine_args.embedding_dtype  # type: ignore

    @property
    def sampling_rate(self) -> int:
        raise NotImplementedError

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(
        self, sentences_or_audios: list[Union[str, AudioInputType]]
    ) -> INPUT_FEATURE:
        """
        takes a list of sentences, or a list of audios.
        Audios could be raw byte array of the wave file
        """

    @abstractmethod
    def encode_post(
        self, embedding: OUT_FEATURES, skip_quanitzation=True
    ) -> EmbeddingReturnType:
        """runs post encoding such as normalization"""

    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        sample_text = ["warm " * n_tokens] * max(1, batch_size // 2)
        # sample_audios = [sf.SoundFile()] * max(1, batch_size // 2)  # type: ignore
        inp: list[Union[AudioInner, EmbeddingInner]] = [
            # TODO: warmup for audio
            # AudioInner(content=AudioSingle(audio=audio), future=None)  # type: ignore
            # for audio in sample_audios
        ] + [
            EmbeddingInner(
                content=EmbeddingSingle(sentence=s), future=None  # type: ignore
            )
            for s in sample_text
        ]
        random.shuffle(inp)

        return run_warmup(self, inp)


class BaseClassifer(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"classify"}

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: list[str]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> dict[str, float]:
        """runs post encoding such as normalization"""

    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        sample = ["warm " * n_tokens] * batch_size
        inp = [
            PredictInner(content=PredictSingle(sentence=s), future=None)  # type: ignore
            for s in sample
        ]
        return run_warmup(self, inp)


class BaseCrossEncoder(BaseTransformer):  # Inherit from ABC(Abstract base class)
    capabilities = {"rerank"}

    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, queries_docs: list[tuple[str, str]]) -> INPUT_FEATURE:
        """takes care of the tokenization and feature preparation"""

    @abstractmethod
    @quant_embedding_decorator()
    def encode_post(self, embedding: OUT_FEATURES) -> list[float]:
        """runs post encoding such as normalization"""

    def warmup(self, *, batch_size: int = 64, n_tokens=1) -> tuple[float, float, str]:
        sample = ["warm " * n_tokens] * batch_size
        inp = [
            ReRankInner(
                content=ReRankSingle(query=s, document=s), future=None  # type: ignore
            )
            for s in sample
        ]
        return run_warmup(self, inp)


def run_warmup(model, inputs) -> tuple[float, float, str]:
    inputs_formated = [i.content.to_input() for i in inputs]
    start = perf_counter()
    feat = model.encode_pre(inputs_formated)
    tokenization_time = perf_counter()
    embed = model.encode_core(feat)
    inference_time = perf_counter()
    model.encode_post(embed)
    post_time = perf_counter()

    analytics_string = (
        f"Getting timings for batch_size={len(inputs)}"
        " and avg tokens per sentence="
        f"{model.tokenize_lengths([i.content.str_repr() for i in inputs])[0]}\n"
        f"\t{(tokenization_time - start)*1000:.2f} \t ms tokenization\n"
        f"\t{(inference_time-tokenization_time)*1000:.2f} \t ms inference\n"
        f"\t{(post_time-inference_time)*1000:.2f} \t ms post-processing\n"
        f"\t{(post_time - start)*1000:.2f} \t ms total\n"
        f"embeddings/sec: {len(inputs) / (post_time - start):.2f}"
    )

    emb_per_sec = len(inputs) / (post_time - start)
    return emb_per_sec, inference_time - tokenization_time, analytics_string
