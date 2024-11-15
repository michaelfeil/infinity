# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from infinity_emb._optional_imports import CHECK_TORCH, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import AudioInputType
from infinity_emb.transformer.abstract import BaseAudioEmbedModel
from infinity_emb.transformer.quantization.interface import quant_embedding_decorator

if TYPE_CHECKING:
    from torch import Tensor

if CHECK_TORCH.is_available:
    import torch
if CHECK_TORCH.is_available and CHECK_TRANSFORMERS.is_available:
    from transformers import AutoModel, AutoProcessor  # type: ignore


class TorchAudioModel(BaseAudioEmbedModel):
    """Audio model for CLAP models"""

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        self.model = AutoModel.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            # attn_implementation="eager" if engine_args.bettertransformer else None,
        )

        # self.model = to_bettertransformer(
        #     self.model,
        #     engine_args,
        #     logger,
        # )
        self.processor = AutoProcessor.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self.engine_args = engine_args

        if engine_args.compile:
            self.model.vision_model = torch.compile(self.model.vision_model, dynamic=True)
            self.model.text_model = torch.compile(self.model.text_model, dynamic=True)

        assert hasattr(
            self.model, "get_text_features"
        ), f"AutoModel of {engine_args.model_name_or_path} does not have get_text_features method"
        assert hasattr(
            self.model, "get_audio_features"
        ), f"AutoModel of {engine_args.model_name_or_path} does not have get_audio_features method"
        self.max_length = None
        if hasattr(self.model.config, "max_length"):
            self.max_length = self.model.config.max_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "max_length"
        ):
            self.max_length = self.model.config.text_config.max_length
        self._sampling_rate = self.processor.feature_extractor.sampling_rate

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def encode_pre(self, sentences_or_audios: list[Union[str, "AudioInputType"]]):
        text_list: list[str] = []
        audio_list: list[Any] = []
        type_is_audio: list[bool] = []

        for audio_or_text in sentences_or_audios:
            if isinstance(audio_or_text, str):
                text_list.append(audio_or_text)
                type_is_audio.append(False)
            else:
                audio_list.append(audio_or_text)
                type_is_audio.append(True)

        preprocessed = self.processor(
            audios=audio_list if audio_list else None,
            text=text_list if text_list else None,
            return_tensors="pt",
            padding=True,
            sampling_rate=self.sampling_rate,
        )
        preprocessed.pop("token_type_ids", None)

        preprocessed = {k: v.to(self.model.device) for k, v in preprocessed.items()}

        return (preprocessed, type_is_audio)

    def _normalize_cpu(self, tensor: Optional["Tensor"]) -> Iterable["Tensor"]:
        if tensor is None:
            return iter([])
        return iter((tensor / tensor.norm(p=2, dim=-1, keepdim=True)).cpu().numpy())

    def encode_core(
        self, features_and_types: tuple[dict[str, "Tensor"], list[bool]]
    ) -> tuple["Tensor", "Tensor", list[bool]]:
        """
        Computes sentence embeddings
        """
        features, type_is_audio = features_and_types
        with torch.no_grad():
            # TODO: torch.cuda.stream()
            if "input_ids" in features:
                text_embeds = self.model.get_text_features(
                    input_ids=features.get("input_ids")[:, : self.max_length],  # type: ignore
                    attention_mask=features.get("attention_mask")[:, : self.max_length],  # type: ignore
                )
            else:
                text_embeds = None  # type: ignore

            if "input_features" in features:
                audio_embeds = self.model.get_audio_features(
                    input_features=features.get("input_features"),
                )
            else:
                audio_embeds = None

        return text_embeds, audio_embeds, type_is_audio

    @quant_embedding_decorator()
    def encode_post(self, out_features) -> list[float]:
        text_embeds, audio_embeds, type_is_audio = out_features
        text_embeds = self._normalize_cpu(text_embeds)
        audio_embeds = self._normalize_cpu(audio_embeds)

        embeddings = list(
            next(audio_embeds if is_audio else text_embeds) for is_audio in type_is_audio
        )

        return embeddings

    def tokenize_lengths(self, text_list: list[str]) -> list[int]:
        preprocessed = self.processor(
            text=text_list,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return [len(t) for t in preprocessed["input_ids"]]
