# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from infinity_emb._optional_imports import (
    CHECK_COLPALI_ENGINE,
    CHECK_PIL,
    CHECK_TORCH,
    CHECK_TRANSFORMERS,
)
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, Dtype
from infinity_emb.transformer.abstract import BaseTIMM
from infinity_emb.transformer.quantization.interface import (
    quant_embedding_decorator,
    quant_interface,
)
from infinity_emb.transformer.vision import IMAGE_COL_MODELS

if TYPE_CHECKING:
    from PIL.Image import Image as ImageClass
    from torch import Tensor

if CHECK_TORCH.is_available:
    import torch
if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoConfig, AutoModel, AutoProcessor  # type: ignore
if CHECK_PIL.is_available:
    from PIL import Image


class TIMM(BaseTIMM):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args: "EngineArgs"):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        base_config = dict(
            pretrained_model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        config = AutoConfig.from_pretrained(**base_config)
        self.is_colipali = config.architectures[0] in IMAGE_COL_MODELS
        self.mock_image = Image.new("RGB", (128, 128), color="black")

        extra_model_args = dict(**base_config)
        extra_processor_args = dict(**base_config)
        device = engine_args.device
        if device == Device.auto and torch.cuda.is_available():
            device = Device.cuda
        if device == "cuda" and engine_args.dtype in (Dtype.float16, Dtype.bfloat16):
            extra_model_args["torch_dtype"] = engine_args.dtype.value
        elif device == "cuda" and engine_args.dtype in (Dtype.auto):
            extra_model_args["torch_dtype"] = "float16"

        if self.is_colipali:
            CHECK_COLPALI_ENGINE.mark_required()
            from colpali_engine.models import (  # type: ignore
                ColIdefics2,
                ColIdefics2Processor,
                ColPali,
                ColPaliProcessor,
                ColQwen2,
                ColQwen2Processor,
            )

            model_cls = {
                "ColPali": ColPali,
                "ColQwen2": ColQwen2,
                "ColIdefics2": ColIdefics2,
            }[config.architectures[0]]
            processor_cls = {
                "ColPali": ColPaliProcessor,
                "ColQwen2": ColQwen2Processor,
                "ColIdefics2": ColIdefics2Processor,
            }[config.architectures[0]]

            self.model = model_cls.from_pretrained(
                **extra_model_args,
            )

            self.processor = processor_cls.from_pretrained(
                **extra_processor_args,
            )
        else:
            self.model = AutoModel.from_pretrained(
                **extra_model_args
                # attn_implementation="eager" if engine_args.bettertransformer else None,
            )

            self.processor = AutoProcessor.from_pretrained(
                **extra_processor_args,
            )
            assert hasattr(
                self.model, "get_text_features"
            ), f"AutoModel of {engine_args.model_name_or_path} does not have get_text_features method"
            assert hasattr(
                self.model, "get_image_features"
            ), f"AutoModel of {engine_args.model_name_or_path} does not have get_image_features method"
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if engine_args.dtype in (Dtype.float16, Dtype.auto):
                self.model = self.model.half()

        if engine_args.dtype in (Dtype.int8, Dtype.fp8):
            self.model = quant_interface(self.model, engine_args.dtype, device=device)
        self.engine_args = engine_args

        if engine_args.compile:
            if self.is_colipali:
                self.model = torch.compile(self.model, dynamic=True)
            else:
                self.model.vision_model = torch.compile(self.model.vision_model, dynamic=True)
                self.model.text_model = torch.compile(self.model.text_model, dynamic=True)

        self.max_length = None
        if hasattr(self.model.config, "max_length"):
            self.max_length = self.model.config.max_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "text_config") and hasattr(
            self.model.config.text_config, "max_length"
        ):
            self.max_length = self.model.config.text_config.max_length

    def encode_pre(self, sentences_or_images: list[Union[str, "ImageClass"]]):
        # return input_tuples
        text_list: list[str] = []
        image_list: list[Any] = []
        type_is_img: list[bool] = []

        for im_or_text in sentences_or_images:
            if isinstance(im_or_text, str):
                text_list.append(im_or_text)
                type_is_img.append(False)
            else:
                image_list.append(im_or_text)
                type_is_img.append(True)
        if self.is_colipali:
            preprocessed_q = {}  # type: ignore
            if text_list:
                preprocessed_q = {
                    k: v.to(self.model.device)
                    for k, v in self.processor.process_queries(text_list).items()
                }

            preprocessed_i = {}  # type: ignore
            if image_list:
                preprocessed_i = {
                    k: v.to(self.model.device)
                    for k, v in self.processor.process_images(image_list).items()
                }
            preprocessed = (preprocessed_q, preprocessed_i)  # type: ignore
        else:
            preprocessed = self.processor(
                images=image_list if image_list else None,
                text=text_list if text_list else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            preprocessed = {k: v.to(self.model.device) for k, v in preprocessed.items()}  # type: ignore

        return (preprocessed, type_is_img)

    def _normalize_cpu(self, tensor: Optional["Tensor"], normalize: bool) -> Iterable["Tensor"]:
        if tensor is None:
            return iter([])
        tensor = tensor.to(torch.float32)
        if normalize:
            return iter((tensor / tensor.norm(p=2, dim=-1, keepdim=True)).cpu().numpy())
        else:
            return iter(tensor.cpu().numpy())

    def encode_core(
        self, features_and_types: tuple[dict[str, "Tensor"], list[bool]]
    ) -> tuple["Tensor", "Tensor", list[bool]]:
        """
        Computes sentence embeddings
        """
        features, type_is_img = features_and_types
        text_embeds, image_embeds = None, None  # type: ignore
        with torch.no_grad():
            # TODO: torch.cuda.stream()
            if self.is_colipali:
                text, image = features
                if text:
                    text_embeds: "Tensor" = self.model.forward(  # type: ignore
                        **text,
                    )
                if image:
                    image_embeds: "Tensor" = self.model.forward(  # type: ignore
                        **image,
                    )
            else:
                if "input_ids" in features:
                    text_embeds: "Tensor" = self.model.get_text_features(  # type: ignore
                        input_ids=features.get("input_ids"),  # requires int32
                        attention_mask=features.get("attention_mask"),
                    )
                if "pixel_values" in features:
                    image_embeds: "Tensor" = self.model.get_image_features(  # type: ignore
                        pixel_values=features.get("pixel_values").to(self.model.dtype),  # type: ignore
                        # requires float32 or float16 or bfloat16
                    )
        return text_embeds, image_embeds, type_is_img  # type: ignore

    @quant_embedding_decorator()
    def encode_post(self, out_features) -> list[float]:
        text_embeds, image_embeds, type_is_img = out_features
        text_embeds = self._normalize_cpu(text_embeds, normalize=not self.is_colipali)
        image_embeds = self._normalize_cpu(image_embeds, normalize=not self.is_colipali)

        embeddings = list(next(image_embeds if is_img else text_embeds) for is_img in type_is_img)
        return embeddings

    def tokenize_lengths(self, text_list: list[str]) -> list[int]:
        if self.is_colipali:
            preprocessed = self.processor(
                text=text_list,
                images=[self.mock_image] * len(text_list),
                truncation=True,
                max_length=self.max_length,
            )
            return [len(t) for t in preprocessed["input_ids"]]
        else:
            preprocessed = self.processor(
                text=text_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            return [len(t) for t in preprocessed["input_ids"]]
