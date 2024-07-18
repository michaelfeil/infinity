from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from infinity_emb._optional_imports import CHECK_TORCH, CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Dtype
from infinity_emb.transformer.abstract import BaseClipVisionModel
from infinity_emb.transformer.quantization.interface import quant_embedding_decorator

if TYPE_CHECKING:
    from PIL.Image import Image as ImageClass
    from torch import Tensor

if CHECK_TORCH.is_available:
    import torch
if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoModel, AutoProcessor  # type: ignore


class ClipLikeModel(BaseClipVisionModel):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_TRANSFORMERS.mark_required()
        self.model = AutoModel.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            # attn_implementation="eager" if engine_args.bettertransformer else None,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if engine_args.dtype in (Dtype.float16, Dtype.auto):
                self.model = self.model.half()
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
            self.model.vision_model = torch.compile(
                self.model.vision_model, dynamic=True
            )
            self.model.text_model = torch.compile(self.model.text_model, dynamic=True)

        assert hasattr(
            self.model, "get_text_features"
        ), f"AutoModel of {engine_args.model_name_or_path} does not have get_text_features method"
        assert hasattr(
            self.model, "get_image_features"
        ), f"AutoModel of {engine_args.model_name_or_path} does not have get_image_features method"
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

        preprocessed = self.processor(
            images=image_list if image_list else None,
            text=text_list if text_list else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        preprocessed = {k: v.to(self.model.device) for k, v in preprocessed.items()}

        return (preprocessed, type_is_img)

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
        features, type_is_img = features_and_types
        with torch.no_grad():
            # TODO: torch.cuda.stream()
            if "input_ids" in features:
                text_embeds = self.model.get_text_features(
                    input_ids=features.get("input_ids"),
                    attention_mask=features.get("attention_mask"),
                )
            else:
                text_embeds = None  # type: ignore
            if "pixel_values" in features:
                image_embeds = self.model.get_image_features(
                    pixel_values=features.get("pixel_values"),
                )
            else:
                image_embeds = None

        return text_embeds, image_embeds, type_is_img

    @quant_embedding_decorator()
    def encode_post(self, out_features) -> list[float]:
        text_embeds, image_embeds, type_is_img = out_features
        text_embeds = self._normalize_cpu(text_embeds)
        image_embeds = self._normalize_cpu(image_embeds)
        embeddings = list(
            next(image_embeds if is_img else text_embeds) for is_img in type_is_img
        )
        return embeddings

    def tokenize_lengths(self, text_list: list[str]) -> list[int]:
        preprocessed = self.processor(
            text=text_list, truncation=True, max_length=self.max_length
        )
        return [len(t) for t in preprocessed["input_ids"]]
