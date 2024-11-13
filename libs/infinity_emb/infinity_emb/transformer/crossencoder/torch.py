# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_SENTENCE_TRANSFORMERS, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.quantization.interface import (
    quant_interface,
)

if CHECK_TORCH.is_available and CHECK_SENTENCE_TRANSFORMERS.is_available:
    import torch
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
else:

    class CrossEncoder:  # type: ignore[no-redef]
        pass


if TYPE_CHECKING:
    from torch import Tensor


from infinity_emb.transformer.acceleration import to_bettertransformer

__all__ = [
    "CrossEncoderPatched",
]


class CrossEncoderPatched(CrossEncoder, BaseCrossEncoder):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args: EngineArgs):
        CHECK_SENTENCE_TRANSFORMERS.mark_required()

        model_kwargs = {}
        if engine_args.bettertransformer:
            model_kwargs["attn_implementation"] = "eager"

        ls = engine_args._loading_strategy
        assert ls is not None

        if ls.loading_dtype is not None:  # type: ignore
            model_kwargs["torch_dtype"] = ls.loading_dtype

        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=ls.device_placement,
            automodel_args=model_kwargs,
        )
        self.model.to(ls.device_placement)

        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.model.eval()  # type: ignore

        self.model = to_bettertransformer(
            self.model,  # type: ignore
            engine_args,
            logger,
        )

        self.model.to(ls.loading_dtype)

        if ls.quantization_dtype is not None:
            self.model = quant_interface(  # TODO: add ls.quantization_dtype and ls.placement
                self.model, engine_args.dtype, device=Device[self.model.device.type]
            )

        if engine_args.compile:
            logger.info("using torch.compile(dynamic=True)")
            self.model = torch.compile(self.model, dynamic=True)

    def encode_pre(self, input_tuples: list[tuple[str, str]]):
        # return input_tuples
        texts = [[t[0].strip(), t[1].strip()] for t in input_tuples]

        tokenized = self.tokenizer(
            texts, padding=True, truncation="longest_first", return_tensors="pt"
        )
        return tokenized

    def encode_core(self, features: dict[str, "Tensor"]):
        """
        Computes sentence embeddings
        """
        with torch.no_grad():
            features = {k: v.to(self.model.device) for k, v in features.items()}
            out_features = self.model(**features, return_dict=True)["logits"]

        return out_features.detach().cpu()

    def encode_post(self, out_features) -> list[float]:
        return out_features.flatten().to(torch.float32).numpy()

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]
