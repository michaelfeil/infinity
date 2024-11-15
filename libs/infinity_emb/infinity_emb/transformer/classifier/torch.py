# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from infinity_emb._optional_imports import CHECK_TRANSFORMERS, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseClassifer
from infinity_emb.transformer.acceleration import to_bettertransformer
from infinity_emb.transformer.quantization.interface import quant_interface
from infinity_emb.primitives import Device

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoTokenizer, pipeline  # type: ignore
if CHECK_TORCH.is_available:
    import torch


class SentenceClassifier(BaseClassifer):
    def __init__(
        self,
        *,
        engine_args: EngineArgs,
    ) -> None:
        CHECK_TRANSFORMERS.mark_required()
        model_kwargs = {}
        if engine_args.bettertransformer:
            model_kwargs["attn_implementation"] = "eager"
        ls = engine_args._loading_strategy
        assert ls is not None

        if ls.loading_dtype is not None:  # type: ignore
            model_kwargs["torch_dtype"] = ls.loading_dtype

        self._pipe = pipeline(
            task="text-classification",
            model=engine_args.model_name_or_path,
            trust_remote_code=engine_args.trust_remote_code,
            device=ls.device_placement,
            top_k=None,
            revision=engine_args.revision,
            model_kwargs=model_kwargs,
        )

        self._pipe.model = to_bettertransformer(
            self._pipe.model,
            engine_args,
            logger,
        )

        if ls.quantization_dtype is not None:
            self._pipe.model = quant_interface(  # TODO: add ls.quantization_dtype and ls.placement
                self._pipe.model, engine_args.dtype, device=Device[self._pipe.model.device.type]
            )

        if engine_args.compile:
            logger.info("using torch.compile(dynamic=True)")
            self._pipe.model = torch.compile(self._pipe.model, dynamic=True)

        self._infinity_tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )

    def encode_pre(self, sentences: list[str]):
        """takes care of the tokenization and feature preparation"""
        return sentences

    def encode_core(self, features):
        """runs plain inference, on cpu/gpu"""
        return self._pipe(features, batch_size=256, truncation=True, padding=True)

    def encode_post(self, classes) -> dict[str, float]:
        """runs post encoding such as normalization"""
        return classes

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
        ).encodings
        return [len(t.tokens) for t in tks]
