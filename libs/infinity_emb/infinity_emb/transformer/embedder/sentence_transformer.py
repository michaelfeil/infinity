# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from infinity_emb._optional_imports import (
    CHECK_SENTENCE_TRANSFORMERS,
    CHECK_TORCH,
)
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.acceleration import to_bettertransformer
from infinity_emb.transformer.quantization.interface import (
    quant_embedding_decorator,
    quant_interface,
)

if TYPE_CHECKING:
    from torch import Tensor
    from infinity_emb.primitives import EmbeddingReturnType


if CHECK_SENTENCE_TRANSFORMERS.is_available:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
else:

    class SentenceTransformer:  # type: ignore[no-redef]
        pass


if CHECK_TORCH.is_available:
    import torch
    import torch._dynamo.config
    import torch._inductor.config

    # torch._inductor.config.coordinate_descent_tuning = True
    try:
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
    except Exception:
        pass


class SentenceTransformerPatched(SentenceTransformer, BaseEmbedder):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args=EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_SENTENCE_TRANSFORMERS.mark_required()

        model_kwargs = {}
        if engine_args.bettertransformer:
            model_kwargs["attn_implementation"] = "eager"

        ls = engine_args._loading_strategy
        assert ls is not None

        if ls.loading_dtype is not None:
            model_kwargs["torch_dtype"] = ls.loading_dtype

        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=ls.device_placement,
            model_kwargs=model_kwargs,
        )
        self.to(ls.device_placement)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        fm = self._first_module()

        self.normalize_embeddings = True

        self.mode_colbert = False
        if "colbert" in fm.auto_model.config.architectures[0].lower():
            self.mode_colbert = True
            self.normalize_embeddings = False

        self._infinity_tokenizer = copy.deepcopy(fm.tokenizer)
        self.eval()
        self.engine_args = engine_args

        fm.auto_model = to_bettertransformer(
            fm.auto_model,
            engine_args,
            logger,
        )

        fm.to(ls.loading_dtype)

        if ls.quantization_dtype is not None:
            fm.auto_model = quant_interface(  # TODO: add ls.quantization_dtype and ls.placement
                fm.auto_model, engine_args.dtype, device=Device[self.device.type]
            )

        if engine_args.compile:
            logger.info("using torch.compile(dynamic=True)")
            fm.auto_model = torch.compile(fm.auto_model, dynamic=True)

    def encode_pre(self, sentences) -> dict[str, "Tensor"]:
        features = self.tokenize(sentences)
        return features

    def encode_core(self, features: dict[str, "Tensor"]) -> "Tensor":
        """
        Computes sentence embeddings
        """

        with torch.no_grad():
            features = util.batch_to_device(features, self.device)  # type: ignore
            out: dict[str, "Tensor"] = self.forward(features)
            if not self.mode_colbert:
                out_features = out["sentence_embedding"].detach().cpu()
            else:
                out_features = {  # type: ignore # noqa
                    "token_embeddings": out["token_embeddings"].detach().cpu(),
                    "attention_mask": out["attention_mask"].detach().cpu(),
                }

        return out_features

    @quant_embedding_decorator()
    def encode_post(
        self,
        out_features: "Tensor",
    ) -> "EmbeddingReturnType":
        with torch.inference_mode():
            if not self.mode_colbert:
                embeddings: "Tensor" = out_features.to(torch.float32)
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_np: np.ndarray = embeddings.numpy()
            else:
                # remove the attention mask for two inputs with 5 and 3 tokens that's [[1,1,1,1,1],[1,1,1,0,0]]
                # and convert to list of numpy arrays
                embeddings_np = [  # type: ignore # noqa
                    z[m].numpy()
                    for z, m in zip(
                        out_features["token_embeddings"].to(torch.float32),  # type: ignore
                        out_features["attention_mask"].bool(),  # type: ignore
                    )
                ]

        return embeddings_np

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
