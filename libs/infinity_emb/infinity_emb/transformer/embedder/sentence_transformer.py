from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Mapping

import numpy as np

from infinity_emb._optional_imports import (
    CHECK_SENTENCE_TRANSFORMERS,
    CHECK_TORCH,
)
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, Dtype, EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.acceleration import to_bettertransformer
from infinity_emb.transformer.quantization.interface import quant_interface

if TYPE_CHECKING:
    from torch import Tensor


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
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True


class SentenceTransformerPatched(SentenceTransformer, BaseEmbedder):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args=EngineArgs):
        CHECK_TORCH.mark_required()
        CHECK_SENTENCE_TRANSFORMERS.mark_required()
        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=engine_args.device.resolve(),
        )
        self.to(self.device)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        fm = self._first_module()
        self._infinity_tokenizer = copy.deepcopy(fm.tokenizer)
        self.eval()

        self.embedding_dtype = engine_args.embedding_dtype

        if not (self.device.type == "mps" or not engine_args.bettertransformer):
            fm.auto_model = to_bettertransformer(
                fm.auto_model,
                logger,
            )

        if self.device.type == "cuda" and engine_args.dtype in [
            Dtype.auto,
            Dtype.float16,
        ]:
            logger.info("Switching to half() precision (cuda: fp16). ")
            self.half()

        if engine_args.dtype in (Dtype.int8, Dtype.fp8):
            fm.auto_model = quant_interface(
                fm.auto_model, engine_args.dtype, device=Device[self.device.type]
            )

        if engine_args.compile:
            logger.info("using torch.compile()")
            fm.auto_model = torch.compile(fm.auto_model, dynamic=True)

    def encode_pre(self, sentences) -> Mapping[str, "Tensor"]:
        features = self.tokenize(sentences)

        return features

    def encode_core(self, features: Mapping[str, "Tensor"]) -> "Tensor":
        """
        Computes sentence embeddings
        """

        with torch.no_grad():
            features = util.batch_to_device(features, self.device)
            out_features: "Tensor" = self.forward(features)["sentence_embedding"]

        return out_features.detach().cpu()

    def encode_post(
        self, out_features: "Tensor", normalize_embeddings: bool = True
    ) -> EmbeddingReturnType:
        with torch.inference_mode():
            embeddings: "Tensor" = out_features.to(torch.float32)
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings_np: np.ndarray = embeddings.numpy()

            if self.embedding_dtype.value != "float32":
                raise NotImplementedError(
                    f"EmbeddingDtype for {self.embedding_dtype} not implemented"
                )

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
