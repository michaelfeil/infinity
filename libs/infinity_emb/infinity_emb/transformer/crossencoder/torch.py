from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from infinity_emb._optional_imports import CHECK_SENTENCE_TRANSFORMERS, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Dtype
from infinity_emb.transformer.abstract import BaseCrossEncoder

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

        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            device=engine_args.device.resolve(),
            trust_remote_code=engine_args.trust_remote_code,
        )
        self.model.to(self._target_device)  # type: ignore

        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.model.eval()  # type: ignore

        if not (self._target_device.type == "mps" or not engine_args.bettertransformer):
            self.model = to_bettertransformer(
                self.model,  # type: ignore
                logger,
            )

        if self._target_device.type == "cuda" and engine_args.dtype in [
            Dtype.auto,
            Dtype.float16,
        ]:
            logger.info(
                "Switching to half() precision (cuda: fp16). "
                "Disable by the setting the env var `INFINITY_DISABLE_HALF`"
            )
            self.model.to(dtype=torch.float16)

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
        return out_features.flatten()

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
