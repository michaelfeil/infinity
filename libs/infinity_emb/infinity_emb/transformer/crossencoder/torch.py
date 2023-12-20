import copy
import os
from typing import List

from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseCrossEncoder

try:
    import torch
    from sentence_transformers import CrossEncoder  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

    class CrossEncoder:  # type: ignore
        pass

    class Module:  # type: ignore
        pass

    TORCH_AVAILABLE = False

from infinity_emb.transformer.acceleration import to_bettertransformer

__all__ = [
    "CrossEncoderPatched",
]


class CrossEncoderPatched(CrossEncoder, BaseCrossEncoder):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, model_name_or_path, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed."
                " `pip install infinity-emb[torch]` "
                "or pip install infinity-emb[torch,optimum]`"
            )
        super().__init__(model_name_or_path, **kwargs)

        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.model.eval()

        self.model = to_bettertransformer(self.model, logger)

        if self._target_device.type == "cuda" and not os.environ.get(
            "INFINITY_DISABLE_HALF", ""
        ):
            logger.info(
                "Switching to half() precision (cuda: fp16). "
                "Disable by the setting the env var `INFINITY_DISABLE_HALF`"
            )
            self.model.to(dtype=torch.float16)

    def encode_pre(self, input_tuples):
        # TODO: improve
        return input_tuples

    def encode_core(self, features):
        """
        Computes sentence embeddings
        """
        with torch.inference_mode():
            out_features = self.predict(
                features,
                batch_size=32,
                activation_fct=lambda x: x,
                show_progress_bar=False,
            )

        return out_features

    def encode_post(self, out_features) -> List[float]:
        return out_features

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            # truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]
