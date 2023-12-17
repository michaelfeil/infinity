import copy
import os
from typing import List

from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseCrossEncoder

try:
    import torch
    from sentence_transformers import CrossEncoder
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    torch, Tensor = None, None

    class CrossEncoder:
        pass

    class Module:
        pass

    TORCH_AVAILABLE = False

try:
    from optimum.bettertransformer import BetterTransformer

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

__all__ = [
    "CrossEncoderPatched",
]


class CrossEncoderPatched(CrossEncoder, BaseCrossEncoder):
    """CrossEncoder with .encode_core() and no microbatching"""

    def __init__(self, *args, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed."
                " `pip install infinity-emb[torch]` "
                "or pip install infinity-emb[torch,optimum]`"
            )
        super().__init__(*args, **kwargs)

        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.

        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)
        self.model.eval()

        if OPTIMUM_AVAILABLE and not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
            logger.info(
                "Adding optimizations via Huggingface optimum. "
                "Disable by setting the env var `INFINITY_DISABLE_OPTIMUM`"
            )
            try:
                self.model = BetterTransformer.transform(self.model)
            except Exception as ex:
                logger.exception(f"BetterTransformer failed with {ex}")
                exit(1)
        elif not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
            logger.info(
                "No optimizations via Huggingface optimum,"
                " it is disabled via env INFINITY_DISABLE_OPTIMUM "
            )
        else:
            logger.info(
                "No optimizations via Huggingface optimum, "
                "install `pip install infinity-emb[optimum]`"
            )

        if self._target_device.type == "cuda" and os.environ.get(
            "INFINITY_TORCH_ENABLE_HALF", False
        ):
            logger.info(
                "Switching to half() precision (fp16). "
                "Enabled by the setting the env var `INFINITY_TORCH_ENABLE_HALF`"
            )
            self.half()

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
