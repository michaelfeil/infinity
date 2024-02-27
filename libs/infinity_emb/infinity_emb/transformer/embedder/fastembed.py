import copy
from typing import Dict, List

import numpy as np

from infinity_emb.log_handler import logger
from infinity_emb.primitives import EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder

try:
    from fastembed.embedding import TextEmbedding  # type: ignore

    from infinity_emb.transformer.utils_optimum import normalize

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


class Fastembed(BaseEmbedder):
    def __init__(self, model_name_or_path, **kwargs):
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is not installed." "`pip install infinity-emb[fastembed]`"
            )
        logger.warning(
            "deprecated: fastembed inference"
            " is deprecated and will be removed in the future."
        )

        providers = ["CPUExecutionProvider"]

        if not kwargs.get("cache_dir"):
            from infinity_emb.transformer.utils import infinity_cache_dir

            kwargs["cache_dir"] = infinity_cache_dir()
        if kwargs.pop("device", None) != "cpu":
            providers = ["CUDAExecutionProvider"] + providers
        kwargs.pop("trust_remote_code", None)
        if kwargs.pop("revision", None) is not None:
            logger.warning("revision is not used for CrossEncoder")

        self.model = TextEmbedding(
            model_name_or_path,
            **kwargs,
        ).model
        self._infinity_tokenizer = copy.deepcopy(self.model.tokenizer)
        self.model.model.set_providers(providers)

    def encode_pre(self, sentences: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.model.tokenizer.encode_batch(sentences)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
            "token_type_ids": np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            ),
        }
        return onnx_input

    def encode_core(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        model_output = self.model.model.run(None, features)
        last_hidden_state = model_output[0][:, 0]
        return last_hidden_state

    def encode_post(self, embedding: np.ndarray) -> EmbeddingReturnType:
        return normalize(embedding).astype(np.float32)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        tks = self._infinity_tokenizer.encode_batch(
            sentences,
        )
        return [len(t.tokens) for t in tks]
