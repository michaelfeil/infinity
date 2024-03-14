import copy
from typing import Dict, List

import numpy as np

from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device, EmbeddingReturnType, PoolingMethod
from infinity_emb.transformer.abstract import BaseEmbedder

try:
    from fastembed.embedding import TextEmbedding  # type: ignore

    from infinity_emb.transformer.utils_optimum import normalize

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


class Fastembed(BaseEmbedder):
    def __init__(self, *, engine_args: EngineArgs) -> None:
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is not installed." "`pip install infinity-emb[fastembed]`"
            )
        logger.warning(
            "deprecated: fastembed inference"
            " is deprecated and will be removed in the future."
        )

        providers = ["CPUExecutionProvider"]

        if engine_args.device != Device.cpu:
            providers = ["CUDAExecutionProvider"] + providers

        if engine_args.revision is not None:
            logger.warning("revision is not used for fastembed")

        self.model = TextEmbedding(
            model_name=engine_args.model_name_or_path, cache_dir=None
        ).model
        if self.model is None:
            raise ValueError("fastembed model is not available")
        if engine_args.pooling_method != PoolingMethod.auto:
            logger.warning("pooling_method is not used for fastembed")
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
