import copy
from typing import Dict, List

import numpy as np

from infinity_emb.inference.primitives import NpEmbeddingType
from infinity_emb.transformer.abstract import BaseTransformer

try:
    from fastembed.embedding import DefaultEmbedding, normalize

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

    class DefaultEmbedding:
        def __init__(self, *args, **kwargs) -> None:
            pass


class Fastembed(DefaultEmbedding, BaseTransformer):
    def __init__(self, *args, **kwargs):
        if not FASTEMBED_AVAILABLE:
            raise ImportError(
                "fastembed is not installed." "`pip install infinity-emb[fastembed]`"
            )
        if not kwargs.get("cache_dir"):
            from infinity_emb.transformer.utils import infinity_cache_dir

            kwargs["cache_dir"] = infinity_cache_dir()
        super(DefaultEmbedding, self).__init__(*args, **kwargs)
        self._infinity_tokenizer = copy.deepcopy(self.model.tokenizer)
        self.model.model.set_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def encode_pre(self, sentences: List[str]) -> Dict[str, np.ndarray[int]]:
        encoded = self.model.tokenizer.encode_batch(sentences)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        }

        if not self.model.exclude_token_type_ids:
            onnx_input["token_type_ids"] = np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            )
        return onnx_input

    def encode_core(self, features: Dict[str, np.ndarray[int]]) -> np.ndarray:
        model_output = self.model.model.run(None, features)
        last_hidden_state = model_output[0][:, 0]
        return last_hidden_state

    def encode_post(self, embedding: np.ndarray) -> NpEmbeddingType:
        return normalize(embedding).astype(np.float32)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        # tks = self._infinity_tokenizer.encode_batch(
        #     sentences,
        # )
        # return [len(t.tokens) for t in tks]
        return [len(s) for s in sentences]
