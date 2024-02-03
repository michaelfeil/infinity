import copy
import os
from typing import Dict, List

import numpy as np

from infinity_emb.primitives import EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.utils_optimum import (
    device_to_onnx,
    get_onnx_files,
    optimize_model,
)

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


class OptimumCrossEncoder(BaseEmbedder):
    def __init__(self, model_name_or_path, **kwargs):
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum.onnxruntime is not installed."
                "`pip install optimum[onnxruntime]`"
            )
        provider = device_to_onnx(kwargs.get("device"))

        onnx_file = get_onnx_files(
            model_name_or_path,
            None,
            use_auth_token=True,
            prefer_quantized="cpu" in provider.lower(),
        )

        self.model = optimize_model(
            model_name_or_path,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not os.environ.get("INFINITY_ONNX_DISABLE_OPTIMIZE", False),
            model_class=ORTModelForSequenceClassification,
        )
        self.model.use_io_binding = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

    def encode_pre(self, input_tuples: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            input_tuples,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation="longest_first",
            return_tensors="np",
        )
        return encoded

    def encode_core(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.model(**features, return_dict=True)

        return outputs.logits

    def encode_post(self, out_features: np.ndarray) -> EmbeddingReturnType:
        return out_features.flatten().astype(np.float32)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences, padding=False, truncation=True
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
