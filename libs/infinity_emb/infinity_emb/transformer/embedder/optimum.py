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
    from optimum.onnxruntime import ORTModelForFeatureExtraction  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


def mean_pooling(last_hidden_states: np.ndarray, attention_mask: np.ndarray):
    input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

    sum_embeddings = np.sum(
        last_hidden_states.astype(float) * input_mask_expanded, axis=1
    )
    mask_sum = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)

    return sum_embeddings / mask_sum


def normalize(input_array: np.ndarray, p=2, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


class OptimumEmbedder(BaseEmbedder):
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
            model_class=ORTModelForFeatureExtraction,
        )
        self.model.use_io_binding = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

    def encode_pre(self, sentences: List[str]) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            sentences,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        return encoded

    def encode_core(self, features: Dict[str, np.ndarray]) -> tuple:
        outputs = self.model(**features)
        return (outputs["last_hidden_state"], features["attention_mask"])

    def encode_post(self, embedding: tuple) -> EmbeddingReturnType:
        embeddings = mean_pooling(embedding[0], embedding[1])
        return normalize(embeddings).astype(np.float32)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences, padding=False, truncation=True
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
