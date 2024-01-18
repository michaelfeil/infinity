import copy
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from infinity_emb.log_handler import logger
from infinity_emb.primitives import EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder

try:
    # Cache dir
    from huggingface_hub import HfApi, HfFolder  # type: ignore
    from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore
    from optimum.onnxruntime import (  # type: ignore
        ORTModelForFeatureExtraction,
        ORTOptimizer,
    )
    from optimum.onnxruntime.configuration import OptimizationConfig  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False


def mean_pooling(last_hidden_states: np.ndarray, attention_mask: np.ndarray):
    input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

    sum_embeddings = np.sum(last_hidden_states * input_mask_expanded, axis=1)
    mask_sum = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)

    return sum_embeddings / mask_sum


def normalize(input_array: np.ndarray, p=2, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


def device_to_onnx(device) -> Union[str, List[str]]:
    if device == "cpu":
        return "CPUExecutionProvider"
    elif device == "cuda":
        return "CudaExecutionProvider"
    elif device == "mps":
        return "CoreMLExecutionProvider"
    elif device is None:
        raise ValueError(
            "for the optimum engine, the `device=auto` "
            "is not supported yet, select `cpu` or `cuda` explicitly"
        )
    else:
        raise ValueError(f"Unknown device {device}")


def optimize_model(
    model_name_or_path,
    execution_provider: List[str],
    file_name: str,
    optimize_model=False,
    model_class=None,
):
    if model_class is None:
        model_class = ORTModelForFeatureExtraction

    unoptimized_model_path = model_class.from_pretrained(
        model_name_or_path, provider=execution_provider, file_name=file_name
    )
    if not optimize_model:
        return unoptimized_model_path

    logger.info("Optimizing model")
    optimizer = ORTOptimizer.from_pretrained(unoptimized_model_path)

    first_execution_provider = (
        execution_provider
        if isinstance(execution_provider, str)
        else execution_provider[0]
    )

    optimization_config = OptimizationConfig(
        optimization_level=99,
        optimize_with_onnxruntime_only=False,
        optimize_for_gpu="cuda" in first_execution_provider.lower(),
    )

    path_folder = (
        Path(model_name_or_path)
        if Path(model_name_or_path).exists()
        else Path(HF_HUB_CACHE) / "infinity_onnx" / model_name_or_path
    )

    optimized_model_path = optimizer.optimize(
        optimization_config=optimization_config,
        save_dir=path_folder.as_posix(),
        # if larger than 2gb use external data format
        use_external_data_format=False,
        one_external_file=True,
    )

    model = model_class.from_pretrained(
        optimized_model_path,
        provider=execution_provider,
        file_name=Path(file_name).name.replace(".onnx", "_optimized.onnx"),
    )

    return model


def get_onnx_files(
    model_id: str, revision: str, use_auth_token: Union[bool, str] = True
):
    """gets the onnx files from the repo"""
    if isinstance(use_auth_token, bool):
        token = HfFolder().get_token()
    else:
        token = use_auth_token
    repo_files = map(
        Path, HfApi().list_repo_files(model_id, revision=revision, token=token)
    )
    pattern = "**.onnx"
    return [p for p in repo_files if p.match(pattern)]


class OptimumEmbedder(BaseEmbedder):
    def __init__(self, model_name_or_path, **kwargs):
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum.onnxruntime is not installed."
                "`pip install optimum[onnxruntime]`"
            )
        providers = device_to_onnx(kwargs.get("device", "auto"))

        onnx_files = get_onnx_files(model_name_or_path, None, use_auth_token=True)
        if len(onnx_files) >= 0:
            logger.info(
                f"Found {len(onnx_files)} onnx files: "
                f"{onnx_files}, selecting {onnx_files[-1]}"
            )
        self.model = optimize_model(
            model_name_or_path,
            execution_provider=providers,
            file_name=onnx_files[-1].as_posix(),
            optimize_model=not os.environ.get("INFINITY_ONNX_DISABLE_OPTIMIZE", False),
        )
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

    def encode_core(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.model(**features)
        embeddings = mean_pooling(
            outputs["last_hidden_state"], features["attention_mask"]
        )
        return embeddings

    def encode_post(self, embedding: np.ndarray) -> EmbeddingReturnType:
        return normalize(embedding).astype(np.float32)

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences, padding=False, truncation=True
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
