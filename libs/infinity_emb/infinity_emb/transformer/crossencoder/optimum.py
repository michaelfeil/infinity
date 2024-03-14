import copy
import os
from typing import Dict, List, Tuple

import numpy as np

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.utils_optimum import (
    device_to_onnx,
    get_onnx_files,
    optimize_model,
)

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore

    OPTIMUM_AVAILABLE = True
except (ImportError, RuntimeError):
    OPTIMUM_AVAILABLE = False


class OptimumCrossEncoder(BaseCrossEncoder):
    def __init__(self, *, engine_args: EngineArgs):
        if not OPTIMUM_AVAILABLE:
            raise ImportError(
                "optimum.onnxruntime is not installed."
                "`pip install optimum[onnxruntime]`"
            )
        provider = device_to_onnx(engine_args.device)

        onnx_file = get_onnx_files(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            use_auth_token=True,
            prefer_quantized="cpu" in provider.lower(),
        )

        self.model = optimize_model(
            engine_args.model_name_or_path,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not os.environ.get("INFINITY_ONNX_DISABLE_OPTIMIZE", False),
            model_class=ORTModelForSequenceClassification,
        )
        self.model.use_io_binding = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self.config = AutoConfig.from_pretrained(
            engine_args.model_name_or_path,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

    def encode_pre(self, queries_docs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        encoded = self.tokenizer(
            queries_docs,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation="longest_first",
            return_tensors="np",
        )
        # Windows requires int64
        encoded = {k: v.astype(np.int64) for k, v in encoded.items()}
        return encoded

    def encode_core(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.model(**features, return_dict=True)

        return outputs.logits

    def encode_post(self, out_features: np.ndarray) -> List[float]:
        return out_features.flatten().astype(np.float32).tolist()

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences, padding=False, truncation=True
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
