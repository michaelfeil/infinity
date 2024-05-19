import copy
import os

import numpy as np

from infinity_emb.args import EngineArgs
from infinity_emb.primitives import EmbeddingReturnType, PoolingMethod
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.utils_optimum import (
    cls_token_pooling,
    device_to_onnx,
    get_onnx_files,
    mean_pooling,
    normalize,
    optimize_model,
)

try:
    from optimum.onnxruntime import (  # type: ignore[import-untyped]
        ORTModelForFeatureExtraction,
    )
    from transformers import AutoConfig, AutoTokenizer  # type: ignore[import-untyped]

    OPTIMUM_AVAILABLE = True
except (ImportError, RuntimeError):
    OPTIMUM_AVAILABLE = False


class OptimumEmbedder(BaseEmbedder):
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

        self.pooling = (
            mean_pooling
            if engine_args.pooling_method == PoolingMethod.mean
            else cls_token_pooling
        )

        self.model = optimize_model(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not os.environ.get("INFINITY_ONNX_DISABLE_OPTIMIZE", False),
            model_class=ORTModelForFeatureExtraction,
        )
        self.model.use_io_binding = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self.config = AutoConfig.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )
        self._infinity_tokenizer = copy.deepcopy(self.tokenizer)

    def encode_pre(self, sentences: list[str]) -> dict[str, np.ndarray]:
        encoded = self.tokenizer(
            sentences,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation="longest_first",
            return_tensors="np",
        )
        # int64 is required for onnxruntime on Windows
        encoded = {k: v.astype(np.int64) for k, v in encoded.items()}
        return encoded

    def encode_core(self, onnx_input: dict[str, np.ndarray]) -> dict:
        outputs = self.model(**onnx_input)
        return {
            "token_embeddings": outputs["last_hidden_state"],
            "attention_mask": onnx_input["attention_mask"],
        }

    def encode_post(self, embedding: dict) -> EmbeddingReturnType:
        embedding = self.pooling(  # type: ignore
            embedding["token_embeddings"], embedding["attention_mask"]
        )

        return normalize(embedding).astype(np.float32)

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences,
                padding=False,
                truncation="longest_first",
            )
        else:
            tks = self._infinity_tokenizer(
                sentences, padding=False, truncation="longest_first"
            )

        return [len(t) for t in tks["input_ids"]]
