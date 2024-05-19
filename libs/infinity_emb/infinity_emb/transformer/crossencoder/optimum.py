import copy
import os

import numpy as np

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME
from infinity_emb.args import EngineArgs
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.utils_optimum import (
    device_to_onnx,
    get_onnx_files,
    optimize_model,
)

if CHECK_ONNXRUNTIME.is_available:
    try:
        from optimum.onnxruntime import (  # type: ignore
            ORTModelForSequenceClassification,
        )
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
    except (ImportError, RuntimeError) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)


class OptimumCrossEncoder(BaseCrossEncoder):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_ONNXRUNTIME.mark_required()
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
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
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

    def encode_pre(self, queries_docs: list[tuple[str, str]]) -> dict[str, np.ndarray]:
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

    def encode_core(self, features: dict[str, np.ndarray]) -> np.ndarray:
        outputs = self.model(**features, return_dict=True)

        return outputs.logits

    def encode_post(self, out_features: np.ndarray) -> list[float]:
        return out_features.flatten().astype(np.float32).tolist()

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        if hasattr(self._infinity_tokenizer, "encode_batch"):
            tks = self._infinity_tokenizer.encode_batch(
                sentences, padding=False, truncation=True
            )
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
