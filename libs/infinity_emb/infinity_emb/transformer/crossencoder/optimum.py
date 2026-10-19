# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import copy

import numpy as np

from infinity_emb._optional_imports import CHECK_ONNXRUNTIME
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import RerankLimits
from infinity_emb.transformer.abstract import BaseCrossEncoder
from infinity_emb.transformer.crossencoder import truncate_texts_to_tokens
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
    except (ImportError, RuntimeError, Exception) as ex:
        CHECK_ONNXRUNTIME.mark_dirty(ex)


class OptimumCrossEncoder(BaseCrossEncoder):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_ONNXRUNTIME.mark_required()
        provider = device_to_onnx(engine_args.device)

        onnx_file = get_onnx_files(
            model_name_or_path=engine_args.model_name_or_path,
            revision=engine_args.revision,
            use_auth_token=True,
            prefer_quantized=("cpu" in provider.lower() or "openvino" in provider.lower()) and not engine_args.onnx_do_not_prefer_quantized,
        )

        self.model = optimize_model(
            engine_args.model_name_or_path,
            execution_provider=provider,
            file_name=onnx_file.as_posix(),
            optimize_model=not engine_args.onnx_disable_optimize,
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

    def encode_pre(
        self, queries_docs: list[tuple[str, str, RerankLimits]]
    ) -> dict[str, np.ndarray]:
        queries = [t[0] for t in queries_docs]
        documents = [t[1] for t in queries_docs]
        limits = [t[2] if len(t) > 2 else RerankLimits() for t in queries_docs]

        model_max = (
            getattr(self.config, "max_position_embeddings", None)
            or self.tokenizer.model_max_length
        )

        def pair_max_length(limit: RerankLimits) -> int:
            if (limit.max_pair_tokens or 0) > 0:
                return min(limit.max_pair_tokens, model_max)
            return model_max

        # 1) head-truncate the query and the document independently, then
        # 2) cap the joined pair (longest side trimmed first) to max_pair_tokens.
        queries = truncate_texts_to_tokens(
            self.tokenizer, queries, [lim.max_query_tokens for lim in limits]
        )
        documents = truncate_texts_to_tokens(
            self.tokenizer, documents, [lim.max_tokens_per_doc for lim in limits]
        )
        encodings = [
            self.tokenizer(
                q,
                d,
                max_length=pair_max_length(lim),
                truncation="longest_first",
                return_token_type_ids=False,
            )
            for q, d, lim in zip(queries, documents, limits)
        ]
        encoded = self.tokenizer.pad(encodings, padding=True, return_tensors="np")
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
            tks = self._infinity_tokenizer.encode_batch(sentences, padding=False, truncation=True)
        else:
            tks = self._infinity_tokenizer(sentences, padding=False, truncation=True)

        return [len(t) for t in tks["input_ids"]]
