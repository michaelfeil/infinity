import copy
import json
import subprocess
from typing import Union

import numpy as np

from infinity_emb._optional_imports import CHECK_OPTIMUM_NEURON, CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.primitives import EmbeddingReturnType, PoolingMethod
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.utils_optimum import (
    cls_token_pooling,
    mean_pooling,
    normalize,
)

if CHECK_OPTIMUM_NEURON.is_available and CHECK_TORCH.is_available:
    import torch
    from optimum.neuron import NeuronModelForFeatureExtraction  # type: ignore
    from transformers import AutoConfig, AutoTokenizer  # type: ignore[import-untyped]


__all__ = [
    "NeuronOptimumEmbedder",
]


def get_nc_count() -> Union[int, None]:
    """Returns the number of neuron cores on the current instance."""
    try:
        cmd = "neuron-ls --json-output"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        print("inferring nc_count from `neuron-ls`")
        print(result.stdout.decode("utf-8"))
        json_output = json.loads(result.stdout)
        count = sum([x["nc_count"] for x in json_output])
        print(f"nc_count={count}")
        return count
    except Exception:
        return None


def pad_up_to_size(desired_max_bs, input_ids):
    """input_ids a 2D array with batch_size on dim=0

    makes sure the func runs with self.batch_size
    """
    # access a from TestSample
    batch_size = input_ids.shape[0]

    if batch_size < desired_max_bs:
        # handle the event of input_ids.shape[0] != batch_size
        # Neuron cores expect constant batch_size
        input_ids = torch.concat(
            (
                input_ids,
                # add missing_batch_size dummy
                torch.zeros(
                    [desired_max_bs - batch_size, *input_ids.size()[1:]],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ),
            ),
            dim=0,
        )
    elif batch_size > desired_max_bs:
        raise ValueError(
            f"The specified batch_size ({batch_size}) exceeds the model static batch size ({desired_max_bs})"
        )
    # return the forward pass that requires constant batch size
    return input_ids


class NeuronOptimumEmbedder(BaseEmbedder):
    def __init__(self, *, engine_args: EngineArgs):
        CHECK_OPTIMUM_NEURON.mark_required()

        self.pooling = (
            mean_pooling
            if engine_args.pooling_method == PoolingMethod.mean
            else cls_token_pooling
        )

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

        compiler_args = {"num_cores": get_nc_count(), "auto_cast_type": "fp16"}
        input_shapes = {
            "batch_size": 4,
            "sequence_length": (
                self.config.max_position_embeddings
                if hasattr(self.config, "max_position_embeddings")
                else 512
            ),
        }
        self.model = NeuronModelForFeatureExtraction.from_pretrained(
            model_id=engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            export=True,
            **compiler_args,
            **input_shapes,
        )
        self.batch_size = self.model.neuron_config.input_shapes["batch_size"]

    def encode_pre(self, sentences: list[str]) -> dict[str, np.ndarray]:
        input_dict = self.tokenizer(
            sentences,
            max_length=self.config.max_position_embeddings,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
        )
        input_dict.pop("token_type_ids", None)
        return input_dict

    def encode_core(self, input_dict: dict[str, np.ndarray]) -> dict:
        """requires constant batch size, which is a bit of extra work"""
        for key, tensor in input_dict.items():
            actual_bsize = tensor.shape[0]
            input_dict[key] = pad_up_to_size(self.batch_size, tensor)
        with torch.inference_mode():
            outputs = self.model(**input_dict)
        return {
            "token_embeddings": outputs["last_hidden_state"][:actual_bsize],
            "attention_mask": input_dict["attention_mask"][:actual_bsize],
        }

    def encode_post(self, embedding: dict) -> EmbeddingReturnType:
        embedding = self.pooling(  # type: ignore
            embedding["token_embeddings"].numpy(), embedding["attention_mask"].numpy()
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
