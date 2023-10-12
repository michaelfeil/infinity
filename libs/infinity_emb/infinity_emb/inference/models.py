import copy
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util  # type: ignore
from torch import Tensor

from infinity_emb.inference.primitives import NpEmbeddingType
from infinity_emb.log_handler import logger

__all__ = [
    "InferenceEngine",
    "InferenceEngineTypeHint",
    "DummyTransformer",
    "CT2SentenceTransformer",
    "BaseTransformer",
]

INPUT_FEATURE = Any
OUT_FEATURES = Any


class BaseTransformer(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod  # Decorator to define an abstract method
    def encode_pre(self, sentences: List[str]) -> INPUT_FEATURE:
        pass

    @abstractmethod
    def encode_core(self, features: INPUT_FEATURE) -> OUT_FEATURES:
        pass

    @abstractmethod
    def encode_post(self, embedding: OUT_FEATURES) -> NpEmbeddingType:
        pass

    @abstractmethod
    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        pass


class DummyTransformer(BaseTransformer):
    """fix-13 dimension embedding, filled with length of sentence"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def encode_pre(self, sentences: List[str]) -> np.ndarray:
        return np.asarray(sentences)

    def encode_core(self, features: np.ndarray) -> NpEmbeddingType:
        lengths = np.array([[len(s) for s in features]])
        # embedding of size 13
        return np.ones([len(features), 13]) * lengths.T

    def encode_post(self, embedding: NpEmbeddingType):
        return embedding

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        return [len(s) for s in sentences]


class SentenceTransformerPatched(SentenceTransformer, BaseTransformer):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self._target_device
        self.eval()
        self.to(device)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        self._infinity_tokenizer = copy.deepcopy(self._first_module().tokenizer)

    def encode_pre(self, sentences) -> Dict[str, Tensor]:
        features = self.tokenize(sentences)

        return features

    def encode_core(self, features: Dict[str, Tensor]) -> Tensor:
        """
        Computes sentence embeddings
        """

        with torch.inference_mode():
            device = self._target_device
            features = util.batch_to_device(features, device)
            out_features = self.forward(features)["sentence_embedding"]

        return out_features

    def encode_post(
        self, out_features: Tensor, normalize_embeddings: bool = True
    ) -> NpEmbeddingType:
        with torch.inference_mode():
            embeddings = out_features.detach().cpu()
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings_out: np.ndarray = embeddings.numpy()

        return embeddings_out

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            # truncation="longest_first",
        ).encodings
        return [len(t.tokens) for t in tks]


class CT2SentenceTransformer(SentenceTransformerPatched):
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences
    / text to embeddings.
    Extension of sentence_transformers.SentenceTransformer using a CTranslate2
    model for accelerated inference-only.
    Adapted from https://gist.github.com/guillaumekln/fb125fc3eb108d1a304b7432486e712f

    :param model_name_or_path: If it is a filepath on disc,
        it loads the model from that path.
        If it is not a path, it first tries to download a
        pre-trained SentenceTransformer model.
        If that fails, tries to construct a model from Huggingface
        models repository with that name.
    :param modules:
        This parameter can be used to create custom SentenceTransformer
        models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used
        for computation.
        If None, checks if a GPU can be used.
    :param cache_folder: Path to store models.
        Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable.
    :param use_auth_token:
        HuggingFace authentication token to download private models.
    :param compute_type: weight quantization, scheme for computing,
        (possible values are: int8, int8_float16, int16, float16).
    :param force: force new conversion with CTranslate2, even if it already exists.
    :param vmap: Optional path to a vocabulary mapping file that will be included
        in the converted model directory.
    """

    def __init__(
        self,
        *args,
        compute_type="default",
        force=False,
        vmap: Union[str, None] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self[0] = CT2Transformer(
            self[0],
            compute_type=compute_type,
            force=force,
            vmap=vmap,
        )


class CT2Transformer(torch.nn.Module):
    """Wrapper around a sentence_transformers.models.Transformer
    which routes the forward call to a CTranslate2 encoder model.

    :param compute_type: weight quantization, scheme for computing,
        default uses same as quantization
        (possible values are: int8, int8_float16, int16, float16).
    :param force: force new conversion with CTranslate2, even if it already exists.
    :param vmap: Optional path to a vocabulary mapping file that will be included
        in the converted model directory.
    """

    def __init__(
        self,
        transformer,
        compute_type="default",
        force=False,
        vmap: Union[str, None] = None,
    ):
        super().__init__()
        try:
            import ctranslate2  # type: ignore
        except ImportError:
            logger.exception(
                "for running the CT2SentenceTransformer,"
                " it is required to install CTranslate2 by running "
                " `pip install ctranslate2>=3.16.0`"
            )

        self.tokenizer = transformer.tokenizer
        self._tokenize = transformer.tokenize
        self.compute_type = compute_type
        self.encoder = None

        # Convert to the CTranslate2 model format, if not already done.
        model_dir = transformer.auto_model.config.name_or_path
        self.ct2_model_dir = os.path.join(
            model_dir,
            "ctranslate2_" + ctranslate2.__version__,
        )

        if not os.path.exists(os.path.join(self.ct2_model_dir, "model.bin")) or force:
            if os.path.exists(self.ct2_model_dir) and not os.listdir(
                self.ct2_model_dir
            ):
                force = True
            converter = ctranslate2.converters.TransformersConverter(model_dir)
            converter.convert(self.ct2_model_dir, force=force, vmap=vmap)
        self._ctranslate2_encoder_cls = ctranslate2.Encoder
        self._storage_view = ctranslate2.StorageView

    def children(self):
        # Do not consider the "transformer" attribute as a
        # child module so that it will stay on the CPU.
        return []

    def half(self):
        self.to(dtype="float16")
        return self

    def to(
        self,
        device: int | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        non_blocking: bool = False,
    ) -> "CT2Transformer":
        if not isinstance(device, int):
            raise ValueError("param `dtype` needs to be of type int")
        if not isinstance(dtype, str) or dtype is not None:
            raise ValueError("param `dtype` needs to be of type str")

        if dtype and not ("float" in dtype or "int" in dtype):
            raise ValueError(
                "dtype should be one of `int8`, `float16`, `int8_float16`, `float32`"
            )
        elif dtype:
            new_dtype = True
            self.compute_type = new_dtype
        else:
            new_dtype = False

        if device and (device.startswith("cuda") or device.startswith("cpu")):
            raise ValueError(
                "for param `device`, f'cuda:{index}' or f'cpu:{index}' are supported"
            )
        elif device:
            if ":" in device:
                new_device = device.split(":")[0]
                new_index = device.split(":")[1]
            else:
                new_device = device
                new_index = "0"
        else:
            new_device = ""
            new_index = ""

        if new_device or new_dtype or new_index:
            self.encoder = self._ctranslate2_encoder_cls(
                self.ct2_model_dir,
                device=new_device,
                device_index=new_index,
                intra_threads=torch.get_num_threads(),
                compute_type=self.compute_type,
            )
        return self

    def forward(self, features):
        """overwrites torch forward method with CTranslate model"""
        device = features["input_ids"].device

        if self.encoder is None:
            # The encoder is lazy-loaded to correctly resolve the target device.
            self.encoder = self._ctranslate2_encoder_cls(
                self.ct2_model_dir,
                device=device.type,
                device_index=device.index or 0,
                intra_threads=torch.get_num_threads(),
                compute_type=self.compute_type,
            )

        input_ids = features["input_ids"].to(torch.int32)
        length = features["attention_mask"].sum(1, dtype=torch.int32)

        if device.type == "cpu":
            # PyTorch CPU tensors do not implement the Array interface
            # so a roundtrip to Numpy
            # is required for both the input and output.
            input_ids = input_ids.numpy()
            length = length.numpy()

        input_ids = self._storage_view.from_array(input_ids)
        length = self._storage_view.from_array(length)

        outputs = self.encoder.forward_batch(input_ids, length)

        last_hidden_state = outputs.last_hidden_state
        if device.type == "cpu":
            last_hidden_state = np.array(last_hidden_state)

        features["token_embeddings"] = torch.as_tensor(
            last_hidden_state, device=device
        ).to(torch.float32)

        return features

    def tokenize(self, *args, **kwargs):
        return self._tokenize(*args, **kwargs)


def length_tokenizer(
    _sentences: List[str],
) -> List[int]:
    return [len(i) for i in _sentences]


def get_lengths_with_tokenize(
    _sentences: List[str], tokenize: Callable = length_tokenizer
) -> Tuple[List[int], int]:
    _lengths = tokenize(_sentences)
    return _lengths, sum(_lengths)


class InferenceEngine(Enum):
    torch = SentenceTransformerPatched
    debugengine = DummyTransformer
    ctranslate2 = CT2SentenceTransformer


types: Dict[str, str] = {e.name: e.name for e in InferenceEngine}
InferenceEngineTypeHint = Enum("InferenceEngineTypeHint", types)  # type: ignore
