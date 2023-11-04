import copy
import os
from typing import Dict, List, Union

import numpy as np

from infinity_emb.inference.primitives import NpEmbeddingType
from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseTransformer

try:
    import torch
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    from torch import Tensor
    from torch.nn import Module

    TORCH_AVAILABLE = True
except ImportError:
    torch, Tensor = None, None

    class SentenceTransformer:
        pass

    class Module:
        pass

    TORCH_AVAILABLE = False

try:
    from optimum.bettertransformer import BetterTransformer

    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False

__all__ = [
    "SentenceTransformerPatched",
    "CT2SentenceTransformer",
]


class SentenceTransformerPatched(SentenceTransformer, BaseTransformer):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *args, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed."
                " `pip install infinity-emb[torch]` "
                "or pip install infinity-emb[torch,optimum]`"
            )
        super().__init__(*args, **kwargs)
        device = self._target_device
        self.to(device)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        fm = self._first_module()
        self._infinity_tokenizer = copy.deepcopy(fm.tokenizer)
        if OPTIMUM_AVAILABLE and not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
            logger.info(
                "Adding optimizations via Huggingface optimum. "
                "Disable by setting the env var `INFINITY_DISABLE_OPTIMUM`"
            )
            try:
                fm.auto_model = BetterTransformer.transform(fm.auto_model)
            except Exception as ex:
                logger.exception(f"BetterTransformer failed with {ex}")
                exit(1)
        elif not os.environ.get("INFINITY_DISABLE_OPTIMUM", False):
            logger.info(
                "No optimizations via Huggingface optimum,"
                " it is disabled via env INFINITY_DISABLE_OPTIMUM "
            )
        else:
            logger.info(
                "No optimizations via Huggingface optimum, "
                "install `pip install infinity-emb[optimum]`"
            )

        self.eval()
        if self._target_device.type == "cuda" and os.environ.get(
            "INFINITY_TORCH_ENABLE_HALF", False
        ):
            logger.info(
                "Switching to half() precision (fp16)."
                "Enabled by the setting the env var `INFINITY_TORCH_ENABLE_HALF`"
            )
            self.half()

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
            embeddings = out_features.detach().cpu().to(torch.float32)
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self[0] = CT2Transformer(
            self[0],
            compute_type=compute_type,
            force=force,
            vmap=vmap,
        )


class CT2Transformer(Module):
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
