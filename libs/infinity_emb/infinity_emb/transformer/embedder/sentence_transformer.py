import copy
import os
from pathlib import Path
from typing import Dict, List, Union

import numpy as np

from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Dtype, EmbeddingReturnType
from infinity_emb.transformer.abstract import BaseEmbedder
from infinity_emb.transformer.acceleration import to_bettertransformer

try:
    import torch
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE  # type: ignore
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    from torch import Tensor
    from torch.nn import Module

    TORCH_AVAILABLE = True
except ImportError:
    torch, Tensor = None, None  # type: ignore

    class SentenceTransformer:  # type: ignore
        pass

    class Module:  # type: ignore
        pass

    TORCH_AVAILABLE = False


__all__ = [
    "SentenceTransformerPatched",
    "CT2SentenceTransformer",
]


class SentenceTransformerPatched(SentenceTransformer, BaseEmbedder):
    """SentenceTransformer with .encode_core() and no microbatching"""

    def __init__(self, *, engine_args=EngineArgs):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed."
                " `pip install infinity-emb[torch]` "
                "or pip install infinity-emb[torch,optimum]`"
            )
        super().__init__(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
            device=engine_args.device.value,
        )
        self.to(self.device)
        # make a copy of the tokenizer,
        # to be able to could the tokens in another thread
        # without corrupting the original.
        fm = self._first_module()
        self._infinity_tokenizer = copy.deepcopy(fm.tokenizer)
        self.eval()

        fm.auto_model = to_bettertransformer(
            fm.auto_model,
            logger,
            disable=(engine_args.device == "mps" and not engine_args.bettertransformer),
        )

        if self.device.type == "cuda" and engine_args.dtype in [
            Dtype.auto,
            Dtype.float16,
        ]:
            logger.info("Switching to half() precision (cuda: fp16). ")
            self.half()
        if engine_args.compile:
            logger.info("using torch.compile()")
            fm.auto_model = torch.compile(fm.auto_model, dynamic=True)

    def encode_pre(self, sentences) -> Dict[str, Tensor]:
        features = self.tokenize(sentences)

        return features

    def encode_core(self, features: Dict[str, Tensor]) -> Tensor:
        """
        Computes sentence embeddings
        """

        with torch.inference_mode():
            features = util.batch_to_device(features, self.device)
            out_features = self.forward(features)["sentence_embedding"]

        return out_features

    def encode_post(
        self, out_features: Tensor, normalize_embeddings: bool = True
    ) -> EmbeddingReturnType:
        with torch.inference_mode():
            embeddings = out_features.detach().cpu().to(torch.float32)
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings: np.ndarray = embeddings.numpy()  # type: ignore

        return embeddings  # type: ignore

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
            # max_length=self._infinity_tokenizer.model_max_length,
            truncation="longest_first",
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
        *,
        engine_args=EngineArgs,
        ct2_compute_type: str = "default",
    ):
        self._prefered_device = engine_args.device.value
        super().__init__(engine_args=engine_args)
        self[0] = CT2Transformer(
            self[0],
            compute_type=ct2_compute_type,
            force=None,
            vmap=None,
        )

    @property
    def device(self):
        if self._prefered_device is not None:
            return torch.device(self._prefered_device)
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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

        logger.warning(
            "deprecated: ct2 inference is deprecated and will be removed in the future."
        )

        self.tokenizer = transformer.tokenizer
        self._tokenize = transformer.tokenize
        self.compute_type = compute_type
        self.encoder = None

        # Convert to the CTranslate2 model format, if not already done.
        model_dir = transformer.auto_model.config.name_or_path
        self.ct2_model_dir = os.path.join(
            HUGGINGFACE_HUB_CACHE,
            "ctranslate2_" + ctranslate2.__version__,
            str(Path(model_dir).name.replace("/", "_")),
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
