from typing import Dict, List, Optional

try:
    # autotokenizer
    import torch
    from transformers import AutoTokenizer, pipeline  # type: ignore

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseClassifer
from infinity_emb.transformer.acceleration import to_bettertransformer


class SentenceClassifier(BaseClassifer):
    def __init__(self, model_name_or_path, device: Optional[str] = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is not installed."
                " `pip install infinity-emb[torch]` "
                "or pip install infinity-emb[torch,optimum]`"
            )

        used_device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline(
            task="text-classification",
            model=model_name_or_path,
            device=used_device,
            top_k=None,
            torch_dtype=torch.float32 if used_device == "cpu" else torch.float16,
        )
        self._pipe.model = to_bettertransformer(self._pipe.model, logger=logger)

        self._infinity_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def encode_pre(self, sentences: List[str]):
        """takes care of the tokenization and feature preparation"""
        return sentences

    def encode_core(self, features):
        """runs plain inference, on cpu/gpu"""
        return self._pipe(features, batch_size=256, truncation=True, padding=True)

    def encode_post(self, classes) -> Dict[str, float]:
        """runs post encoding such as normlization"""
        return classes

    def tokenize_lengths(self, sentences: List[str]) -> List[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
        ).encodings
        return [len(t.tokens) for t in tks]
