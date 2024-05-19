from infinity_emb._optional_imports import CHECK_TRANSFORMERS
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.primitives import Device
from infinity_emb.transformer.abstract import BaseClassifer
from infinity_emb.transformer.acceleration import to_bettertransformer

if CHECK_TRANSFORMERS.is_available:
    from transformers import AutoTokenizer, pipeline  # type: ignore


class SentenceClassifier(BaseClassifer):
    def __init__(
        self,
        *,
        engine_args: EngineArgs,
    ) -> None:
        CHECK_TRANSFORMERS.mark_required()
        self._pipe = pipeline(
            task="text-classification",
            model=engine_args.model_name_or_path,
            trust_remote_code=engine_args.trust_remote_code,
            device=engine_args.device.resolve(),
            top_k=None,
            revision=engine_args.revision,
        )
        if self._pipe.device.type != "cpu":  # and engine_args.dtype == "float16":
            self._pipe.model = self._pipe.model.half()

        if not (engine_args.device == Device.mps or not engine_args.bettertransformer):
            self._pipe.model = to_bettertransformer(
                self._pipe.model,
                logger,
            )

        self._infinity_tokenizer = AutoTokenizer.from_pretrained(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            trust_remote_code=engine_args.trust_remote_code,
        )

    def encode_pre(self, sentences: list[str]):
        """takes care of the tokenization and feature preparation"""
        return sentences

    def encode_core(self, features):
        """runs plain inference, on cpu/gpu"""
        return self._pipe(features, batch_size=256, truncation=True, padding=True)

    def encode_post(self, classes) -> dict[str, float]:
        """runs post encoding such as normalization"""
        return classes

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """gets the lengths of each sentences according to tokenize/len etc."""
        tks = self._infinity_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=False,
            return_token_type_ids=False,
            return_attention_mask=False,
            return_length=False,
        ).encodings
        return [len(t.tokens) for t in tks]
