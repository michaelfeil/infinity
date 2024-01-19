import json
from pathlib import Path
from typing import Tuple, Union

from infinity_emb.log_handler import logger
from infinity_emb.primitives import (
    Device,
)
from infinity_emb.transformer.abstract import BaseCrossEncoder, BaseEmbedder
from infinity_emb.transformer.utils import (
    EmbedderEngine,
    InferenceEngine,
    PredictEngine,
    RerankEngine,
)


def get_engine_type_from_config(
    model_name_or_path: str, engine: InferenceEngine
) -> Union[EmbedderEngine, RerankEngine]:
    if engine in [InferenceEngine.debugengine, InferenceEngine.fastembed]:
        return EmbedderEngine.from_inference_engine(engine)

    if Path(model_name_or_path).is_dir():
        logger.debug("model is a directory, opening config.json")
        config_path = Path(model_name_or_path) / "config.json"
    else:
        from huggingface_hub import hf_hub_download  # type: ignore

        config_path = hf_hub_download(
            model_name_or_path,
            filename="config.json",
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    if any(
        "SequenceClassification" in arch for arch in config.get("architectures", [])
    ):
        id2label = config.get("id2label", {"0": "dummy"})
        if len(id2label) < 2:
            return RerankEngine.from_inference_engine(engine)
        else:
            return PredictEngine.from_inference_engine(engine)
    else:
        return EmbedderEngine.from_inference_engine(engine)


def select_model(
    model_name_or_path: str,
    batch_size: int,
    engine: InferenceEngine = InferenceEngine.torch,
    model_warmup=True,
    device: Device = Device.auto,
) -> Tuple[Union[BaseCrossEncoder, BaseEmbedder], float]:
    logger.info(
        f"model=`{model_name_or_path}` selected, using engine=`{engine.value}`"
        f" and device=`{device.value}`"
    )
    # TODO: add EncoderEngine
    unloaded_engine = get_engine_type_from_config(model_name_or_path, engine)

    loaded_engine = unloaded_engine.value(model_name_or_path, device=device.value)

    min_inference_t = 4e-3
    if model_warmup:
        # size one, warm up warm start timings.
        loaded_engine.warmup(batch_size=batch_size, n_tokens=1)
        # size one token
        min_inference_t = min(
            loaded_engine.warmup(batch_size=1, n_tokens=1)[1] for _ in range(10)
        )
        emb_per_sec_short, _, log_msg = loaded_engine.warmup(batch_size=64, n_tokens=1)
        logger.info(log_msg)
        # now warm up with max_token, max batch size
        emb_per_sec, _, log_msg = loaded_engine.warmup(
            batch_size=batch_size, n_tokens=512
        )
        logger.info(log_msg)
        logger.info(
            f"model warmed up, between {emb_per_sec:.2f}-{emb_per_sec_short:.2f}"
            f" embeddings/sec at batch_size={batch_size}"
        )

    return loaded_engine, min_inference_t
