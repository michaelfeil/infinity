# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import json
from pathlib import Path
from typing import Union, TYPE_CHECKING


from infinity_emb.log_handler import logger
from functools import partial
from infinity_emb.transformer.utils import (
    AudioEmbedEngine,
    EmbedderEngine,
    ImageEmbedEngine,
    InferenceEngine,
    PredictEngine,
    RerankEngine,
)

if TYPE_CHECKING:
    from infinity_emb.transformer.abstract import BaseTypeHint  # , CallableReturningBaseTypeHint
    from infinity_emb.args import (
        EngineArgs,
    )


def get_engine_type_from_config(
    engine_args: "EngineArgs",
) -> Union[EmbedderEngine, RerankEngine, PredictEngine, ImageEmbedEngine, AudioEmbedEngine]:
    """resolved the class of inference engine path from config.json of the repo."""
    if engine_args.engine in [InferenceEngine.debugengine]:
        return EmbedderEngine.from_inference_engine(engine_args.engine)

    if Path(engine_args.model_name_or_path).is_dir():
        logger.debug("model is a directory, opening config.json")
        config_path = Path(engine_args.model_name_or_path) / "config.json"
    else:
        from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

        config_path = hf_hub_download(
            engine_args.model_name_or_path,
            revision=engine_args.revision,
            filename="config.json",
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    if any("SequenceClassification" in arch for arch in config.get("architectures", [])):
        id2label = config.get("id2label", {"0": "dummy"})
        if len(id2label) < 2:
            return RerankEngine.from_inference_engine(engine_args.engine)
        else:
            return PredictEngine.from_inference_engine(engine_args.engine)
    if config.get("vision_config"):
        return ImageEmbedEngine.from_inference_engine(engine_args.engine)
    if config.get("audio_config") and "clap" in config.get("model_type", "").lower():
        return AudioEmbedEngine.from_inference_engine(engine_args.engine)

    else:
        return EmbedderEngine.from_inference_engine(engine_args.engine)


def _get_engine_replica(unloaded_engine, engine_args, device_map) -> "BaseTypeHint":
    engine_args_copy = engine_args.copy()
    engine_args_copy._loading_strategy.device_placement = device_map
    loaded_engine = unloaded_engine.value(engine_args=engine_args_copy)

    if engine_args.model_warmup:
        # size one, warm up warm start timings.
        # loaded_engine.warmup(batch_size=engine_args.batch_size, n_tokens=1)
        # size one token
        min(loaded_engine.warmup(batch_size=1, n_tokens=1)[1] for _ in range(5))
        emb_per_sec_short, max_inference_temp, log_msg = loaded_engine.warmup(
            batch_size=engine_args.batch_size, n_tokens=1
        )

        logger.info(log_msg)
        # now warm up with max_token, max batch size
        loaded_engine.warmup(batch_size=engine_args.batch_size, n_tokens=512)
        emb_per_sec, _, log_msg = loaded_engine.warmup(
            batch_size=engine_args.batch_size, n_tokens=512
        )
        logger.info(log_msg)
        logger.info(
            f"model warmed up, between {emb_per_sec:.2f}-{emb_per_sec_short:.2f}"
            f" embeddings/sec at batch_size={engine_args.batch_size}"
        )
    return loaded_engine


def select_model(
    engine_args: "EngineArgs",
) -> list[partial["BaseTypeHint"]]:
    """based on engine args, fully instantiates the Engine."""
    logger.info(
        f"model=`{engine_args.model_name_or_path}` selected, "
        f"using engine=`{engine_args.engine.value}`"
        f" and device=`{engine_args.device.resolve()}`"
    )
    unloaded_engine = get_engine_type_from_config(engine_args)

    engine_replicas = []

    for device_map in engine_args._loading_strategy.device_mapping:  # type: ignore
        engine_replicas.append(
            partial(_get_engine_replica, unloaded_engine, engine_args, device_map)
        )
    assert len(engine_replicas) > 0, "No engine replicas were loaded"

    return engine_replicas
