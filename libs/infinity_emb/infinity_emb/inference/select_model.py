from time import perf_counter
from typing import List, Optional, Tuple

from infinity_emb.inference.primitives import EmbeddingResult, NpEmbeddingType
from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseTransformer
from infinity_emb.transformer.utils import InferenceEngine


def select_model_to_functional(
    model_name_or_path: str, batch_size: int, engine: InferenceEngine, model_warmup=True
):
    logger.info(f"model=`{model_name_or_path}` selected, using engine=`{engine.value}`")
    init_engine = engine.value(model_name_or_path)

    min_inference_t = 4e-3
    if model_warmup:
        # size one, warm up warm start timings.
        runtime_check_callable(init_engine, log=False)
        # size one
        runtime_check_callable(init_engine, log=True)
        min_inference_t = min(
            runtime_check_callable(init_engine, log=False)[1] for _ in range(10)
        )
        # warm-up with max batch size
        emb_per_sec_short, _ = runtime_check_callable(
            init_engine, sample=["up"] * batch_size
        )
        # warm-up with max batch size
        emb_per_sec, _ = runtime_check_callable(
            init_engine,
            sample=["warming up with max batch size and 1K tokens per sentence " * 76]
            * batch_size,
        )
        logger.info(
            f"model warmed up, between {emb_per_sec:.2f}-{emb_per_sec_short:.2f}"
            f" embeddings/sec at batch_size={batch_size}"
        )

    return init_engine, min_inference_t


def runtime_check_callable(
    model: BaseTransformer, sample: Optional[List[str]] = None, log=True
) -> Tuple[float, float]:
    if sample is None:
        sample = ["warm"]
    inp = [EmbeddingResult(sentence=s, future=None) for s in sample]  # type: ignore
    start = perf_counter()
    sentences = [item.sentence for item in inp]
    feat = model.encode_pre(sentences)
    tokenization_time = perf_counter()
    embed = model.encode_core(feat)
    inference_time = perf_counter()
    embeddings = model.encode_post(embed)
    for i, item in enumerate(inp):
        item.embedding = embeddings[i]
    post_time = perf_counter()

    if not len(inp) == len(sample):
        raise ValueError(
            "The output of the callable function is not of the same length as the input"
        )

    if not isinstance(inp[0].embedding, NpEmbeddingType):
        raise ValueError(
            "The output of the callable function is not of type EmbeddingResult"
        )

    if log:
        logger.info(
            (
                f"Getting timings for batch_size={len(sample)}"
                f" and avg tokens per sentence={model.tokenize_lengths(sample)[0]}\n"
                f"\t{(tokenization_time - start)*1000:.2f} \t ms tokenization\n"
                f"\t{(inference_time-tokenization_time)*1000:.2f} \t ms inference\n"
                f"\t{(post_time-inference_time)*1000:.2f} \t ms post-processing\n"
                f"\t{(post_time - start)*1000:.2f} \t ms total\n"
                f"embeddings/sec: {len(sample) / (post_time - start):.2f}"
            )
        )

    emb_per_sec = len(sample) / (post_time - start)
    return emb_per_sec, inference_time - tokenization_time
