import asyncio
import inspect
import sys

import numpy as np
import pytest
import torch
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from infinity_emb import AsyncEmbeddingEngine, AsyncEngineArray, EngineArgs
from infinity_emb.primitives import (
    Device,
    EmbeddingDtype,
    InferenceEngine,
    ModelNotDeployedError,
)

# Only compile on Linux 3.9-3.11 with torch
SHOULD_TORCH_COMPILE = sys.platform == "linux" and sys.version_info < (3, 12)


@pytest.mark.anyio
async def test_async_api_debug():
    sentences = ["Embedded this is sentence via Infinity.", "Paris is in France."]
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(engine=InferenceEngine.debugengine)
    )
    async with engine:
        embeddings, usage = await engine.embed(sentences)
        embeddings = np.array(embeddings)
        assert usage == sum([len(s) for s in sentences])
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10
        for idx, s in enumerate(sentences):
            assert embeddings[idx][0] == len(s), f"{embeddings}, {s}"


@pytest.mark.anyio
async def test_async_api_torch():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="michaelfeil/bge-small-en-v1.5",
            engine=InferenceEngine.torch,
            revision="main",
            device="cpu",
        )
    )
    assert engine.capabilities == {"embed"}
    async with engine:
        embeddings, usage = await engine.embed(sentences=sentences)
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], np.ndarray)
        embeddings = np.array(embeddings)
        assert usage == sum([len(s) for s in sentences])
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10

        # test if model denies classification and reranking
        with pytest.raises(ModelNotDeployedError):
            await engine.classify(sentences=sentences)
        with pytest.raises(ModelNotDeployedError):
            await engine.rerank(query="dummy", docs=sentences)


@pytest.mark.anyio
@pytest.mark.parametrize("engine", [InferenceEngine.torch, InferenceEngine.optimum])
async def test_engine_reranker_torch_opt(engine):
    model_unpatched = CrossEncoder(
        "mixedbread-ai/mxbai-rerank-xsmall-v1",
    )
    query = "Where is Paris?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish",
    ] * 20
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="mixedbread-ai/mxbai-rerank-xsmall-v1",
            engine=InferenceEngine.torch,
            model_warmup=False,
        )
    )

    query_docs = [(query, doc) for doc in documents]

    async with engine:
        rankings, usage = await engine.rerank(query=query, docs=documents)

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(rankings, rankings_unpatched, rtol=1e-1, atol=1e-1)
    assert usage == sum([len(query) + len(d) for d in documents])
    assert len(rankings) == len(documents)
    np.testing.assert_almost_equal(rankings[:3], [0.83, 0.085, 0.028], decimal=2)


@pytest.mark.anyio
async def test_async_api_torch_CLASSIFY():
    sentences = ["This is awesome.", "I am depressed."]
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="SamLowe/roberta-base-go_emotions",
            engine="torch",
            model_warmup=True,
            device="cpu",
        )
    )
    assert engine.capabilities == {"classify"}

    async with engine:
        predictions, usage = await engine.classify(sentences=sentences)
    assert usage == sum([len(s) for s in sentences])
    assert len(predictions) == len(sentences)
    assert predictions[0][0]["label"] == "admiration"
    assert 0.95 > predictions[0][0]["score"] > 0.94
    assert predictions[1][0]["label"] == "sadness"
    assert 0.81 > predictions[1][0]["score"] > 0.79


@pytest.mark.anyio
async def test_async_api_torch_lengths_via_tokenize_usage():
    sentences = ["Hi", "how", "school", "Pizza Hi"]
    device = "auto"
    if torch.backends.mps.is_available():
        device = "cpu"
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            engine=InferenceEngine.torch,
            device=device,
            lengths_via_tokenize=True,
            model_warmup=False,
            compile=SHOULD_TORCH_COMPILE,
        )
    )
    async with engine:
        embeddings, usage = await engine.embed(sentences=sentences)
        embeddings = np.array(embeddings)
        # usage should be similar to
        assert usage == 5
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_torch_clip_embed():
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg"
    ]  # a photo of two cats
    sentences = [
        "a photo of two cats",
        "a photo of a cat",
        "a photo of a dog",
        "a photo of a car",
    ]
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
            engine=InferenceEngine.torch,
            model_warmup=True,
        )
    )
    async with engine:
        t1, t2 = asyncio.create_task(
            engine.embed(sentences=sentences)
        ), asyncio.create_task(engine.image_embed(images=image_urls))
        emb_text, usage_text = await t1
        emb_image, usage_image = await t2
        emb_text_np = np.array(emb_text)  # type: ignore
        emb_image_np = np.array(emb_image)  # type: ignore

    assert emb_text_np.shape[0] == len(sentences)
    assert emb_image_np.shape[0] == len(image_urls)
    assert emb_text_np.shape[1] >= 10
    assert emb_image_np.shape == emb_image_np[: len(image_urls)].shape

    assert usage_text == sum([len(s) for s in sentences])

    # check if cat image and two cats are most similar
    for i in range(1, len(sentences)):
        assert np.dot(emb_text_np[0], emb_image_np[0]) > np.dot(
            emb_text_np[i], emb_image_np[0]
        )


@pytest.mark.anyio
@pytest.mark.skipif(sys.platform != "linux", reason="only run these on Linux")
@pytest.mark.parametrize(
    "embedding_dtype",
    [
        EmbeddingDtype.float32,
        EmbeddingDtype.int8,
        # EmbeddingDtype.uint8,
        EmbeddingDtype.ubinary,
    ],
)
async def test_async_api_torch_embedding_quant(embedding_dtype: EmbeddingDtype):
    sentences = ["Hi", "how", "school", "Pizza Hi"]
    device = "auto"
    if torch.backends.mps.is_available():
        device = "cpu"
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="michaelfeil/bge-small-en-v1.5",
            engine=InferenceEngine.torch,
            device=Device[device],
            lengths_via_tokenize=True,
            model_warmup=False,
            compile=SHOULD_TORCH_COMPILE,
            embedding_dtype=embedding_dtype,
        )
    )
    async with engine:
        emb, usage = await engine.embed(sentences=sentences)
        embeddings = np.array(emb)  # type: ignore

    if embedding_dtype == EmbeddingDtype.int8:
        assert embeddings.dtype == np.int8
    elif embedding_dtype == EmbeddingDtype.uint8:
        assert embeddings.dtype == np.uint8
    elif embedding_dtype == EmbeddingDtype.ubinary:
        embeddings_up = np.unpackbits(embeddings, axis=-1).astype(int)  # type: ignore
        assert embeddings_up.max() == 1
        assert embeddings_up.min() == 0
    # usage should be similar to
    assert usage == 5
    assert embeddings.shape[0] == len(sentences)
    assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_async_api_failing():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine.from_args(EngineArgs())
    with pytest.raises(ValueError):
        await engine.embed(sentences=sentences)

    await engine.astart()
    assert not engine.is_overloaded()
    assert engine.overload_status()
    await engine.astop()


@pytest.mark.anyio
async def test_async_api_failing_revision():
    with pytest.raises(OSError):
        # revision with just Readme.
        AsyncEmbeddingEngine.from_args(
            EngineArgs(
                model_name_or_path="BAAI/bge-small-en-v1.5",
                revision="a32952c6d05d45f64f9f709a092c00839bcfe70a",
            )
        )


@pytest.mark.parametrize("method_name", list(pytest.ENGINE_METHODS))  # type: ignore
def test_args_between_array_and_engine_same(method_name: str):
    array_method = inspect.getfullargspec(getattr(AsyncEngineArray, method_name))
    engine_method = inspect.getfullargspec(getattr(AsyncEmbeddingEngine, method_name))

    assert "model" in array_method.kwonlyargs
    assert sorted(array_method.args + array_method.kwonlyargs) == sorted(
        engine_method.args + engine_method.kwonlyargs + ["model"]
    )
