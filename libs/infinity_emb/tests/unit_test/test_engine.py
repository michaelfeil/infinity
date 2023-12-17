

import numpy as np
import pytest

from infinity_emb import AsyncEmbeddingEngine, transformer

@pytest.mark.anyio
async def test_async_api_debug():
    sentences = ["Embedded this is sentence via Infinity.", "Paris is in France."]
    engine = AsyncEmbeddingEngine(engine=transformer.InferenceEngine.debugengine)
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
    engine = AsyncEmbeddingEngine(
        engine=transformer.InferenceEngine.torch, device="auto"
    )
    async with engine:
        embeddings, usage = await engine.embed(sentences)
        embeddings = np.array(embeddings)
        assert usage == sum([len(s) for s in sentences])
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_async_api_torch_usage():
    sentences = ["Hi", "how", "school", "Pizza Hi"]
    engine = AsyncEmbeddingEngine(
        engine=transformer.InferenceEngine.torch,
        device="auto",
        lengths_via_tokenize=True,
    )
    async with engine:
        embeddings, usage = await engine.embed(sentences)
        embeddings = np.array(embeddings)
        # usage should be similar to
        assert usage == 5
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10


@pytest.mark.anyio
async def test_async_api_fastembed():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine(
        engine=transformer.InferenceEngine.fastembed, device="cpu"
    )
    async with engine:
        embeddings, usage = await engine.embed(sentences)
        embeddings = np.array(embeddings)
        assert usage == sum([len(s) for s in sentences])
        assert embeddings.shape[0] == len(sentences)
        assert embeddings.shape[1] >= 10
        assert not engine.is_overloaded()


@pytest.mark.anyio
async def test_async_api_failing():
    sentences = ["Hi", "how"]
    engine = AsyncEmbeddingEngine()
    with pytest.raises(ValueError):
        await engine.embed(sentences)

    await engine.astart()
    assert not engine.is_overloaded()
    assert engine.overload_status()

    with pytest.raises(ValueError):
        await engine.astart()
    await engine.astop()

