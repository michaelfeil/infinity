import numpy as np
import pytest
from sentence_transformers import CrossEncoder  # type: ignore

from infinity_emb import AsyncEmbeddingEngine, transformer
from infinity_emb.primitives import ModelNotDeployedError


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
        model_name_or_path="BAAI/bge-small-en-v1.5",
        engine=transformer.InferenceEngine.torch,
        device="auto",
    )

    async with engine:
        assert engine.capabilities == {"embed"}
        embeddings, usage = await engine.embed(sentences)
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
async def test_async_api_torch_CROSSENCODER():
    query = "Where is Paris?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish",
    ]
    engine = AsyncEmbeddingEngine(
        model_name_or_path="BAAI/bge-reranker-base",
        engine=transformer.InferenceEngine.torch,
        device="auto",
        model_warmup=True,
    )

    async with engine:
        assert engine.capabilities == {"rerank"}
        rankings, usage = await engine.rerank(query=query, docs=documents)

    assert usage == sum([len(query) + len(d) for d in documents])
    assert len(rankings) == len(documents)
    np.testing.assert_almost_equal(rankings, [0.9958, 0.9439, 0.000037], decimal=3)


@pytest.mark.anyio
async def test_engine_crossencoder_vs_sentence_transformers():
    model_unpatched = CrossEncoder("BAAI/bge-reranker-base")
    query = "Where is Paris?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish",
    ] * 100
    engine = AsyncEmbeddingEngine(
        model_name_or_path="BAAI/bge-reranker-base",
        engine=transformer.InferenceEngine.torch,
        device="auto",
        model_warmup=False,
    )

    query_docs = [(query, doc) for doc in documents]

    async with engine:
        rankings, _ = await engine.rerank(query=query, docs=documents)

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(rankings, rankings_unpatched, rtol=1e-2, atol=1e-2)


@pytest.mark.anyio
async def test_async_api_torch_CLASSIFY():
    sentences = ["This is awesome.", "I am depressed."]
    engine = AsyncEmbeddingEngine(
        model_name_or_path="SamLowe/roberta-base-go_emotions",
        engine="torch",
        model_warmup=True,
    )

    async with engine:
        assert engine.capabilities == {"classify"}
        predictions, usage = await engine.classify(sentences=sentences)
    assert usage == sum([len(s) for s in sentences])
    assert len(predictions) == len(sentences)
    assert predictions[0][0]["label"] == "admiration"
    assert 0.95 > predictions[0][0]["score"] > 0.94
    assert predictions[1][0]["label"] == "sadness"
    assert 0.81 > predictions[1][0]["score"] > 0.79


@pytest.mark.anyio
async def test_async_api_torch_usage():
    sentences = ["Hi", "how", "school", "Pizza Hi"]
    engine = AsyncEmbeddingEngine(
        engine=transformer.InferenceEngine.torch,
        device="auto",
        lengths_via_tokenize=True,
        model_warmup=False,
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
        engine=transformer.InferenceEngine.fastembed, device="cpu", model_warmup=False
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
