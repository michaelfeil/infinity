import numpy as np
import pytest
import torch
from sentence_transformers import CrossEncoder  # type: ignore

from infinity_emb import AsyncEmbeddingEngine, EngineArgs
from infinity_emb.primitives import InferenceEngine, ModelNotDeployedError


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
            model_name_or_path="BAAI/bge-small-en-v1.5",
            engine=InferenceEngine.torch,
            revision="main",
            device="cpu",
        )
    )
    assert engine.capabilities == {"embed"}
    async with engine:
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
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="BAAI/bge-reranker-base",
            engine=InferenceEngine.torch,
            revision=None,
            device="auto",
            model_warmup=True,
        )
    )

    assert engine.capabilities == {"rerank"}

    async with engine:
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
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="BAAI/bge-reranker-base",
            engine=InferenceEngine.torch,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_warmup=False,
        )
    )

    query_docs = [(query, doc) for doc in documents]

    async with engine:
        rankings, _ = await engine.rerank(query=query, docs=documents)

    rankings_unpatched = model_unpatched.predict(query_docs)

    np.testing.assert_allclose(rankings, rankings_unpatched, rtol=1e-2, atol=1e-2)


@pytest.mark.anyio
async def test_async_api_optimum_crossencoder():
    query = "Where is Paris?"
    documents = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "You can now purchase my favorite dish",
    ]
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path="Xenova/bge-reranker-base",
            engine=InferenceEngine.optimum,
            revision=None,
            device="cpu",
            model_warmup=False,
        )
    )

    assert engine.capabilities == {"rerank"}

    async with engine:
        rankings, usage = await engine.rerank(query=query, docs=documents)

    assert usage == sum([len(query) + len(d) for d in documents])
    assert len(rankings) == len(documents)
    np.testing.assert_almost_equal(rankings, [0.99743, 0.966, 0.000037], decimal=3)


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
async def test_async_api_torch_usage():
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
        )
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
    engine = AsyncEmbeddingEngine.from_args(
        EngineArgs(
            engine=InferenceEngine.fastembed,
            device="cpu",
            model_warmup=False,
        )
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
    engine = AsyncEmbeddingEngine.from_args(EngineArgs())
    with pytest.raises(ValueError):
        await engine.embed(sentences)

    await engine.astart()
    assert not engine.is_overloaded()
    assert engine.overload_status()

    with pytest.raises(ValueError):
        await engine.astart()
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
