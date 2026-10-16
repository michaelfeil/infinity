import numpy as np
import pytest

from infinity_emb import AsyncEmbeddingEngine, AsyncEngineArray, EngineArgs


ITERABLES = pytest.mark.parametrize(
    "as_iterable",
    [list, tuple, iter, lambda engines: (engine for engine in engines)],
    ids=["list", "tuple", "iterator", "generator"],
)


def make_engine(name: str) -> AsyncEmbeddingEngine:
    return AsyncEmbeddingEngine.from_args(
        EngineArgs(
            model_name_or_path=name,
            engine="debugengine",
            model_warmup=False,
        )
    )


@pytest.fixture
def engines():
    return [make_engine("first-model"), make_engine("second-model")]


@ITERABLES
def test_engine_array_preserves_engines(engines, as_iterable):
    array = AsyncEngineArray(as_iterable(engines))

    assert list(array) == engines
    assert list(array) == engines
    assert not array.is_running()
    for index, engine in enumerate(engines):
        assert array[index] is engine
        assert array[engine.engine_args.served_model_name] is engine


@ITERABLES
def test_engine_array_rejects_empty_iterables(as_iterable):
    with pytest.raises(ValueError, match="Engines cannot be empty"):
        AsyncEngineArray(as_iterable([]))


@ITERABLES
def test_engine_array_rejects_duplicate_names(as_iterable):
    engines = [make_engine("same-model"), make_engine("same-model")]

    with pytest.raises(ValueError, match="Engines must have unique model names"):
        AsyncEngineArray(as_iterable(engines))


@ITERABLES
@pytest.mark.anyio
async def test_engine_array_iterable_lifecycle(engines, as_iterable):
    array = AsyncEngineArray(as_iterable(engines))
    try:
        await array.astart()
        assert array.is_running()
        assert all(engine.is_running for engine in engines)

        for engine in engines:
            embeddings, usage = await array.embed(
                model=engine.engine_args.served_model_name,
                sentences=["Hi", "hello"],
                matryoshka_dim=4,
            )
            np.testing.assert_array_equal(embeddings, [[2.0] * 4, [5.0] * 4])
            assert usage == 7
    finally:
        await array.astop()

    assert not array.is_running()
    assert all(not engine.is_running for engine in engines)
