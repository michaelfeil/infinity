import asyncio
import inspect
from uuid import uuid4

import pytest

from infinity_emb import AsyncEngineArray, EngineArgs, SyncEngineArray
from infinity_emb.sync_engine import WeakAsyncLifeMixin


def test_weaklifemixin():
    async def asleep():
        await asyncio.sleep(2)
        return 1

    mixin = WeakAsyncLifeMixin()
    res = mixin.async_run(asleep)
    assert res.result(timeout=10) == 1
    del mixin
    # TODO: CHECK that no more tasks can be scheduled


def test_sync_engine_double():
    model1 = str(uuid4())
    model2 = str(uuid4())
    s_eng_array1 = SyncEngineArray(
        [
            EngineArgs(
                model_name_or_path=model1,
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
        ]
    )
    s_eng_array2 = SyncEngineArray(
        [
            EngineArgs(
                model_name_or_path=model2,
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
        ]
    )

    future_result1 = s_eng_array1.embed(model=model1, sentences=["Hello world!"])
    future_result2 = s_eng_array2.embed(model=model2, sentences=["Hello world!"])
    embedding = future_result1.result()
    embedding2 = future_result2.result()
    s_eng_array1.stop()
    s_eng_array2.stop()

    assert (embedding[0][0] == embedding2[0][0]).all()
    # free memory
    s_eng_array1 = None
    s_eng_array2 = None


def test_sync_engine():
    model = str(uuid4())
    s_eng_array = SyncEngineArray(
        [
            EngineArgs(
                model_name_or_path=model,
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
            EngineArgs(
                model_name_or_path=str(uuid4()),
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
        ]
    )

    future_result1 = s_eng_array.embed(model=model, sentences=["Hello world!"])
    future_result2 = s_eng_array.embed(model=model, sentences=["Hello world!"])
    embedding = future_result1.result()
    embedding2 = future_result2.result()

    assert (embedding[0][0] == embedding2[0][0]).all()
    s_eng_array.stop()


@pytest.mark.parametrize(
    "model_id, method, payload",
    [
        ("michaelfeil/bge-small-en-v1.5", "embed", {"sentences": ["Hello world!"]}),
        (
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M",
            "image_embed",
            {"images": [pytest.IMAGE_SAMPLE_URL]},
        ),
        (
            "philschmid/tiny-bert-sst2-distilled",
            "classify",
            {"sentences": ["I love this movie"]},
        ),
        (
            "mixedbread-ai/mxbai-rerank-xsmall-v1",
            "rerank",
            {"query": "I love this movie", "docs": ["I love this movie"]},
        ),
    ],
)
def test_sync_engine_on_model(model_id, method: str, payload: dict):
    try:
        s_eng_array = SyncEngineArray(
            [
                EngineArgs(
                    model_name_or_path=model_id,
                    device="cpu",  # type: ignore
                    engine="torch",  # type: ignore
                    model_warmup=False,
                ),
            ]
        )

        future_result = getattr(s_eng_array, method)(model=model_id, **payload)
        embedding = future_result.result()

        assert len(embedding) > 0
    finally:
        s_eng_array.stop()


@pytest.mark.parametrize("method_name", list(pytest.ENGINE_METHODS) + ["from_args"])  # type: ignore
def test_args_between_sync_and_async_same(method_name: str):
    sync_method = inspect.getfullargspec(getattr(SyncEngineArray, method_name))
    async_method = inspect.getfullargspec(getattr(AsyncEngineArray, method_name))
    if method_name in list(pytest.ENGINE_METHODS):  # type: ignore
        assert "model" in sync_method.kwonlyargs
        assert "model" in async_method.kwonlyargs
    assert sync_method.args == async_method.args
    assert sync_method.kwonlyargs == async_method.kwonlyargs


if __name__ == "__main__":
    test_sync_engine()
