import uvicorn
from fastapi import FastAPI

from infinity_emb.args import EngineArgs
from infinity_emb.infinity_server import (
    create_server,
)
from infinity_emb.cli import v1, v2

from infinity_emb.cli import (
    UVICORN_LOG_LEVELS,
    Device,
    Dtype,
    InferenceEngine,
    PoolingMethod,
)


def test_create_server():
    app = create_server(engine_args_list=[EngineArgs(engine="debugengine")])
    assert isinstance(app, FastAPI)


def test_patched_create_uvicorn_v1(mocker):
    mocker.patch("uvicorn.run")
    v1(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore[arg-type]
        engine=InferenceEngine.torch,
        device=Device.auto,
        dtype=Dtype.auto,
        pooling_method=PoolingMethod.auto,
    )
    assert uvicorn.run.call_count == 1


def test_patched_create_uvicorn_v2(mocker):
    mocker.patch("uvicorn.run")
    v2(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore[arg-type]
        engine=[InferenceEngine.torch],
        model_id=["michaelfeil/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"],
        device=[Device.auto],
        dtype=[Dtype.auto],
        pooling_method=[PoolingMethod.auto],
    )
    assert uvicorn.run.call_count == 1
