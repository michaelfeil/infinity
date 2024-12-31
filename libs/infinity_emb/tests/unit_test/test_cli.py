import subprocess
import sys

import pytest
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


# only run subprocess on non-windows
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_v1_help():
    log = subprocess.run(["infinity_emb", "v1", "--help"])
    assert log.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_v2_help():
    log = subprocess.run(["infinity_emb", "v2", "--help"])
    assert log.returncode == 0


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_v1_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "v1", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_v2_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "v2", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_v2_weird():
    log = subprocess.run(
        [
            "infinity_emb",
            "v2",
            "--model-id",
            "model1",
            "--model-id",
            "model2",
            "--model-id",
            "model3",
            "--batch-size",
            "32",
            "--batch-size",
            "32",
        ]
    )
    assert log.returncode == 1


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_cli_preload(version):
    log = subprocess.run(["infinity_emb", f"{version}", "--preload-only"])
    assert log.returncode == 0


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
