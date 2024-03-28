import subprocess
import sys

import pytest
import typer
import uvicorn
from fastapi import FastAPI

from infinity_emb.args import EngineArgs
from infinity_emb.infinity_server import (
    UVICORN_LOG_LEVELS,
    Device,
    Dtype,
    InferenceEngine,
    PoolingMethod,
    _start_uvicorn,
    cli,
    create_server,
)


# only run subprocess on non-windows
@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0


def test_patched_cli_help(mocker):
    mocker.patch("typer.run")
    cli()
    typer.run.assert_called_once_with(_start_uvicorn)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_cli_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


def test_create_server():
    app = create_server(EngineArgs(engine="debugengine"))
    assert isinstance(app, FastAPI)


def test_patched_create_uvicorn(mocker):
    mocker.patch("uvicorn.run")
    _start_uvicorn(
        log_level=UVICORN_LOG_LEVELS.debug,  # type: ignore[arg-type]
        engine=InferenceEngine.names_enum().torch,
        device=Device.names_enum().auto,
        dtype=Dtype.names_enum().auto,
        pooling_method=PoolingMethod.names_enum().auto,
    )
    assert uvicorn.run.call_count == 1
