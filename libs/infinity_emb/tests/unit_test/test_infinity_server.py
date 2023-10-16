import subprocess

from fastapi import FastAPI

from infinity_emb import create_server
from infinity_emb.transformer.utils import InferenceEngine


def test_cli_help():
    log = subprocess.run(["infinity_emb", "--help"])
    assert log.returncode == 0


def test_cli_wrong_batch_size():
    log = subprocess.run(["infinity_emb", "--batch-size", "WrongArgument"])
    assert log.returncode == 2


def test_create_server():
    app = create_server(engine=InferenceEngine.debugengine)
    assert isinstance(app, FastAPI)
