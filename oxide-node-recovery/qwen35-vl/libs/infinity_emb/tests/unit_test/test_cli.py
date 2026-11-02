import subprocess
import sys

import pytest


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
