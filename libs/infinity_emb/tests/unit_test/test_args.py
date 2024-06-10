from infinity_emb.args import EngineArgs
from infinity_emb.primitives import Device, InferenceEngine


def test_EngineArgs_no_input():
    EngineArgs()


def test_engine_args():
    args = EngineArgs(
        model_name_or_path="michaelfeil/bge-small-en-v1.5",
        batch_size=64,
        revision=None,
        trust_remote_code=True,
        engine="torch",
        model_warmup=False,
        vector_disk_cache_path="",
        device="cpu",
        lengths_via_tokenize=False,
    )

    assert args.model_name_or_path == "michaelfeil/bge-small-en-v1.5"
    assert args.batch_size == 64
    assert args.revision is None
    assert args.trust_remote_code
    assert args.engine == InferenceEngine.torch
    assert not args.model_warmup
    assert args.vector_disk_cache_path == ""
    assert args.device == Device.cpu
    assert not args.lengths_via_tokenize


def test_multiargs():
    EngineArgs.from_env()
