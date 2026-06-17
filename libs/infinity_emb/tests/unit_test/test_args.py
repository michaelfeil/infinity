import pytest

from infinity_emb.args import EngineArgs
from infinity_emb.engine import _clamp_to_ceiling
from infinity_emb.primitives import Device, InferenceEngine


def test_EngineArgs_no_input():
    args = EngineArgs()
    # rerank token ceilings are unset by default (no limit)
    assert args.max_query_tokens is None
    assert args.max_tokens_per_doc is None
    assert args.max_pair_tokens is None


def test_engine_args_rerank_limits():
    args = EngineArgs(
        model_name_or_path="michaelfeil/bge-small-en-v1.5",
        device="cpu",
        max_query_tokens=64,
        max_tokens_per_doc=256,
        max_pair_tokens=320,
    )
    assert args.max_query_tokens == 64
    assert args.max_tokens_per_doc == 256
    assert args.max_pair_tokens == 320


def test_engine_args_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        EngineArgs(model_name_or_path="michaelfeil/bge-small-en-v1.5", max_query_tokens=0)


@pytest.mark.parametrize(
    "requested, ceiling, expected",
    [
        (None, None, None),  # no limit at all
        (None, 256, 256),  # client unset -> use the server ceiling
        (128, None, 128),  # no ceiling -> use the request as-is
        (128, 256, 128),  # client lowers below the ceiling
        (512, 256, 256),  # client cannot exceed the ceiling -> clamped down
    ],
)
def test_clamp_to_ceiling(requested, ceiling, expected):
    assert _clamp_to_ceiling(requested, ceiling) == expected


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
