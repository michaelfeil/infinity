import pytest

from infinity_emb.args import EngineArgs
from infinity_emb.inference.select_model import select_model
from infinity_emb.primitives import Device, InferenceEngine


@pytest.mark.parametrize("engine", [e for e in InferenceEngine])
def test_engine(engine):
    select_model(
        EngineArgs(
            engine=engine,
            model_name_or_path=(pytest.DEFAULT_BERT_MODEL),
            batch_size=4,
            device=Device.cpu,
        )
    )
