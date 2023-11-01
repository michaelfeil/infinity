import pytest

from infinity_emb.inference.select_model import select_model_to_functional
from infinity_emb.transformer.utils import InferenceEngine


@pytest.mark.parametrize("engine", [e for e in InferenceEngine])
def test_engine(engine):
    select_model_to_functional(
        engine=engine,
        model_name_or_path=pytest.DEFAULT_BERT_MODEL,
        batch_size=4,
    )
