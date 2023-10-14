import pytest

from infinity_emb.transformer.utils import InferenceEngine
from infinity_emb.inference.select_model import select_model_to_functional


@pytest.mark.parametrize("engine", [e for e in InferenceEngine])
def test_engine(engine):
    select_model_to_functional(
        engine=engine,
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=4,
    )
