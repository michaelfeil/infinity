import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from infinity_emb.inference.batch_handler import matryososka_slice
from infinity_emb.primitives import EmbeddingDtype, MatryoshkaDimError
from infinity_emb.transformer.embedder.sentence_transformer import SentenceTransformerPatched


@pytest.fixture
def colbert_embeddings():
    # Exercise the native ColBERT postprocessor with deterministic features,
    # without loading model weights or claiming pretrained-model coverage.
    model = SimpleNamespace(mode_colbert=True, embedding_dtype=EmbeddingDtype.float32)
    features = {
        "token_embeddings": torch.arange(2 * 11 * 8, dtype=torch.float32).reshape(2, 11, 8),
        "attention_mask": torch.tensor([[1] * 3 + [0] * 8, [1] * 11]),
    }
    embeddings = SentenceTransformerPatched.encode_post(model, features)
    assert [embedding.shape for embedding in embeddings] == [(3, 8), (11, 8)]
    return embeddings


def assert_slice(embeddings, dimensions):
    original = [embedding.copy() for embedding in embeddings]
    result = matryososka_slice(embeddings, dimensions)

    if not dimensions:
        assert result is embeddings
    assert len(result) == len(embeddings)
    for actual, before, after in zip(result, original, embeddings):
        expected = before if not dimensions else before[..., :dimensions]
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == before.dtype
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("dimensions", [None, 0, 1, 4, 8])
def test_matryoshka_dense_embeddings(dimensions):
    embeddings = list(np.arange(16, dtype=np.float32).reshape(2, 8))
    assert_slice(embeddings, dimensions)


@pytest.mark.parametrize("dimensions", [None, 0, 1, 4, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_matryoshka_colbert_embeddings(colbert_embeddings, dimensions, reverse):
    embeddings = colbert_embeddings[::-1] if reverse else colbert_embeddings
    assert_slice(embeddings, dimensions)


@pytest.mark.parametrize("dimensions", [-1, 9])
def test_matryoshka_dense_invalid_dimensions(dimensions):
    embeddings = list(np.arange(16, dtype=np.float32).reshape(2, 8))
    message = f"matryoshka_dim={dimensions} is not in a valid range. Select between 1 and 8."
    with pytest.raises(MatryoshkaDimError, match=re.escape(message)):
        matryososka_slice(embeddings, dimensions)


@pytest.mark.parametrize("dimensions", [-1, 9])
@pytest.mark.parametrize("reverse", [False, True])
def test_matryoshka_colbert_invalid_dimensions(colbert_embeddings, dimensions, reverse):
    embeddings = colbert_embeddings[::-1] if reverse else colbert_embeddings
    message = f"matryoshka_dim={dimensions} is not in a valid range. Select between 1 and 8."
    with pytest.raises(MatryoshkaDimError, match=re.escape(message)):
        matryososka_slice(embeddings, dimensions)
