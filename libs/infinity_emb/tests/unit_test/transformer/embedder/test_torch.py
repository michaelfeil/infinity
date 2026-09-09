import numpy as np
import pytest
import torch

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.embedder.sentence_transformer import (
    SentenceTransformerPatched,
)

try:
    from fastembed import TextEmbedding  # type: ignore

    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False


@pytest.mark.skipif(not FASTEMBED_AVAILABLE, reason="fastembed not available")
def test_sentence_transformer_equals_fastembed(
    text=(
        "This is a test sentence for the sentence transformer and fastembed. The PyTorch API of nested tensors is in prototype stage and will change in the near future."
    ),
    model_name="BAAI/bge-small-en-v1.5",
) -> None:
    model_st = SentenceTransformerPatched(engine_args=EngineArgs(model_name_or_path=model_name))
    model_fastembed = TextEmbedding(model_name)

    embedding_st = model_st.encode_post(model_st.encode_core(model_st.encode_pre([text])))
    embedding_fast = np.array(list(model_fastembed.embed(documents=[text])))

    assert embedding_fast.shape == embedding_st.shape
    assert np.allclose(embedding_st[0], embedding_fast[0], atol=1e-3)
    # cosine similarity
    sim = torch.nn.functional.cosine_similarity(
        torch.tensor(embedding_st), torch.tensor(embedding_fast)
    )
    assert sim > 0.99


def test_sentence_transformer_qwen3_embedding(
    model_name="Qwen/Qwen3-Embedding-0.6B",
) -> None:
    """Qwen3 (decoder-only, last-token pooling) requires transformers>=4.51.0.

    Expected values are taken from the model card of the model,
    https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """
    model = SentenceTransformerPatched(
        engine_args=EngineArgs(
            model_name_or_path=model_name,
            device="cpu",
            bettertransformer=False,
            model_warmup=False,
        )
    )
    instruction = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    queries = [
        f"{instruction}What is the capital of China?",
        f"{instruction}Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]
    embeddings = model.encode_post(model.encode_core(model.encode_pre(queries + documents)))

    assert embeddings.shape == (4, 1024)
    # embeddings are l2-normalized
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4)

    similarity = embeddings[:2] @ embeddings[2:].T
    expected = np.array([[0.7646, 0.1414], [0.1355, 0.6000]])
    assert np.allclose(similarity, expected, atol=2e-2), similarity
