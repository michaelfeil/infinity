import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.embedder.optimum import OptimumEmbedder


def test_embedder_optimum(size="large"):
    model = OptimumEmbedder(
        engine_args=EngineArgs(model_name_or_path=f"Xenova/bge-{size}-en-v1.5", device="cpu")
    )
    st_model = SentenceTransformer(model_name_or_path=f"BAAI/bge-{size}-en-v1.5", device="cpu")

    sentences = ["This is awesome.", "I am depressed."]

    encode_pre = model.encode_pre(sentences)
    encode_core = model.encode_core(encode_pre)
    embeds = model.encode_post(encode_core)

    embeds_orig = st_model.encode(sentences)

    assert len(embeds) == len(sentences)

    for r, e in zip(embeds, embeds_orig):
        cosine_sim = np.dot(r, e) / (np.linalg.norm(e) * np.linalg.norm(r))
        assert cosine_sim > 0.94
    np.testing.assert_allclose(embeds, embeds_orig, atol=0.25)


def test_embedder_optimum_openvino_cpu(size="large"):
    model = OptimumEmbedder(
        engine_args=EngineArgs(model_name_or_path=f"BAAI/bge-{size}-en-v1.5", device="openvino")
    )
    st_model = SentenceTransformer(model_name_or_path=f"BAAI/bge-{size}-en-v1.5", device="cpu")

    sentences = ["This is awesome.", "I am depressed."]

    encode_pre = model.encode_pre(sentences)
    encode_core = model.encode_core(encode_pre)
    embeds = model.encode_post(encode_core)

    embeds_orig = st_model.encode(sentences)

    assert len(embeds) == len(sentences)

    for r, e in zip(embeds, embeds_orig):
        cosine_sim = np.dot(r, e) / (np.linalg.norm(e) * np.linalg.norm(r))
        assert cosine_sim > 0.94
    np.testing.assert_allclose(embeds, embeds_orig, atol=0.25)

