import timeit

import numpy as np
from fastembed.embedding import FlagEmbedding
from sentence_transformers import SentenceTransformer

model_name_or_path = "BAAI/bge-small-en-v1.5"

model_fast = FlagEmbedding(model_name_or_path)
model_st = SentenceTransformer(model_name_or_path)

sample_sentence = [f"{list(range(i))} " for i in range(64)]

got = np.stack(list(model_fast.embed(sample_sentence)))
want = model_st.encode(sample_sentence, normalize_embeddings=True)

# FAILS here Mismatched elements: 24384 / 24576 (99.2%)
np.testing.assert_almost_equal(got, want)

# 2.0177175840362906 vs 2.4251126241870224
print(
    timeit.timeit(lambda: list(model_fast.embed(sample_sentence)), number=10),
    "vs",
    timeit.timeit(
        lambda: model_st.encode(sample_sentence, normalize_embeddings=True), number=10
    ),
)
