import asyncio
import random
import time

import numpy as np
import pytest


class Helpers:
    @staticmethod
    async def util_batch_embedding(
        client,
        sts_bechmark_dataset,
        model_base,
        prefix: str,
        model_name: str,
        batch_size: int,
        downsample: int = 2,
        decimal=3,
    ):
        sentences = []
        for d in sts_bechmark_dataset:
            for item in d:
                sentences.append(item.texts[0])
        random.shuffle(sentences)
        sentences = sentences[::downsample]
        # sentences = sentences[:batch_size*2]
        dummy_sentences = ["test" * 512] * batch_size

        async def _post_batch(inputs):
            return await client.post(
                f"{prefix}/embeddings", json=dict(input=inputs, model=model_name)
            )

        response = await _post_batch(inputs=dummy_sentences)

        _request_size = int(batch_size * 1.5)
        tasks = [
            _post_batch(inputs=sentences[sl : sl + _request_size])
            for sl in range(0, len(sentences), _request_size)
        ]
        start = time.perf_counter()
        _responses = await asyncio.gather(*tasks)
        end = time.perf_counter()
        time_api = end - start

        responses = []
        for response in _responses:
            responses.extend(response.json()["data"])
        for i in range(len(responses)):
            responses[i] = responses[i]["embedding"]

        model_base.encode(
            dummy_sentences,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        start = time.perf_counter()
        encodings = model_base.encode(sentences, batch_size=batch_size).tolist()
        end = time.perf_counter()
        time_st = end - start

        responses = np.array(responses)  # type: ignore
        encodings = np.array(encodings)

        for r, e in zip(responses, encodings):
            cosine_sim = np.dot(r, e) / (np.linalg.norm(e) * np.linalg.norm(r))
            assert cosine_sim > 0.94
        np.testing.assert_almost_equal(
            np.array(responses), np.array(encodings), decimal=decimal
        )
        assert time_api / time_st < 2.5

    @staticmethod
    async def embedding_verify(client, model_base, prefix, model_name, decimal=3):
        possible_inputs = [
            ["This is a test sentence."],
            ["This is a test sentence.", "This is another test sentence."],
        ]

        for inp in possible_inputs:
            response = await client.post(
                f"{prefix}/embeddings", json=dict(input=inp, model=model_name)
            )
            assert (
                response.status_code == 200
            ), f"{response.status_code}, {response.text}"
            rdata = response.json()
            assert "data" in rdata and isinstance(rdata["data"], list)
            assert all("embedding" in d for d in rdata["data"])
            assert len(rdata["data"]) == len(inp)

            want_embeddings = model_base.encode(inp)

            for embedding, st_embedding in zip(rdata["data"], want_embeddings):
                np.testing.assert_almost_equal(
                    embedding["embedding"], st_embedding, decimal=decimal
                )


@pytest.fixture
def helpers():
    return Helpers
