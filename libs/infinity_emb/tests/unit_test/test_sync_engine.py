from uuid import uuid4

from infinity_emb import EngineArgs, SyncEngineArray


def test_sync_engine():
    model = str(uuid4())
    s_eng_array = SyncEngineArray(
        [
            EngineArgs(
                model_name_or_path=model,
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
            EngineArgs(
                model_name_or_path=str(uuid4()),
                device="cpu",
                engine="debugengine",
                model_warmup=False,
            ),
        ]
    )

    future_result1 = s_eng_array.embed(model=model, sentences=["Hello world!"])
    future_result2 = s_eng_array.embed(model=model, sentences=["Hello world!"])
    embedding = future_result1.result()
    embedding2 = future_result2.result()

    assert (embedding[0][0] == embedding2[0][0]).all()
    s_eng_array.stop()


if __name__ == "__main__":
    test_sync_engine()
