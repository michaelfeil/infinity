import io

import numpy as np
import requests  # type: ignore
import soundfile as sf  # type: ignore
import torch
from transformers import ClapModel, ClapProcessor  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.audio.torch import ClapLikeModel


def test_clap_like_model():
    model_name = "laion/clap-htsat-unfused"
    model = ClapLikeModel(
        engine_args=EngineArgs(model_name_or_path=model_name, dtype="float32")
    )
    url = "https://github.com/wirthual/infinity/raw/b849258a5d60ba79f1c600cbca9c4ea77349876d/libs/infinity_emb/tests/data/audio/COMTran_Aerospacebeep1(ID2380)_BSB.wav"
    raw_bytes = requests.get(url, stream=True).content
    data, samplerate = sf.read(io.BytesIO(raw_bytes))

    assert samplerate == 48000
    inputs = ["a sound of a cat", data, "a sound of a cat", data]

    embeddings = model.encode_post(
        model.encode_core(model.encode_pre(inputs, sample_rate=samplerate))
    )

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], np.ndarray)
    assert len(embeddings) == len(inputs)
    embeddings = torch.tensor(embeddings)

    model = ClapModel.from_pretrained(model_name)
    max_length = model.config.text_config.max_length
    processor = ClapProcessor.from_pretrained(model_name)

    inputs_clap = processor(
        text=["a sound of a cat"],
        audios=[data],
        return_tensors="pt",
        padding="max_length",
        sampling_rate=48000,
        max_length=max_length,
    )

    outputs = model(**inputs_clap)

    torch.testing.assert_close(
        outputs.text_embeds[0], embeddings[0], check_dtype=False, rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        outputs.audio_embeds[0], embeddings[1], check_dtype=False, rtol=0, atol=1e-1
    )

    torch.testing.assert_close(embeddings[0], embeddings[2])
