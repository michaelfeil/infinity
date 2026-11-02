import io

import numpy as np
import pytest
import soundfile as sf  # type: ignore
import torch
from transformers import ClapModel, ClapProcessor  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.audio.torch import TorchAudioModel


def test_clap_like_model(audio_sample):
    model_name = pytest.DEFAULT_AUDIO_MODEL
    model = TorchAudioModel(engine_args=EngineArgs(model_name_or_path=model_name))
    raw_bytes = audio_sample[0].content
    data, samplerate = sf.read(io.BytesIO(raw_bytes))

    assert samplerate == model.sampling_rate
    inputs = ["a sound of a cat", data, "a sound of a cat", data]

    embeddings = model.encode_post(model.encode_core(model.encode_pre(inputs)))

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], np.ndarray)
    assert len(embeddings) == len(inputs)
    embeddings = torch.tensor(embeddings)

    model = ClapModel.from_pretrained(model_name)
    processor = ClapProcessor.from_pretrained(model_name)

    inputs_clap = processor(
        text=["a sound of a cat"],
        audios=[data],
        return_tensors="pt",
        padding=True,
        sampling_rate=48000,
    )

    outputs = model(**inputs_clap)

    torch.testing.assert_close(
        outputs.text_embeds[0], embeddings[0], check_dtype=False, rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        outputs.audio_embeds[0], embeddings[1], check_dtype=False, rtol=0, atol=1e-3
    )

    torch.testing.assert_close(embeddings[0], embeddings[2])
