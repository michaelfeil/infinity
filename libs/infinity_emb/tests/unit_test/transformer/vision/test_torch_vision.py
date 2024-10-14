import numpy as np
import pytest
import torch
from PIL import Image  # type: ignore
from transformers import CLIPModel, CLIPProcessor  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.vision.torch_vision import ClipLikeModel


def test_clip_like_model(image_sample):
    model_name = pytest.DEFAULT_IMAGE_MODEL
    model = ClipLikeModel(
        engine_args=EngineArgs(model_name_or_path=model_name, dtype="auto")
    )
    image = Image.open(image_sample[0].raw)

    inputs = [
        "a photo of a cat",
        image,
        "a photo of a dog",
        image,
    ]
    embeddings = model.encode_post(model.encode_core(model.encode_pre(inputs)))

    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], np.ndarray)
    assert len(embeddings) == len(inputs)
    embeddings = torch.tensor(embeddings)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    inputs_clip = processor(
        text=["a photo of a cat"], images=[image], return_tensors="pt", padding=True
    )

    outputs = model(**inputs_clip)

    torch.testing.assert_close(
        outputs.text_embeds[0], embeddings[0], check_dtype=False, rtol=0, atol=1e-3
    )
    torch.testing.assert_close(
        outputs.image_embeds[0], embeddings[3], check_dtype=False, rtol=0, atol=1e-3
    )
    torch.testing.assert_close(embeddings[1], embeddings[3])
