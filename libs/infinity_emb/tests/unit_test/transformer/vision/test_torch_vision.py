import numpy as np
import pytest
import torch
from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
from PIL import Image  # type: ignore
from transformers import CLIPModel, CLIPProcessor  # type: ignore

from infinity_emb.args import EngineArgs
from infinity_emb.transformer.vision.torch_vision import TIMM


def test_clip_like_model(image_sample):
    model_name = pytest.DEFAULT_IMAGE_MODEL
    model = TIMM(engine_args=EngineArgs(model_name_or_path=model_name, dtype="auto"))
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


@pytest.mark.parametrize("dtype", ["auto"])
def test_colpali(dtype, image_sample):
    model_name = pytest.DEFAULT_IMAGE_COLPALI_MODEL
    revision = "main"

    model = TIMM(
        engine_args=EngineArgs(
            model_name_or_path=model_name, dtype=dtype, revision=revision, device="cpu"
        )
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
    embeddings = [torch.tensor(t) for t in embeddings]

    model = None
    torch.testing.assert_close(embeddings[1], embeddings[3])
    if dtype == "auto":
        model = ColPali.from_pretrained(
            model_name,
        ).eval()

        processor = ColPaliProcessor.from_pretrained(model_name, revision=revision)

        # Your inputs
        images = [
            image,
            image,
            Image.new("RGB", (128, 128), color="white"),
            Image.new("RGB", (128, 128), color="black"),
        ]
        queries = [
            "a photo of a cat",
            "a photo of a dog",
        ]

        # Process the inputs
        batch_images = processor.process_images(images).to(model.device)
        batch_queries = processor.process_queries(queries).to(model.device)

        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**batch_images).to("cpu")
            query_embeddings = model(**batch_queries).to("cpu")
        torch.testing.assert_close(
            query_embeddings[0], embeddings[0], check_dtype=False, rtol=0, atol=5e-3
        )
        torch.testing.assert_close(
            image_embeddings[0].mean(0),
            embeddings[3].mean(0),
            check_dtype=False,
            rtol=0,
            atol=5e-3,
        )


if __name__ == "__main__":
    import requests

    test_colpali("int8", requests.get(pytest.DEFAULT_IMAGE_SAMPLE, stream=True))  # type: ignore
    test_colpali("auto", requests.get(pytest.DEFAULT_IMAGE_SAMPLE, stream=True))  # type: ignore
