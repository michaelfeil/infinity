# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import io
from typing import List, Union

from infinity_emb._optional_imports import CHECK_PIL, CHECK_REQUESTS, CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioCorruption,
    AudioSingle,
    ImageClassType,
    ImageCorruption,
    ImageSingle,
)

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore

if CHECK_REQUESTS.is_available:
    import requests  # type: ignore
if CHECK_SOUNDFILE.is_available:
    import soundfile as sf  # type: ignore


def resolve_from_img_obj(img_obj: "ImageClassType") -> ImageSingle:
    """Resolve an image from a ImageClassType Object."""
    return ImageSingle(image=img_obj)


def resolve_from_img_url(img_url: str) -> ImageSingle:
    """Resolve an image from an URL."""
    try:
        downloaded_img = requests.get(img_url, stream=True).raw
    except Exception as e:
        raise ImageCorruption(f"error downloading image from url: {e}")

    try:
        return ImageSingle(image=Image.open(downloaded_img))
    except Exception as e:
        raise ImageCorruption(f"error opening image from url: {e}")


def resolve_image(img: Union[str, "ImageClassType"]) -> ImageSingle:
    """Resolve a single image."""
    if isinstance(img, Image.Image):
        return resolve_from_img_obj(img)
    elif isinstance(img, str):
        return resolve_from_img_url(img)
    else:
        raise ValueError(
            f"Invalid image type: {img} is neither str nor ImageClassType object"
        )


def resolve_images(images: List[Union[str, "ImageClassType"]]) -> List[ImageSingle]:
    """Resolve images from URLs or ImageClassType Objects using multithreading."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_REQUESTS.mark_required()
    CHECK_PIL.mark_required()

    resolved_imgs = []
    for img in images:
        try:
            resolved_imgs.append(resolve_image(img))
        except Exception as e:
            raise ImageCorruption(
                f"Failed to resolve image: {img}.\nError msg: {str(e)}"
            )

    return resolved_imgs


def resolve_audios(audio_urls: list[str]) -> list[AudioSingle]:
    """Resolve audios from URLs."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_REQUESTS.mark_required()
    CHECK_SOUNDFILE.mark_required()

    try:
        downloaded = [requests.get(url, stream=True).content for url in audio_urls]
        downloaded_audio = [
            sf.read(
                io.BytesIO(raw_bytes),
            )[1]
            for raw_bytes in downloaded
        ]
    except Exception as e:
        raise AudioCorruption(f"Error downloading audios: {e}")
    try:
        return [AudioSingle(audio=audio) for audio in downloaded_audio]
    except Exception as e:
        raise AudioCorruption(f"Error opening audios: {e}")
