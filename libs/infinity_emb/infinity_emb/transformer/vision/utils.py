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


def resolve_audio(audio: Union[str, bytes]) -> AudioSingle:
    if isinstance(audio, bytes):
        try:
            audio_bytes = io.BytesIO(audio)
        except Exception as e:
            raise AudioCorruption(f"Error opening audio: {e}")
    else:
        try:
            downloaded = requests.get(audio, stream=True).content
            audio_bytes = io.BytesIO(downloaded)
        except Exception as e:
            raise AudioCorruption(f"Error downloading audio.\nError msg: {str(e)}")

    try:
        data, rate = sf.read(audio_bytes)
        return AudioSingle(audio=data, sampling_rate=rate)
    except Exception as e:
        raise AudioCorruption(f"Error opening audio: {e}.\nError msg: {str(e)}")


def resolve_audios(
    audio_urls: list[Union[str, bytes]], allowed_sampling_rate: int
) -> list[AudioSingle]:
    """Resolve audios from URLs."""
    CHECK_REQUESTS.mark_required()
    CHECK_SOUNDFILE.mark_required()

    resolved_audios: list[AudioSingle] = []
    for audio in audio_urls:
        try:
            audio_single = resolve_audio(audio)
            resolved_audios.append(audio_single)
        except Exception as e:
            raise AudioCorruption(f"Failed to resolve image: {e}")
    if not (
        all(
            resolved_audios[0].sampling_rate == audio.sampling_rate
            for audio in resolved_audios
        )
    ):
        raise AudioCorruption(
            f"Audio sample rates in one batch need to be consistent for vectorized processing, your requests holds rates of {[a.sampling_rate for a in resolved_audios]}Mhz."
        )
    return resolved_audios
