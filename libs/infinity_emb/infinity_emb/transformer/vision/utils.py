# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import io

from infinity_emb._optional_imports import CHECK_PIL, CHECK_REQUESTS, CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioCorruption,
    AudioSingle,
    ImageCorruption,
    ImageSingle,
)

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore
if CHECK_REQUESTS.is_available:
    import requests  # type: ignore
if CHECK_SOUNDFILE.is_available:
    import soundfile as sf  # type: ignore


def resolve_images(image_urls: list[str]) -> list[ImageSingle]:
    """Resolve images from URLs."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_REQUESTS.mark_required()
    CHECK_PIL.mark_required()

    try:
        downloaded = [requests.get(url, stream=True).raw for url in image_urls]
    except Exception as e:
        raise ImageCorruption(f"Error downloading images: {e}")
    try:
        return [ImageSingle(image=Image.open(content)) for content in downloaded]
    except Exception as e:
        raise ImageCorruption(f"Error opening images: {e}")


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
