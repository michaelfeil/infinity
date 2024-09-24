# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil

import asyncio
import io
from typing import List, Union

from infinity_emb._optional_imports import CHECK_AIOHTTP, CHECK_PIL, CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioCorruption,
    AudioSingle,
    ImageClassType,
    ImageCorruption,
    ImageSingle,
)

if CHECK_AIOHTTP.is_available:
    import aiohttp

if CHECK_PIL.is_available:
    from PIL import Image  # type: ignore

if CHECK_SOUNDFILE.is_available:
    import soundfile as sf  # type: ignore


def resolve_from_img_obj(img_obj: "ImageClassType") -> ImageSingle:
    """Resolve an image from a ImageClassType Object."""
    return ImageSingle(image=img_obj)


async def resolve_from_img_url(
    img_url: str, session: "aiohttp.ClientSession"
) -> ImageSingle:
    """Resolve an image from an URL."""
    try:
        # requests.get(img_url, stream=True).raw
        downloaded_img = await (await session.get(img_url)).read()
    except Exception as e:
        raise ImageCorruption(
            f"error opening an image in your request image from url: {e}"
        )

    try:
        img = Image.open(io.BytesIO(downloaded_img))
        if img.size[0] < 3 or img.size[1] < 3:
            # https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png
            raise ImageCorruption(
                f"An image in your request is too small for processing {img.size}"
            )
        return ImageSingle(image=img)
    except Exception as e:
        raise ImageCorruption(
            f"error opening the payload from an image in your request from url: {e}"
        )


async def resolve_image(
    img: Union[str, "ImageClassType"], session: "aiohttp.ClientSession"
) -> ImageSingle:
    """Resolve a single image."""
    if isinstance(img, Image.Image):
        return resolve_from_img_obj(img)
    elif isinstance(img, str):
        return await resolve_from_img_url(img, session=session)
    else:
        raise ValueError(
            f"Invalid image type: {img} is neither str nor ImageClassType object"
        )


async def resolve_images(
    images: List[Union[str, "ImageClassType"]]
) -> List[ImageSingle]:
    """Resolve images from URLs or ImageClassType Objects using multithreading."""
    # TODO: improve parallel requests, safety, error handling
    CHECK_AIOHTTP.mark_required()
    CHECK_PIL.mark_required()

    resolved_imgs = []

    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            resolved_imgs = await asyncio.gather(
                *[resolve_image(img, session) for img in images]
            )
    except Exception as e:
        raise ImageCorruption(
            f"Failed to resolve image: {images}.\nError msg: {str(e)}"
        )

    return resolved_imgs


async def resolve_audio(
    audio: Union[str, bytes],
    allowed_sampling_rate: int,
    session: "aiohttp.ClientSession",
) -> AudioSingle:
    if isinstance(audio, bytes):
        try:
            audio_bytes = io.BytesIO(audio)
        except Exception as e:
            raise AudioCorruption(f"Error opening audio: {e}")
    else:
        try:
            downloaded = await (await session.get(audio)).read()
            # downloaded = requests.get(audio, stream=True).content
            audio_bytes = io.BytesIO(downloaded)
        except Exception as e:
            raise AudioCorruption(f"Error downloading audio.\nError msg: {str(e)}")

    try:
        data, rate = sf.read(audio_bytes)
        if rate != allowed_sampling_rate:
            raise AudioCorruption(
                f"Audio sample rate is not {allowed_sampling_rate}Hz, it is {rate}Hz."
            )
        return AudioSingle(audio=data, sampling_rate=rate)
    except Exception as e:
        raise AudioCorruption(f"Error opening audio: {e}.\nError msg: {str(e)}")


async def resolve_audios(
    audio_urls: list[Union[str, bytes]], allowed_sampling_rate: int
) -> list[AudioSingle]:
    """Resolve audios from URLs."""
    CHECK_AIOHTTP.mark_required()
    CHECK_SOUNDFILE.mark_required()

    resolved_audios: list[AudioSingle] = []
    async with aiohttp.ClientSession(trust_env=True) as session:
        try:
            resolved_audios = await asyncio.gather(
                *[
                    resolve_audio(audio, allowed_sampling_rate, session)
                    for audio in audio_urls
                ]
            )
        except Exception as e:
            raise AudioCorruption(f"Failed to resolve audio: {e}")

    return resolved_audios
