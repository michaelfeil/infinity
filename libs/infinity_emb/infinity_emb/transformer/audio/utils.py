import asyncio
import io
from typing import Union

from infinity_emb._optional_imports import CHECK_AIOHTTP, CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioCorruption,
    AudioSingle,
)

if CHECK_AIOHTTP.is_available:
    import aiohttp

if CHECK_SOUNDFILE.is_available:
    import soundfile as sf  # type: ignore


async def resolve_audio(
    audio: Union[str, bytes],
    allowed_sampling_rate: int,
    session: "aiohttp.ClientSession",
) -> AudioSingle:
    if isinstance(audio, bytes):
        try:
            audio_bytes = io.BytesIO(audio)
        except Exception as e:
            raise AudioCorruption(f"Error opening audio from bytes: {e}")
    else:
        try:
            downloaded = await (await session.get(audio)).read()
            #
            audio_bytes = io.BytesIO(downloaded)
        except Exception as e:
            raise AudioCorruption(
                f"Error downloading audio from {audio}. \nError msg: {str(e)}"
            )

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
