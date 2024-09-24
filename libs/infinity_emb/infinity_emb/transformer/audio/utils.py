import io
from typing import Union

import requests

from infinity_emb._optional_imports import CHECK_REQUESTS, CHECK_SOUNDFILE
from infinity_emb.primitives import (
    AudioCorruption,
    AudioSingle,
)

if CHECK_REQUESTS.is_available:
    import requests  # type: ignore

if CHECK_SOUNDFILE.is_available:
    import soundfile as sf  # type: ignore


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
            raise AudioCorruption(f"Failed to resolve audio: {e}")
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
