from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.audio_embedding_input import AudioEmbeddingInput
from ...models.http_validation_error import HTTPValidationError
from ...models.open_ai_embedding_result import OpenAIEmbeddingResult
from ...types import Response


def _get_kwargs(
    *,
    body: AudioEmbeddingInput,
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "post",
        "url": "/embeddings_audio",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = OpenAIEmbeddingResult.from_dict(response.json())

        return response_200
    if response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AudioEmbeddingInput,
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Deprecated: Use `embeddings` with `modality` set to `audio`

     Encode Embeddings from Audio files

    Supports URLs of Audios and Base64-encoded Audios

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_audio\",
        json={
            \"model\":\"laion/larger_clap_general\",
            \"input\": [
                \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/l
    ibs/infinity_emb/tests/data/audio/beep.wav\",
                \"data:audio/wav;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDAUDIO\"
            ]
        })
    ```

    Args:
        body (AudioEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AudioEmbeddingInput,
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Deprecated: Use `embeddings` with `modality` set to `audio`

     Encode Embeddings from Audio files

    Supports URLs of Audios and Base64-encoded Audios

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_audio\",
        json={
            \"model\":\"laion/larger_clap_general\",
            \"input\": [
                \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/l
    ibs/infinity_emb/tests/data/audio/beep.wav\",
                \"data:audio/wav;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDAUDIO\"
            ]
        })
    ```

    Args:
        body (AudioEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OpenAIEmbeddingResult]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AudioEmbeddingInput,
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Deprecated: Use `embeddings` with `modality` set to `audio`

     Encode Embeddings from Audio files

    Supports URLs of Audios and Base64-encoded Audios

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_audio\",
        json={
            \"model\":\"laion/larger_clap_general\",
            \"input\": [
                \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/l
    ibs/infinity_emb/tests/data/audio/beep.wav\",
                \"data:audio/wav;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDAUDIO\"
            ]
        })
    ```

    Args:
        body (AudioEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: AudioEmbeddingInput,
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Deprecated: Use `embeddings` with `modality` set to `audio`

     Encode Embeddings from Audio files

    Supports URLs of Audios and Base64-encoded Audios

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_audio\",
        json={
            \"model\":\"laion/larger_clap_general\",
            \"input\": [
                \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/l
    ibs/infinity_emb/tests/data/audio/beep.wav\",
                \"data:audio/wav;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDAUDIO\"
            ]
        })
    ```

    Args:
        body (AudioEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OpenAIEmbeddingResult]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
