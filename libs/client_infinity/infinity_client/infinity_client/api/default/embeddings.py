from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.open_ai_embedding_input import OpenAIEmbeddingInput
from ...models.open_ai_embedding_result import OpenAIEmbeddingResult
from ...types import Response


def _get_kwargs(
    *,
    body: OpenAIEmbeddingInput,
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "post",
        "url": "/embeddings",
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
    body: OpenAIEmbeddingInput,
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"BAAI/bge-small-en-v1.5\",\"input\":[\"A sentence to encode.\"]})

    Args:
        body (OpenAIEmbeddingInput):

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
    body: OpenAIEmbeddingInput,
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"BAAI/bge-small-en-v1.5\",\"input\":[\"A sentence to encode.\"]})

    Args:
        body (OpenAIEmbeddingInput):

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
    body: OpenAIEmbeddingInput,
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"BAAI/bge-small-en-v1.5\",\"input\":[\"A sentence to encode.\"]})

    Args:
        body (OpenAIEmbeddingInput):

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
    body: OpenAIEmbeddingInput,
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"BAAI/bge-small-en-v1.5\",\"input\":[\"A sentence to encode.\"]})

    Args:
        body (OpenAIEmbeddingInput):

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
