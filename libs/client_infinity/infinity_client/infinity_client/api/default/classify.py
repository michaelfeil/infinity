from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.classify_input import ClassifyInput
from ...models.classify_result import ClassifyResult
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: ClassifyInput,
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "post",
        "url": "/classify",
    }

    _body = body.to_dict()

    _kwargs["json"] = _body
    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ClassifyResult, HTTPValidationError]]:
    if response.status_code == HTTPStatus.OK:
        response_200 = ClassifyResult.from_dict(response.json())

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
) -> Response[Union[ClassifyResult, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ClassifyInput,
) -> Response[Union[ClassifyResult, HTTPValidationError]]:
    r"""Classify

     Score or Classify Sentiments

    ```python
    import requests
    requests.post(\"http://..:7997/classify\",
        json={\"model\":\"SamLowe/roberta-base-go_emotions\",\"input\":[\"I am not having a great
    day.\"]})
    ```

    Args:
        body (ClassifyInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ClassifyResult, HTTPValidationError]]
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
    body: ClassifyInput,
) -> Optional[Union[ClassifyResult, HTTPValidationError]]:
    r"""Classify

     Score or Classify Sentiments

    ```python
    import requests
    requests.post(\"http://..:7997/classify\",
        json={\"model\":\"SamLowe/roberta-base-go_emotions\",\"input\":[\"I am not having a great
    day.\"]})
    ```

    Args:
        body (ClassifyInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ClassifyResult, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ClassifyInput,
) -> Response[Union[ClassifyResult, HTTPValidationError]]:
    r"""Classify

     Score or Classify Sentiments

    ```python
    import requests
    requests.post(\"http://..:7997/classify\",
        json={\"model\":\"SamLowe/roberta-base-go_emotions\",\"input\":[\"I am not having a great
    day.\"]})
    ```

    Args:
        body (ClassifyInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ClassifyResult, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ClassifyInput,
) -> Optional[Union[ClassifyResult, HTTPValidationError]]:
    r"""Classify

     Score or Classify Sentiments

    ```python
    import requests
    requests.post(\"http://..:7997/classify\",
        json={\"model\":\"SamLowe/roberta-base-go_emotions\",\"input\":[\"I am not having a great
    day.\"]})
    ```

    Args:
        body (ClassifyInput):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ClassifyResult, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
