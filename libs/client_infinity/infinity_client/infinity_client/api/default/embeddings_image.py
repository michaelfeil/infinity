from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ...client import AuthenticatedClient, Client
from ...types import Response, UNSET
from ... import errors

from ...models.http_validation_error import HTTPValidationError
from ...models.image_embedding_input import ImageEmbeddingInput
from ...models.open_ai_embedding_result import OpenAIEmbeddingResult
from typing import cast



def _get_kwargs(
    *,
    body: ImageEmbeddingInput,

) -> dict[str, Any]:
    headers: dict[str, Any] = {}


    

    

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/embeddings_image",
    }

    _kwargs["json"] = body.to_dict()


    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs



def _parse_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> HTTPValidationError | OpenAIEmbeddingResult | None:
    if response.status_code == 200:
        response_200 = OpenAIEmbeddingResult.from_dict(response.json())



        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())



        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(*, client: AuthenticatedClient | Client, response: httpx.Response) -> Response[HTTPValidationError | OpenAIEmbeddingResult]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ImageEmbeddingInput,

) -> Response[HTTPValidationError | OpenAIEmbeddingResult]:
    r""" Deprecated: Use `embeddings` with `modality` set to `image`

     Encode Embeddings from Image files

    Supports URLs of Images and Base64-encoded Images

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_image\",
        json={
            \"model\":\"openai/clip-vit-base-patch32\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                \"data:image/png;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDIMAGE\"
            ]
        })
    ```

    Args:
        body (ImageEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OpenAIEmbeddingResult]
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
    client: AuthenticatedClient | Client,
    body: ImageEmbeddingInput,

) -> HTTPValidationError | OpenAIEmbeddingResult | None:
    r""" Deprecated: Use `embeddings` with `modality` set to `image`

     Encode Embeddings from Image files

    Supports URLs of Images and Base64-encoded Images

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_image\",
        json={
            \"model\":\"openai/clip-vit-base-patch32\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                \"data:image/png;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDIMAGE\"
            ]
        })
    ```

    Args:
        body (ImageEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OpenAIEmbeddingResult
     """


    return sync_detailed(
        client=client,
body=body,

    ).parsed

async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ImageEmbeddingInput,

) -> Response[HTTPValidationError | OpenAIEmbeddingResult]:
    r""" Deprecated: Use `embeddings` with `modality` set to `image`

     Encode Embeddings from Image files

    Supports URLs of Images and Base64-encoded Images

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_image\",
        json={
            \"model\":\"openai/clip-vit-base-patch32\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                \"data:image/png;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDIMAGE\"
            ]
        })
    ```

    Args:
        body (ImageEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OpenAIEmbeddingResult]
     """


    kwargs = _get_kwargs(
        body=body,

    )

    response = await client.get_async_httpx_client().request(
        **kwargs
    )

    return _build_response(client=client, response=response)

async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ImageEmbeddingInput,

) -> HTTPValidationError | OpenAIEmbeddingResult | None:
    r""" Deprecated: Use `embeddings` with `modality` set to `image`

     Encode Embeddings from Image files

    Supports URLs of Images and Base64-encoded Images

    ```python
    import requests
    requests.post(\"http://..:7997/embeddings_image\",
        json={
            \"model\":\"openai/clip-vit-base-patch32\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                \"data:image/png;base64,iVBORw0KGgoDEMOoSAMPLEoENCODEDIMAGE\"
            ]
        })
    ```

    Args:
        body (ImageEmbeddingInput): LEGACY, DO NO LONGER UPDATE

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OpenAIEmbeddingResult
     """


    return (await asyncio_detailed(
        client=client,
body=body,

    )).parsed
