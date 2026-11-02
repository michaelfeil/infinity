from http import HTTPStatus
from typing import Any, Dict, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.open_ai_embedding_input_audio import OpenAIEmbeddingInputAudio
from ...models.open_ai_embedding_input_image import OpenAIEmbeddingInputImage
from ...models.open_ai_embedding_input_text import OpenAIEmbeddingInputText
from ...models.open_ai_embedding_result import OpenAIEmbeddingResult
from ...types import Response


def _get_kwargs(
    *,
    body: Union["OpenAIEmbeddingInputAudio", "OpenAIEmbeddingInputImage", "OpenAIEmbeddingInputText"],
) -> Dict[str, Any]:
    headers: Dict[str, Any] = {}

    _kwargs: Dict[str, Any] = {
        "method": "post",
        "url": "/embeddings",
    }

    _body: Dict[str, Any]
    if isinstance(body, OpenAIEmbeddingInputText):
        _body = body.to_dict()
    elif isinstance(body, OpenAIEmbeddingInputAudio):
        _body = body.to_dict()
    else:
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
    body: Union["OpenAIEmbeddingInputAudio", "OpenAIEmbeddingInputImage", "OpenAIEmbeddingInputText"],
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings. Supports with multimodal inputs. Aligned with OpenAI Embeddings API.

    ## Running Text Embeddings
    ```python
    import requests, base64
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"openai/clip-vit-base-patch32\",\"input\":[\"Two cute cats.\"]})
    ```

    ## Running Image Embeddings
    ```python
    requests.post(\"http://..:7997/embeddings\",
        json={
            \"model\": \"openai/clip-vit-base-patch32\",
            \"encoding_format\": \"base64\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                # can also be base64 encoded
            ],
            # set extra modality to image to process as image
            \"modality\": \"image\"
    )
    ```

    ## Running Audio Embeddings
    ```python
    import requests, base64
    url = \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/in
    finity_emb/tests/data/audio/beep.wav\"

    def url_to_base64(url, modality = \"image\"):
        '''small helper to convert url to base64 without server requiring access to the url'''
        response = requests.get(url)
        response.raise_for_status()
        base64_encoded = base64.b64encode(response.content).decode('utf-8')
        mimetype = f\"{modality}/{url.split('.')[-1]}\"
        return f\"data:{mimetype};base64,{base64_encoded}\"

    requests.post(\"http://localhost:7997/embeddings\",
        json={
            \"model\": \"laion/larger_clap_general\",
            \"encoding_format\": \"float\",
            \"input\": [
                url, url_to_base64(url, \"audio\")
            ],
            # set extra modality to audio to process as audio
            \"modality\": \"audio\"
        }
    )
    ```

    ## Running via OpenAI Client
    ```python
    from openai import OpenAI # pip install openai==1.51.0
    client = OpenAI(base_url=\"http://localhost:7997/\")
    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[url_to_base64(url, \"audio\")],
        encoding_format=\"float\",
        extra_body={
            \"modality\": \"audio\"
        }
    )

    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[\"the sound of a beep\", \"the sound of a cat\"],
        encoding_format=\"base64\", # base64: optional high performance setting
        extra_body={
            \"modality\": \"text\"
        }
    )
    ```

    ### Hint: Run all the above models on one server:
    ```bash
    infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --model-id openai/clip-vit-base-patch32 --model-id
    laion/larger_clap_general
    ```

    Args:
        body (Union['OpenAIEmbeddingInputAudio', 'OpenAIEmbeddingInputImage',
            'OpenAIEmbeddingInputText']):

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
    body: Union["OpenAIEmbeddingInputAudio", "OpenAIEmbeddingInputImage", "OpenAIEmbeddingInputText"],
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings. Supports with multimodal inputs. Aligned with OpenAI Embeddings API.

    ## Running Text Embeddings
    ```python
    import requests, base64
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"openai/clip-vit-base-patch32\",\"input\":[\"Two cute cats.\"]})
    ```

    ## Running Image Embeddings
    ```python
    requests.post(\"http://..:7997/embeddings\",
        json={
            \"model\": \"openai/clip-vit-base-patch32\",
            \"encoding_format\": \"base64\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                # can also be base64 encoded
            ],
            # set extra modality to image to process as image
            \"modality\": \"image\"
    )
    ```

    ## Running Audio Embeddings
    ```python
    import requests, base64
    url = \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/in
    finity_emb/tests/data/audio/beep.wav\"

    def url_to_base64(url, modality = \"image\"):
        '''small helper to convert url to base64 without server requiring access to the url'''
        response = requests.get(url)
        response.raise_for_status()
        base64_encoded = base64.b64encode(response.content).decode('utf-8')
        mimetype = f\"{modality}/{url.split('.')[-1]}\"
        return f\"data:{mimetype};base64,{base64_encoded}\"

    requests.post(\"http://localhost:7997/embeddings\",
        json={
            \"model\": \"laion/larger_clap_general\",
            \"encoding_format\": \"float\",
            \"input\": [
                url, url_to_base64(url, \"audio\")
            ],
            # set extra modality to audio to process as audio
            \"modality\": \"audio\"
        }
    )
    ```

    ## Running via OpenAI Client
    ```python
    from openai import OpenAI # pip install openai==1.51.0
    client = OpenAI(base_url=\"http://localhost:7997/\")
    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[url_to_base64(url, \"audio\")],
        encoding_format=\"float\",
        extra_body={
            \"modality\": \"audio\"
        }
    )

    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[\"the sound of a beep\", \"the sound of a cat\"],
        encoding_format=\"base64\", # base64: optional high performance setting
        extra_body={
            \"modality\": \"text\"
        }
    )
    ```

    ### Hint: Run all the above models on one server:
    ```bash
    infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --model-id openai/clip-vit-base-patch32 --model-id
    laion/larger_clap_general
    ```

    Args:
        body (Union['OpenAIEmbeddingInputAudio', 'OpenAIEmbeddingInputImage',
            'OpenAIEmbeddingInputText']):

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
    body: Union["OpenAIEmbeddingInputAudio", "OpenAIEmbeddingInputImage", "OpenAIEmbeddingInputText"],
) -> Response[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings. Supports with multimodal inputs. Aligned with OpenAI Embeddings API.

    ## Running Text Embeddings
    ```python
    import requests, base64
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"openai/clip-vit-base-patch32\",\"input\":[\"Two cute cats.\"]})
    ```

    ## Running Image Embeddings
    ```python
    requests.post(\"http://..:7997/embeddings\",
        json={
            \"model\": \"openai/clip-vit-base-patch32\",
            \"encoding_format\": \"base64\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                # can also be base64 encoded
            ],
            # set extra modality to image to process as image
            \"modality\": \"image\"
    )
    ```

    ## Running Audio Embeddings
    ```python
    import requests, base64
    url = \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/in
    finity_emb/tests/data/audio/beep.wav\"

    def url_to_base64(url, modality = \"image\"):
        '''small helper to convert url to base64 without server requiring access to the url'''
        response = requests.get(url)
        response.raise_for_status()
        base64_encoded = base64.b64encode(response.content).decode('utf-8')
        mimetype = f\"{modality}/{url.split('.')[-1]}\"
        return f\"data:{mimetype};base64,{base64_encoded}\"

    requests.post(\"http://localhost:7997/embeddings\",
        json={
            \"model\": \"laion/larger_clap_general\",
            \"encoding_format\": \"float\",
            \"input\": [
                url, url_to_base64(url, \"audio\")
            ],
            # set extra modality to audio to process as audio
            \"modality\": \"audio\"
        }
    )
    ```

    ## Running via OpenAI Client
    ```python
    from openai import OpenAI # pip install openai==1.51.0
    client = OpenAI(base_url=\"http://localhost:7997/\")
    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[url_to_base64(url, \"audio\")],
        encoding_format=\"float\",
        extra_body={
            \"modality\": \"audio\"
        }
    )

    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[\"the sound of a beep\", \"the sound of a cat\"],
        encoding_format=\"base64\", # base64: optional high performance setting
        extra_body={
            \"modality\": \"text\"
        }
    )
    ```

    ### Hint: Run all the above models on one server:
    ```bash
    infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --model-id openai/clip-vit-base-patch32 --model-id
    laion/larger_clap_general
    ```

    Args:
        body (Union['OpenAIEmbeddingInputAudio', 'OpenAIEmbeddingInputImage',
            'OpenAIEmbeddingInputText']):

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
    body: Union["OpenAIEmbeddingInputAudio", "OpenAIEmbeddingInputImage", "OpenAIEmbeddingInputText"],
) -> Optional[Union[HTTPValidationError, OpenAIEmbeddingResult]]:
    r"""Embeddings

     Encode Embeddings. Supports with multimodal inputs. Aligned with OpenAI Embeddings API.

    ## Running Text Embeddings
    ```python
    import requests, base64
    requests.post(\"http://..:7997/embeddings\",
        json={\"model\":\"openai/clip-vit-base-patch32\",\"input\":[\"Two cute cats.\"]})
    ```

    ## Running Image Embeddings
    ```python
    requests.post(\"http://..:7997/embeddings\",
        json={
            \"model\": \"openai/clip-vit-base-patch32\",
            \"encoding_format\": \"base64\",
            \"input\": [
                \"http://images.cocodataset.org/val2017/000000039769.jpg\",
                # can also be base64 encoded
            ],
            # set extra modality to image to process as image
            \"modality\": \"image\"
    )
    ```

    ## Running Audio Embeddings
    ```python
    import requests, base64
    url = \"https://github.com/michaelfeil/infinity/raw/3b72eb7c14bae06e68ddd07c1f23fe0bf403f220/libs/in
    finity_emb/tests/data/audio/beep.wav\"

    def url_to_base64(url, modality = \"image\"):
        '''small helper to convert url to base64 without server requiring access to the url'''
        response = requests.get(url)
        response.raise_for_status()
        base64_encoded = base64.b64encode(response.content).decode('utf-8')
        mimetype = f\"{modality}/{url.split('.')[-1]}\"
        return f\"data:{mimetype};base64,{base64_encoded}\"

    requests.post(\"http://localhost:7997/embeddings\",
        json={
            \"model\": \"laion/larger_clap_general\",
            \"encoding_format\": \"float\",
            \"input\": [
                url, url_to_base64(url, \"audio\")
            ],
            # set extra modality to audio to process as audio
            \"modality\": \"audio\"
        }
    )
    ```

    ## Running via OpenAI Client
    ```python
    from openai import OpenAI # pip install openai==1.51.0
    client = OpenAI(base_url=\"http://localhost:7997/\")
    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[url_to_base64(url, \"audio\")],
        encoding_format=\"float\",
        extra_body={
            \"modality\": \"audio\"
        }
    )

    client.embeddings.create(
        model=\"laion/larger_clap_general\",
        input=[\"the sound of a beep\", \"the sound of a cat\"],
        encoding_format=\"base64\", # base64: optional high performance setting
        extra_body={
            \"modality\": \"text\"
        }
    )
    ```

    ### Hint: Run all the above models on one server:
    ```bash
    infinity_emb v2 --model-id BAAI/bge-small-en-v1.5 --model-id openai/clip-vit-base-patch32 --model-id
    laion/larger_clap_general
    ```

    Args:
        body (Union['OpenAIEmbeddingInputAudio', 'OpenAIEmbeddingInputImage',
            'OpenAIEmbeddingInputText']):

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
