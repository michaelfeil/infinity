# Baseten: Infinity Embedding Server Truss

This is a [Truss](https://truss.baseten.co/) to deploy [infinity embedding server](https://github.com/michaelfeil/infinity), a high-throughput, low-latency REST API server for serving vector embeddings.

## Deployment

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. [Required for gated/private models] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`.

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd custom-server/infinity-embedding-server
```

With `infinity-embedding-server` as your working directory, you can deploy the model with the following command, paste your Baseten API key if prompted.

```sh
truss push --publish --trusted
```

## Call your model

### curl

```bash
curl -X POST https://model-xxx.api.baseten.co/development/predict \
        -H "Authorization: Api-Key YOUR_API_KEY" \
        -d '{"input": "text string"}'
```

### request python library

```python
import requests

resp = requests.post(
    "https://model-xxx.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json={"input": "text string"},
)

print(resp.json())
```

### openai python SDK

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["YOUR_API_KEY"],
    base_url="https://bridge.baseten.co/v1/direct"
)

model_id = "xxx"
deployment_id = "xxx"

response = client.embeddings.create(
    input="text string",
    model="BAAI/bge-small-en-v1.5",
    extra_body={
        "baseten": {
            "model_id": model_id,
            "deployment_id": deployment_id
        }
    }
)

print(response.data[0].embedding)
```

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.