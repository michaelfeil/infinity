# API reference

Open `/docs` on your running Infinity v3 server for interactive Swagger UI.
Its `/api-doc/openapi.json` describes the endpoints and types in that exact build.

The [checked-in OpenAPI schema](docs/openapi.json) tracks this documentation's
source revision. Use the running server's schema when integrating a deployed version.

Common endpoints include `/v1/embeddings`, `/rerank`, `/v1/classify` and
`/v1/systemone`. Tasks and modalities depend on the loaded checkpoint.
See [supported models](docs/libtorch-model-coverage.md),
[decision inputs](docs/cli.md#structured-decision-inputs) and
[embedding response formats](docs/embedding-tensors.md).
