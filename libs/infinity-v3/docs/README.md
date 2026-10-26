# Infinity v3 documentation

Infinity v3 is the Rust server in `libs/infinity-v3`. The existing Python server
and client are separate projects in the parent `libs` directory.

- [Build and development commands](../CONTRIBUTING.md#working-in-the-infinity-monorepo)
- [Running the server](cli.md)
- [Supported models and configuration by task](supported-models.md)
- [Performance and Radix MLP](performance.md)
- [Native LibTorch backend](libtorch-backend.md)
- [Native model coverage](libtorch-model-coverage.md)
- [Licensing and retained third-party notices](../LICENSING.md)

The running server serves Swagger UI at `/docs` and its generated OpenAPI schema
at `/api-doc/openapi.json`. [openapi.json](openapi.json) is the checked-in API
contract used by regression tests. The runtime schema describes the current
build and should be used when integrating a client.

JSON is the default response format, including for `Accept: */*`. Request
`Accept: application/msgpack` or `application/x-msgpack` explicitly for binary
responses. Explicit quality preferences are honored, while browser requests
for HTML keep Swagger UI's HTML and assets in their native formats.
