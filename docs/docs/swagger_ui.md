# Swagger UI
Disclaimer: This is the current Swagger UI based on the main branch - which may differ from the Swagger UI of this release.
The Swagger UI and `openapi.json` will be available under `{url}:{port}/docs`, in this case `http://localhost:7997/docs`.

Image and audio inputs accept data URIs with either base64 or percent-encoded payloads.
Percent-encoded payloads preserve the original bytes, including non-UTF-8 image and audio data.
When using `DataURI.make` with text, the declared charset determines the encoded bytes;
the default is UTF-8.

Beyond, a test-deployment on Huggingface is deployed here: https://michaelfeil-infinity.hf.space

<swagger-ui src="https://raw.githubusercontent.com/michaelfeil/infinity/main/docs/assets/openapi.json"/>

