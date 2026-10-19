# Laya on Baseten with Truss

Deploy Infinity v3's native LibTorch Laya server using a prebuilt NVIDIA image
and a pinned copy of
[`michaelfeil/laya-typed-decisions`](https://huggingface.co/michaelfeil/laya-typed-decisions/tree/2741340228fc0c982ae9dd63b34dcd5a9e7ce3ce).

The [config](config.yaml) requests two L4 GPUs, eight CPUs, 4 GiB of host memory,
32 concurrent requests, a 1800-second startup allowance, and 10-second restart
and stop-traffic health thresholds (the
[platform minimums](https://docs.baseten.co/development/model/health-checks#parameters)).
The server binds to
`0.0.0.0:8000` and explicitly creates data-parallel replicas on CUDA devices
`0,1`. Each GPU loads its own model copy; this does not shard the model.
Adjust host memory and concurrency to fit your traffic after measuring the
deployment.

Weights are mounted at `/models/laya`.
Infinity detects the Laya protocol from `rl_agent_config.json` and exposes
typed decisions at `/v1/systemone`. The public API model name is `laya`.
The checkpoint is public and ungated, so this example does not require a
Hugging Face secret.

## Deploy

Install [Truss](https://github.com/basetenlabs/truss), obtain a
[Baseten API key](https://app.baseten.co/settings/account/api_keys), and enable
[no-build custom servers](https://docs.baseten.co/development/model/custom-server#no-build-deployment)
for your Baseten organization. From the repository root:

```sh
python -m pip install --upgrade truss
cd infra/baseten/examples/laya
truss push
```

Enter your Baseten API key when prompted. Current Truss versions create
published deployments by default; do not use `--watch` (development mode)
with this no-build config. The image remains pinned
to `nvidia-3.0.0-alpha.3`; change `base_image.image` to choose another release.

## Call the model

Set the model ID shown by the deployment command and your API key:

```sh
export BASETEN_MODEL_ID="YOUR_MODEL_ID"
export BASETEN_API_KEY="YOUR_API_KEY"

curl --fail-with-body \
  "https://model-${BASETEN_MODEL_ID}.api.baseten.co/environments/production/sync/v1/systemone" \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  -H "Content-Type: application/json" \
  --data-binary @request.json
```

The [sample request](request.json) asks for a team choice, an urgency score,
and a `noul` truth assessment using the repository's Laya reference fixture.
Responses contain typed decision results, rather than embedding vectors.

With `no_build: true`, Baseten preserves server paths: use
`/sync/v1/systemone`, rather than `/predict`. `predict_endpoint` documents
the inference route but does not remap it. Health probes use `/health`.
See the [custom-server routing contract](https://docs.baseten.co/development/model/custom-server#routing).

Infinity v3's [licensing guide](../../../../libs/infinity-v3/LICENSING.md)
and the checkpoint's own license apply to their respective components.
