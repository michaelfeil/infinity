# Deployment

### Docker: Launch the CLI using a pre-built docker container (recommended)
Instead of installing the CLI via pip, you may also use docker to run `michaelf34/infinity`. 
Make sure you mount your accelerator, i.e. install nvidia-docker and activate with `--gpus all`.

```bash
port=7997
model1=michaelfeil/bge-small-en-v1.5
model2=mixedbread-ai/mxbai-rerank-xsmall-v1
volume=$PWD/data

docker run -it --gpus all \
 -v $volume:/app/.cache \
 -p $port:$port \
 michaelf34/infinity:latest \
 v2 \
 --model-id $model1 \
 --model-id $model2 \
 --port $port
```
The cache path at inside the docker container is set by the environment variable `HF_HOME`.

## Runpod.io - Serverless
There is a dedicated guide on how deploy via Runpod Serverless.
https://github.com/runpod-workers/worker-infinity-text-embeddings/

## Modal Labs

A deployment example for usage within are located at repo, including a Github Actions Pipeline.
[michaelfeil/infinity/tree/main/infra/modal](https://github.com/michaelfeil/infinity/tree/88f57af6ffa98b74e3e1332efe013357e465f010/infra/modal)
Beyond, there is a sponsored endpoint at https://infinity.modal.michaelfeil.eu .

## Bento - BentoInfinity
Example repo for deployment via Bento: https://github.com/bentoml/BentoInfinity

## dstack
dstack allows you to provision a VM instance on the cloud of your choice.
Write a service configuration file as below for the deployment of `BAAI/bge-small-en-v1.5` model wrapped in Infinity.

```yaml
type: service

image: michaelf34/infinity:latest
env:
  - MODEL_ID=BAAI/bge-small-en-v1.5
commands:
  - infinity_emb v2 --model-id $MODEL_ID --port 80
port: 80
```

To deploy the service, execute the following dstack command. A prompt will guide you through selecting the desired VM instance for deploying Infinity.

```shell
dstack run . -f infinity/serve.dstack.yml --gpu 16GB
```


## Docker with offline mode / models with custom pip packages

If you want to run infinity in a location without internet access, you can pre-download the model into the dockerfile.
This is also the advised route to go, if you want to use infinity with models that require additional packages such as 
`nomic-ai/nomic-embed-text-v1.5`.

```bash
# clone the repo
git clone https://github.com/michaelfeil/infinity
git checkout tags/0.0.52
cd libs/infinity_emb
# build download stage using docker buildx buildkit.
docker buildx build --target=production-with-download \
--build-arg MODEL_NAME=michaelfeil/bge-small-en-v1.5 --build-arg ENGINE=torch \
-f Dockerfile -t infinity-model-small .
```
You can also set an argument `EXTRA_PACKAGES` if you require to install any extra packages.  `--build-arg EXTRA_PACKAGES="torch_geometric"` 

Rename and push it to your internal docker registry. 

```bash
docker tag infinity-model-small  myregistryhost:5000/myinfinity/infinity:0.0.52-small
docker push myregistryhost:5000/myinfinity/infinity:0.0.52-small
```

Note: You can also save a dockerfile direclty as `.tar`.
This might come in handy if you do not have a shared internal docker registry in your nuclear facility, but still want to leverage the latest semantic search.
https://docs.docker.com/reference/cli/docker/image/save/.

