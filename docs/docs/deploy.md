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
The cache path inside the docker container is set by the environment variable `HF_HOME`.


### AMD Docker: Deploy on AMD Platform (MI200 Series and MI300 Series) 
#### Launch the CLI using a pre-built docker container (recommended) 

```bash
port=7997
model1=michaelfeil/bge-small-en-v1.5
model2=mixedbread-ai/mxbai-rerank-xsmall-v1
volume=$PWD/data

docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --network host \
  -v $volume:/app/.cache \
  -p $port:$port \
  michaelf34/infinity:latest-rocm \
  v2 \
  --model-id $model1 \
  --model-id $model2 \
  --port $port \
  --engine torch \
  --compile \
  --no-bettertransformer
```
The cache path at inside the docker container is set by the environment variable `HF_HOME`.



## Modal Labs

A deployment example for usage within are located at repo, including a Github Actions Pipeline.

The example is located at [michaelfeil/infinity/tree/main/infra/modal](https://github.com/michaelfeil/infinity/tree/c84b15acc35d02005e6f69080a5ed7b0e23d0019/infra/modal).

The GPU and Modal-powered endpoint via this Github Pipeline is free to try out at  [infinity.modal.michaelfeil.eu](https://infinity.modal.michaelfeil.eu), which is available at no cost.

## Runpod.io - Serverless
There is a dedicated guide on how deploy via Runpod Serverless. 
Find out how to deploy it via this Repo:
[github.com/runpod-workers/worker-infinity-text-embeddings](https://github.com/runpod-workers/worker-infinity-text-embeddings/) 

## Bento - BentoInfinity
Example repo for deployment via Bento: https://github.com/bentoml/BentoInfinity

## dstack

dstack allows you to provision a VM instance on the cloud of your choice.
Write a service configuration file as below for the deployment of `BAAI/bge-small-en-v1.5` model wrapped in Infinity.

```yaml
type: service

image: michaelf34/infinity:latest
env:
  - INFINITY_MODEL_ID=BAAI/bge-small-en-v1.5;BAAI/bge-reranker-base;
  - INFINITY_PORT=80
commands:
  - infinity_emb v2
port: 80
```

Then, simply run the following dstack command. After this, a prompt will appear to let you choose which VM instance to deploy the Infinity.

```shell
dstack run . -f infinity/serve.dstack.yml --gpu 16GB
```

For more detailed tutorial and general information about dstack, visit the [official doc](https://dstack.ai/examples/infinity/#run-the-configuration).


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

