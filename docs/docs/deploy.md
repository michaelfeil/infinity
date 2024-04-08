# Deployment

### Docker: Launch the CLI using a pre-built docker container

Launch the Infinity model using a pre-built Docker container by running the following command. This command uses Docker to run the Infinity CLI with the specified model and port. The optional `HF_HOME` environment variable allows you to control the download path at runtime. 

```bash
model=BAAI/bge-small-en-v1.5
port=7997
docker run \
  -it --gpus all -p $port:$port michaelf34/infinity:latest \
  --model-name-or-path $model --port $port
```

### Docker with offline mode

If you want to run infinity in a location without internet access, you can pre-download the model into the dockerfile.

```bash
# clone the repo
git clone https://github.com/michaelfeil/infinity
git checkout tags/0.0.32
cd libs/infinity_emb
# build download stage using docker buildx buildkit.
docker buildx build --target=production-with-download \
--build-arg MODEL_NAME=michaelfeil/bge-small-en-v1.5 --build-arg ENGINE=torch \
-f Dockerfile -t infinity-model-small .
```
You can also set an argument `EXTRA_PACKAGES` if you require to  `--build-arg EXTRA_PACKAGES="einsum torch_geometric"` 

Rename and push it to your internal docker registry. 

```bash
docker tag infinity-model-small  myregistryhost:5000/myinfinity/infinity:0.0.32-small
docker push myregistryhost:5000/myinfinity/infinity:small-0.0.32
```

Note: You can also save a dockerfile direclty as `.tar`.
This might come in handy if you do not have a shared internal docker registry in your nuclear facility, but still want to leverage the latest semantic search.
https://docs.docker.com/reference/cli/docker/image/save/.

### Extending the Dockerfile

Launching multiple models in one dockerfile
  
Multiple models on one GPU is in experimental mode. You can use the following temporary solution:
```Dockerfile
FROM michaelf34/infinity:latest
# Dockerfile-ENTRYPOINT for multiple models via multiple ports
ENTRYPOINT ["/bin/sh", "-c", \
  "(. /app/.venv/bin/activate && infinity_emb --port 8080 --model-name-or-path sentence-transformers/all-MiniLM-L6-v2 &);\
  (. /app/.venv/bin/activate && infinity_emb --port 8081 --model-name-or-path intfloat/e5-large-v2 )"]
```

You can build and run it via:  
```bash
docker build -t custominfinity . && docker run -it --gpus all -p 8080:8080 -p 8081:8081 custominfinity
```

Both models now run on two instances in one dockerfile servers. Otherwise, you could build your own FastAPI/flask instance, which wraps around the Async API.


### dstack
dstack allows you to provision a VM instance on the cloud of your choice.
Write a service configuration file as below for the deployment of `BAAI/bge-small-en-v1.5` model wrapped in Infinity.

```yaml
type: service

image: michaelf34/infinity:latest
env:
  - MODEL_ID=BAAI/bge-small-en-v1.5
commands:
  - infinity_emb --model-name-or-path $MODEL_ID --port 80
port: 80
```

To deploy the service, execute the following dstack command. A prompt will guide you through selecting the desired VM instance for deploying Infinity.

```shell
dstack run . -f infinity/serve.dstack.yml --gpu 16GB
```

