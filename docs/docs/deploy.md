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

