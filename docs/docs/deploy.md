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

