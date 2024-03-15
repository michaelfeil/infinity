# Deployment

### Docker: Launch the CLI using a pre-built docker container
```bash
model=BAAI/bge-small-en-v1.5
port=7997
docker run -it --gpus all -p $port:$port michaelf34/infinity:latest --model-name-or-path $model --port $port
```
The download path at runtime, can be controlled via the environment variable `HF_HOME`.

### dstack
dstack allows you to provision a VM instance on the cloud of your choice. Write a service configuration file as below for the deployment of `BAAI/bge-small-en-v1.5` model wrapped in Infinity.

```yaml
type: service

image: michaelf34/infinity:latest
env:
  - MODEL_ID=BAAI/bge-small-en-v1.5
commands:
  - infinity_emb --model-name-or-path $MODEL_ID --port 80
port: 80
```

Then, simply run the following dstack command. After this, a prompt will appear to let you choose which VM instance to deploy the Infinity.

```shell
dstack run . -f infinity/serve.dstack.yml --gpu 16GB
```

