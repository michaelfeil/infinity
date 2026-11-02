# Running Infinity on AWS Inferentia / Trainium

## Recommended: Use the HuggingFace Neuron AMI (no Docker)

The simplest approach is to run Infinity directly on an EC2 instance with the
HuggingFace Neuron AMI, which comes with `optimum-neuron`, `optimum`, `transformers`,
and `sentence-transformers` pre-installed with compatible Neuron SDK versions.

### 1. Launch an EC2 Instance

- Use the **HuggingFace Neuron AMI** (`huggingface-neuron-*`) from the AWS Marketplace
  - This AMI ships optimum-neuron 0.4.4, neuronx-cc 2.21, Python 3.10 — all compatible
  - Search for `huggingface-neuron` in the EC2 AMI catalog
- Instance type: **inf2.xlarge** (2 NeuronCores, 32 GB), **trn2.3xlarge** (4 NeuronCores, 128 GB), or larger
- Disk: The AMI defaults to 512 GB

### 2. Install Infinity

```bash
# SSH into the instance
ssh ubuntu@<your-instance-ip>

# Activate the pre-installed PyTorch environment
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate

# Clone and install Infinity from source (don't overwrite Neuron packages)
git clone https://github.com/michaelfeil/infinity.git ~/infinity
cd ~/infinity/libs/infinity_emb
pip install --no-deps .

# Install remaining runtime dependencies (most are already present on the HF AMI)
pip install uvicorn fastapi orjson typer httptools pydantic posthog \
    prometheus-fastapi-instrumentator hf_transfer rich
```

### 3. Run Infinity with Neuron engine

```bash
# Single core (uses one NeuronCore)
infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4
```

The first run will compile the model for Neuron (~100 seconds). Subsequent runs use the cached compilation.

### 4. Scale across all NeuronCores (data parallelism)

The Neuron runtime is limited to one model per process. To use all NeuronCores,
run one server process per core, each pinned to a different core:

```bash
# inf2.xlarge has 2 NeuronCores (cores 0 and 1)
NEURON_RT_VISIBLE_CORES=0 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 7997 &
NEURON_RT_VISIBLE_CORES=1 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 7998 &

# trn2.3xlarge has 4 NeuronCores (cores 0-3)
NEURON_RT_VISIBLE_CORES=0 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 7997 &
NEURON_RT_VISIBLE_CORES=1 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 7998 &
NEURON_RT_VISIBLE_CORES=2 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 7999 &
NEURON_RT_VISIBLE_CORES=3 infinity_emb v2 --engine neuron --model-id BAAI/bge-small-en-v1.5 --batch-size 4 --port 8000 &
```

Then use a load balancer (nginx, HAProxy, etc.) to distribute requests across
ports. Throughput scales linearly with cores: 2 cores = 2x, 4 cores = 4x.

### 5. Test it

```bash
curl http://localhost:7997/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["Hello world", "How are you?"], "model": "BAAI/bge-small-en-v1.5"}'
```

## Performance (bge-small-en-v1.5, batch_size=4)

### Latency (serial requests, P50)

| Workload | g5.xlarge (GPU) | inf2.xlarge (1 core) | trn2.3xlarge (1 core) |
|----------|----------------|---------------------|----------------------|
| 1 short sentence | 14.2ms | 25.0ms | 19.0ms |
| 4 short sentences | 16.0ms | 25.6ms | 19.5ms |
| 4 long sentences | 16.2ms | 26.0ms | 20.3ms |

### Throughput (concurrent requests, data parallelism)

| Instance | Cores | Peak emb/s | Concurrency |
|----------|-------|-----------|-------------|
| g5.xlarge (GPU) | 1 GPU | 536 | 8 |
| inf2.xlarge | 1 core | 216 | 4 |
| inf2.xlarge | 2 cores | 427 | 4 |
| trn2.3xlarge | 1 core | 348 | 4 |
| trn2.3xlarge | 4 cores | 753 | 4 |

**Notes:**
- g5.xlarge uses `--engine torch`; inf2/trn2 use `--engine neuron`
- Neuron latency is constant regardless of batch content (padded to compiled batch size)
- trn2 has ~30% lower latency per core than inf2 (19ms vs 25ms)
- Throughput scales linearly with data parallelism (1 process per core)
- Compilation time: ~60-100 seconds on first run (cached after that)

Tested on HuggingFace Neuron AMI (optimum-neuron 0.4.4, neuronx-cc 2.21, SDK 2.27)
and Deep Learning AMI Neuron Ubuntu 22.04 (SDK 2.28) for trn2.

## Tested Stack

| Package | Version |
|---------|---------|
| optimum-neuron | 0.4.4 |
| optimum | 2.0.0 |
| neuronx-cc | 2.21.33363 |
| torch-neuronx | 2.8.0.2.10 |
| torch | 2.8.0 |
| transformers | 4.57.3 |
| Python | 3.10.12 |

## Alternative: Docker

### Build from source

```bash
git clone https://github.com/michaelfeil/infinity
cd infinity
docker buildx build -t infinity-neuron -f ./infra/aws_neuron/Dockerfile.neuron .
```

### Run on EC2

```bash
docker run -it --rm --device=/dev/neuron0 infinity-neuron \
  v2 --model-id BAAI/bge-small-en-v1.5 --batch-size 8
```

**Note:** The host must have the Neuron driver installed. The Docker approach is less tested than the direct AMI approach above.

## Limitations

- The `--engine neuron` flag currently supports **text embeddings only** (no reranking or classification)
- The Neuron engine requires a **constant batch size** (requests are padded automatically)
- Models are compiled on first use; compilation can take 60-120 seconds

## ECS Deployment

See the ECS task definition example below for container orchestration:

```json
{
    "family": "ecs-infinity-neuron",
    "requiresCompatibilities": ["EC2"],
    "placementConstraints": [
        {
            "type": "memberOf",
            "expression": "attribute:ecs.os-type == linux"
        },
        {
            "type": "memberOf",
            "expression": "attribute:ecs.instance-type == inf2.xlarge"
        }
    ],
    "executionRoleArn": "${YOUR_EXECUTION_ROLE}",
    "containerDefinitions": [
        {
            "entryPoint": ["infinity_emb", "v2"],
            "portMappings": [
                {
                    "hostPort": 7997,
                    "protocol": "tcp",
                    "containerPort": 7997
                }
            ],
            "linuxParameters": {
                "devices": [
                    {
                        "containerPath": "/dev/neuron0",
                        "hostPath": "/dev/neuron0",
                        "permissions": ["read", "write"]
                    }
                ],
                "capabilities": {
                    "add": ["IPC_LOCK"]
                }
            },
            "cpu": 0,
            "memoryReservation": 1000,
            "image": "infinity-neuron:latest",
            "essential": true,
            "name": "infinity-neuron"
        }
    ]
}
```
