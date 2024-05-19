# Benchmarking details

Benchmarks are always opinionated. The goal of this benchmark is to find the best possible self-hosted backend for $/token:

1. end-to-end, including the RestAPI server
2. multi-tenant: multiple clients will try to query your server
3. fair batch size: You want to limit request size (sentences per requests) to something low, such that you can load balance requests, scale
4. measured over throughput per token: Idle servers are bad for business. This benchmark is NOT about the latency for a single request against an IDLE server. It partially evaluates the latency under a typical load scenario
5. Bert small / large - the most typical semantic search tasks require a small model (< 1B params)
6. accuracy: each backend must have a ~1e-4 prevision over the torch fp32 embeddings.

## Benchmarking machines:
CPU and NVIDIA:

*  GCP g2-standard-16
*  Intel Cascade Lake
*  1 x NVIDIA L4, cu122

AMD:

*  16 core CPU
*  AMD MI210, rocm5.7 without flash-attn

AWS Inferentia

*  Huggingface AMI (torch-neuronx 1.13, optimum 1.17)
*  inf2.xlarge instance (2 Neuron Cores  with 1 used)

## Reproduction steps:
Install the environment
```bash
pip install "infinity_emb[all]==0.0.25"
```

### sentence-transformers, fastembed, infinity

```bash
git clone https://github.com/michaelfeil/infinity.git
cd infinity
git checkout tags/0.0.25
python ./docs/benchmarks/simple_app.py
```

### huggingface/text-embeddings-inference

using the _cpu_ and _cuda-89_ container (note that cc-89 matches to Nvidia L4)
```bash
docker run -it -p 7997:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-0.6 
--model-id BAAI/bge-small-en-v1.5 --max-client-batch-size 256
```

```bash
docker run -it -p "7997:80" --gpus all --pull always ghcr.io/huggingface/text-embeddings-inference:89-0.6 
--model-id BAAI/bge-large-en-v1.5 --max-client-batch-size 256
```

### tensorrt, onnx-gpu:

```bash
docker buildx build --target production-tensorrt -t inf-trt . && docker run -it -p "7997:7997" --gpus all inf-trt v2 --model-id BAAI/bge-large-en-v1.5 --engine optimum --device "cuda OR tensorrt"
```

## Results

To launch the benchmarks
```bash
make benchmark_embed
```

Below are the following metrics:
*  Requests # / sec (1 request = 256 sentences / 115_000 tokens)
*  time to run benchmark (10 requests = 1_150_000 tokens)

### Results: CPU-only (_BAAI/bge-small-en-v1.5_ | _bert-small_)

| Model                             | Time (seconds) | Requests # / sec (mean) |
|-----------------------------------|----------------|-------------------------|
| infinity-optimum-int8             | 100.490        | 0.10                    |
| infinity-optimum (onnx)           | 125.342        | 0.08                    |
| fastembed (onnx)                  | 125.770        | 0.08                    |
| sentence-transformers (torch)     | 256.884        | 0.04                    |
| infinity (torch)                  | 353.065??      | 0.03 (needs revision)   |
| huggingface/TEI (candle)          | 1104.357       | 0.009                   |



### Results: NVIDIA L4 (_BAAI/bge-large-en-v1.5_ | _bert-large_)

| Model                                        | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| huggingface/TEI (candle, flashbert)         | 0.54                    | 18.491         |
| infinity (torch + compile + fa2)            | 0.51                    | 19.562         |
| tensorrt (via infinity)                     | 0.43                    | 23.367         |
| infinity (onnx-gpu fp16, fused layers)      | 0.41                    | 24.448         |
| sentence-transformers (fp16)                | 0.17                    | 59.107         |


### Results: AMD MI210 (_BAAI/bge-large-en-v1.5_ | _bert-large_)

| Model                                       | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (torch + no compile + fa2 disabled)| 0.75                    | 13.400         |

### Results: AWS INF2 xlarge (_BAAI/bge-large-en-v1.5_ | _bert-large_)

| Model                                       | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (neuron, fp16, constant batch_size 4 / 512 seq)      | 0.11                    | 90.564        |
