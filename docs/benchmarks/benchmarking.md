#

# Benchmarking details
Benchmarks are always optionated. The goal of this benchmark is to find the best possible self-hosted backend for $/token:
  1. end-to-end, including the RestAPI server
  2. multi-tenant: multiple clients will try to query your server
  3. fair batch size: You want to limit request size (sentences per requests) to something low, such that you can load balance requests, scale
  4. measured over throughput per token: Idle servers are bad for buissness (especially since ). This benchmark is NOT about the latency for a single request against an IDLE server. It partially evaluates the latency under a typical load scenario
  5. Bert Small / large - the most typical semantic search tasks require a small model (< 1B params)
  6. accuracy: each backend must have a ~1e-4 prevision over the torch fp32 embeddings.

# Benchmarking machines:
CPU and NVIDIA:
- GCP g2-standard-16
- Intel Cascade Lake
- 1 x NVIDIA L4, cu122
AMD:
- 16 core CPU
- AMD MI210, rocm5.7 without flash-attn

# Reproduction steps:
Install the environment
```bash
pip install "infinity_emb[all]==0.0.25"
```

### sentence-transformers, fastembed, infinity
```bash
python ./docs/benchmarks/simple_app.py
```

### huggingface/text-embeddings-inference
using the cpu 
```bash
docker run -it -p 7997:80 --pull always ghcr.io/huggingface/text-embeddings-inference:cpu-0.6 
--model-id BAAI/bge-small-en-v1.5 --max-client-batch-size 256
```
and 89 cuda container (note that cc-89 matches to Nvidia L4)
```bash
docker run -it -p 797:80 --gpus all --pull always ghcr.io/huggingface/text-embeddings-inference:89-0.6 
--model-id BAAI/bge-large-en-v1.5 --max-client-batch-size 256
```

### tensorrt, onnx-gpu:
```bash
docker buildx build --target production-tensorrt -t inf-trt . && docker run -it -p "7997:7997" --gpus all inf-trt --model-name-or-path BAAI/bge-large-
en-v1.5 --engine optimum --device "cuda OR tensorrt"
```



# Results

To launch the benchmarks
```
make benchmark_embed
```

Below are the following metrics:
- tl;dr: Requests # / sec (1 requests = 256 sentences with 115000 tokens) and time to run benchmark

### Results: CPU-only (BAAI/bge-SMALL-en-v1.5 / bert-small)

| Model                             | Time (seconds) | Requests # / sec (mean) |
|-----------------------------------|----------------|-------------------------|
| infinity-optimum-int8             | 100.490        | 0.10                    |
| infinity-optimum (onnx)           | 125.342        | 0.08                    |
| fastembed (onnx)                  | 125.770        | 0.08                    |
| sentence-transformers (torch)     | 256.884        | 0.04                    |
| infinity (torch / compile)        | 353.065??      | 0.03???                 |
| huggingface/TEI (candle)          | 1104.357       | 0.009                   |



### Results: NVIDIA L4 (BAAI/bge-LARGE-en-v1.5  / bert-large)

| Model                                        | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| huggingface/TEI (candle, flashbert)         | 0.54                    | 18.491         |
| infinity (torch + compile + fa2)            | 0.51                    | 19.562         |
| tensorrt (via infinity)                     | 0.43                    | 23.367         |
| infinity (onnx-gpu fp16, fused layers)      | 0.41                    | 24.448         |
| sentence-transformers (fp16)                | 0.17                    | 59.107         |


### Results: AMD MI210 NVIDIA L4 (BAAI/bge-LARGE-en-v1.5  / bert-large)

| Model                                       | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (torch + no compile + fa2 disabled)| 0.75                    | 13.400         |

### Results: AWS INF2 xlarge (BAAI/bge-SMALL-en-v1.5  / bert-small)

| Model                                       | Requests # / sec (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (neuron, fp16, constant batch_size 4, 512)      | 0.11                    | 90.564        |
