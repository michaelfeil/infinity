# Benchmarking details

This page is a historical benchmark of the legacy `infinity_emb` 0.0.25 release.
Results and commands below are preserved as originally published; they are not
measurements of current Infinity V3, do not evaluate task quality and did not
measure cost.

For the Rust/LibTorch server, see the
[Infinity V3 performance measurements](https://michaelfeil.eu/infinity/main/v3/docs/performance/)
against vLLM and SGLang. Its results use different hardware, software and
workloads from this legacy experiment.

This historical benchmark compared self-hosted embedding backends under a shared
multi-client load:

1. end-to-end, including the REST API server
2. multi-tenant: multiple clients query the server
3. fair input-batch size: limit the number of inputs per HTTP request (sentences per request) so requests can be load-balanced and scaled
4. throughput is reported as HTTP requests/s for the fixed input batch (256 input sentences per request); input tokens/s can be derived by multiplying requests/s by input tokens/request. Idle servers waste capacity, so this benchmark is not single-request latency against an idle server; it partially evaluates latency under typical load
5. checkpoint scope: BERT-small / BERT-large embedding models below 1B parameters in this historical experiment
6. numerical agreement: each backend must agree with Torch FP32 embeddings to approximately 1e-4; this is numerical precision, not task accuracy.

## Benchmark hardware (historical):
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

using the _cpu_ and _cuda-89_ container (note that cc-89 matches the NVIDIA L4)
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
*  HTTP requests/s (mean); one request = 256 input sentences / 115,000 input tokens
*  benchmark duration; 10 requests = 1,150,000 input tokens

### Results: CPU-only (_BAAI/bge-small-en-v1.5_ | _BERT-small_)

| Runtime / engine                  | Time (seconds) | HTTP requests/s (mean) |
|-----------------------------------|----------------|-------------------------|
| infinity-optimum-int8             | 100.490        | 0.10                    |
| infinity-optimum (onnx)           | 125.342        | 0.08                    |
| fastembed (onnx)                  | 125.770        | 0.08                    |
| sentence-transformers (torch)     | 256.884        | 0.04                    |
| infinity (torch)                  | 353.065??      | 0.03 (needs revision)   |
| huggingface/TEI (candle)          | 1104.357       | 0.009                   |



### Results: NVIDIA L4 (_BAAI/bge-large-en-v1.5_ | _BERT-large_)

| Runtime / engine                             | HTTP requests/s (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| huggingface/TEI (candle, flashbert)         | 0.54                    | 18.491         |
| infinity (torch + compile + fa2)            | 0.51                    | 19.562         |
| tensorrt (via infinity)                     | 0.43                    | 23.367         |
| infinity (onnx-gpu fp16, fused layers)      | 0.41                    | 24.448         |
| sentence-transformers (fp16)                | 0.17                    | 59.107         |


### Results: AMD MI210 (_BAAI/bge-large-en-v1.5_ | _BERT-large_)

| Runtime / engine                            | HTTP requests/s (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (torch + no compile + fa2 disabled)| 0.75                    | 13.400         |

### Results: AWS INF2 xlarge (_BAAI/bge-large-en-v1.5_ | _BERT-large_)

| Runtime / engine                            | HTTP requests/s (mean) | Time (seconds) |
|---------------------------------------------|-------------------------|----------------|
| infinity (neuron, fp16, constant batch_size 4 / 512 seq)      | 0.11                    | 90.564        |
