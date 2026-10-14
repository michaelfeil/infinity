<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CLI arguments

To see all options to serve your models, run the following:

```console
$ infinity_v3 --help
infinity_v3 inference server

Usage: infinity_v3 [OPTIONS] --model-id <MODEL_ID>

Options:
      --model-id <MODEL_ID>
          The Hugging Face model ID, can be any model listed on <https://huggingface.co/models> with the `text-embeddings-inference` tag (meaning it's compatible with Text Embeddings Inference)

          Alternatively, the specified ID can also be a path to a local directory containing the necessary model files saved by the `save_pretrained(...)` methods of either Transformers or Sentence Transformers. The aliases `pplx-decider-v1.1-27b` and `d3` select their native decision protocols.

          [env: INFINITY_MODEL_ID=]

      --served-model-name <SERVED_MODEL_NAME>
          Public model names. Repeat per model; defaults to the checkpoint ID

          [env: INFINITY_SERVED_MODEL_NAME=]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: INFINITY_REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Defaults to CPU cores minus one. Values are clamped to 1–16

          [env: INFINITY_TOKENIZATION_WORKERS=]

      --tokenization-queue-capacity <TOKENIZATION_QUEUE_CAPACITY>
          Maximum waiting tokenizer jobs, independent of worker count. Counts individual inputs, not HTTP requests. When full, producers wait for space

          [env: INFINITY_TOKENIZATION_QUEUE_CAPACITY=]
          [default: 1024]

      --dtype <DTYPE>
          Model dtype. Auto selects bfloat16 from model config when supported, otherwise the backend default

          [env: INFINITY_DTYPE=]
          [default: auto]
          [possible values: auto, float16, float32, bfloat16]

      --backend <BACKEND>
          Inference runtime (requires the matching Cargo feature). Candle is deprecated; use libtorch for supported models

          [env: INFINITY_BACKEND=]
          [default: candle]
          [possible values: candle, libtorch]

      --torch-device <TORCH_DEVICE>
          LibTorch accelerator family. Auto selects CUDA when available, otherwise CPU

          [env: INFINITY_TORCH_DEVICE=]
          [default: auto]
          [possible values: auto, cpu, cuda]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: INFINITY_POOLING=]

      --decision-protocol <DECISION_PROTOCOL>
          Serve a decision model with its trained prompt on /v1/systemone

          [env: INFINITY_DECISION_PROTOCOL=]

      --image-allowed-hosts <IMAGE_ALLOWED_HOSTS>
          Exact HTTPS hostnames allowed for image downloads. Empty permits inline data URLs only

          [env: INFINITY_IMAGE_ALLOWED_HOSTS=]

      --image-memory-budget-mib <IMAGE_MEMORY_BUDGET_MIB>
          Memory budget for encoded images, CPU processing and prepared image patches, in MiB

          [env: INFINITY_IMAGE_MEMORY_BUDGET_MIB=]
          [default: 512]

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: INFINITY_MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, infinity_v3 cannot infer this number automatically.

          [env: INFINITY_MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: INFINITY_MAX_BATCH_REQUESTS=]

      --enable-fp8-dynamic [<ENABLE_FP8_DYNAMIC>]
          Experimental calibration-free FP8 MLPs (Ada/Hopper/Blackwell FP16/BF16 dense decoder and BERT/ModernBERT encoders). Bundled in official SM89/90/100/120 images; disabled by default. May change embedding, reranking, and decision accuracy

          [env: INFINITY_ENABLE_FP8_DYNAMIC=]
          [default: false]
          [possible values: true, false]

      --radix-mlp-threshold <RADIX_MLP_THRESHOLD>
          RadixMLP threshold.

          Set the threshold for RadixMLP. If the compression ratio is lower than the threshold, RadixMLP will be used. The default is 0.92 for most models, and 0.0 (force disabled) for bidirectional models.

          [env: INFINITY_RADIX_MLP_THRESHOLD=]
          [default: 0.92]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: INFINITY_MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --max-decision-questions <MAX_DECISION_QUESTIONS>
          Maximum number of questions in one /v1/systemone request. Independent of max-client-batch-size; model option budgets still apply

          [env: INFINITY_MAX_DECISION_QUESTIONS=]
          [default: 512]

      --auto-truncate [<AUTO_TRUNCATE>]
          Automatically truncate inputs that are longer than the maximum supported size

          [env: INFINITY_AUTO_TRUNCATE=]
          [default: false]
          [possible values: true, false]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: INFINITY_DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: INFINITY_DEFAULT_PROMPT=]

      --dense-path <DENSE_PATH>
          Optionally, define the path to the Dense module required for some embedding models.

          Some embedding models require an extra `Dense` module which contains a single Linear layer and an activation function. By default, those `Dense` modules are stored under the `2_Dense` directory, but there might be cases where different `Dense` modules are provided, to convert the pooled embeddings into different dimensions, available as `2_Dense_<dims>` e.g. https://huggingface.co/NovaSearch/stella_en_400M_v5.

          Note that this argument is optional, only required to be set if there is no `modules.json` file or when you want to override a single Dense module path, only when running with the `candle` backend.

          [env: INFINITY_DENSE_PATH=]

      --device-id <DEVICE_ID>
          Use one explicit accelerator device. CUDA otherwise uses all visible devices

          [env: INFINITY_DEVICE_ID=]

      --backend-device-ids <BACKEND_DEVICE_IDS>
          Visible CUDA ordinals for data-parallel replicas, e.g. 0,1, or auto

          [env: INFINITY_BACKEND_DEVICE_IDS=]

      --hf-token <HF_TOKEN>
          Your Hugging Face Hub token

          [env: INFINITY_HF_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=michaelfeil-dev-pod-h100-0]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: INFINITY_PORT=]
          [default: 3000]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: INFINITY_HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: INFINITY_PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.

          [env: INFINITY_API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: INFINITY_JSON_OUTPUT=]

      --disable-spans
          [env: INFINITY_DISABLE_SPANS=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: INFINITY_OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `infinity_v3.server`

          [env: INFINITY_OTLP_SERVICE_NAME=]
          [default: infinity_v3.server]

      --prometheus-port <PROMETHEUS_PORT>
          The Prometheus port to listen on

          [env: INFINITY_PROMETHEUS_PORT=]
          [default: 9000]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: INFINITY_CORS_ALLOW_ORIGIN=]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

Prompt text preserves literal semicolons. Repeat `--default-prompt` for per-model
prompts. `INFINITY_DEFAULT_PROMPT` accepts a literal prompt or a JSON array such
as `'["Represent this; retrieve it", null]'`; `null` disables a model's prompt.
Use a repeated CLI flag for literal text starting with `[`.
